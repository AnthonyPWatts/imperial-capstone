"""Evaluate one fold-safe transductive pseudo-labelling policy."""

from __future__ import annotations

import time as _time
from typing import Callable as _Callable

import numpy as _np
import pandas as _pd

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
PSEUDO_LABEL_CONFIDENCE = 0.98


def select_high_confidence_pseudo_labels(
    probabilities: _np.ndarray,
    *,
    confidence: float = PSEUDO_LABEL_CONFIDENCE,
) -> tuple[_np.ndarray, _np.ndarray, _np.ndarray]:
    """Return positions, hard labels and confidence for the fixed policy."""

    if confidence <= 1 / len(CLASS_LABELS) or confidence > 1:
        raise ValueError("Pseudo-label confidence is outside the valid range.")
    expected_shape = (len(probabilities), len(CLASS_LABELS))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected pseudo-label probability shape {expected_shape}, "
            f"found {probabilities.shape}."
        )
    if not _np.isfinite(probabilities).all():
        raise ValueError("Pseudo-label probabilities are not all finite.")
    if not _np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Pseudo-label probabilities do not sum to one.")
    maximum = probabilities.max(axis=1)
    positions = _np.flatnonzero(maximum >= confidence)
    labels = _np.asarray(CLASS_LABELS)[probabilities[positions].argmax(axis=1)]
    return positions, labels, maximum[positions]


def append_pseudo_labelled_rows(
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_unlabelled: _pd.DataFrame,
    positions: _np.ndarray,
    labels: _np.ndarray,
) -> tuple[_pd.DataFrame, _pd.Series]:
    """Append selected unlabelled rows without mutating either source frame."""

    if len(positions) != len(labels):
        raise ValueError("Pseudo-label positions and labels have different lengths.")
    if len(positions) == 0:
        raise ValueError("The fixed pseudo-label policy selected no rows.")
    if positions.min() < 0 or positions.max() >= len(X_unlabelled):
        raise IndexError("A pseudo-label position is outside the unlabelled frame.")
    X_augmented = _pd.concat(
        [X_training, X_unlabelled.iloc[positions]],
        ignore_index=True,
    )
    y_augmented = _pd.concat(
        [y_training, _pd.Series(labels, dtype=y_training.dtype)],
        ignore_index=True,
    )
    return X_augmented, y_augmented


def evaluate_transductive_pseudo_xgboost(
    partitioned_data: PartitionedData,
    X_competition: _pd.DataFrame,
    accepted_xgboost: CandidateEvaluation,
    cross_validation: object,
    *,
    gpu_fitter: _Callable = fit_gpu_candidate_probabilities,
) -> CandidateEvaluation:
    """Teach on outer training, pseudo-label competition, then refit and score."""

    if (
        accepted_xgboost.cross_validation_fingerprint
        != partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError("Pseudo-label teacher uses different frozen folds.")
    if tuple(X_competition.columns) != tuple(partitioned_data.X_development.columns):
        raise ValueError("Competition and development features do not align.")
    probability_values = _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostics = []
    spec = make_xgboost_spec(variant="depth 8 child 1")
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        iterations = int(
            accepted_xgboost.diagnostics.loc[
                fold_number,
                "selected_iterations",
            ]
        )
        fold_started = _time.perf_counter()
        teacher_probabilities, teacher_seconds = gpu_fitter(
            spec,
            X_training,
            y_training,
            X_competition,
            iterations=iterations,
        )
        positions, pseudo_labels, confidence = (
            select_high_confidence_pseudo_labels(teacher_probabilities)
        )
        X_augmented, y_augmented = append_pseudo_labelled_rows(
            X_training,
            y_training,
            X_competition,
            positions,
            pseudo_labels,
        )
        validation_probabilities, student_seconds = gpu_fitter(
            spec,
            X_augmented,
            y_augmented,
            X_validation,
            iterations=iterations,
        )
        probability_values[validation_positions] = validation_probabilities
        shares = _pd.Series(pseudo_labels).value_counts()
        diagnostics.append(
            {
                "validation_fold": fold_number,
                "selected_iterations": iterations,
                "pseudo_rows": len(positions),
                "pseudo_functional": int(shares.get("functional", 0)),
                "pseudo_repair": int(
                    shares.get("functional needs repair", 0)
                ),
                "pseudo_non_functional": int(
                    shares.get("non functional", 0)
                ),
                "mean_pseudo_confidence": float(confidence.mean()),
                "teacher_seconds": teacher_seconds,
                "student_seconds": student_seconds,
                "total_seconds": _time.perf_counter() - fold_started,
            }
        )
        print(
            f"Completed pseudo-label fold {fold_number}/{CROSS_VALIDATION_FOLDS}: "
            f"{len(positions):,} competition rows in "
            f"{diagnostics[-1]['total_seconds']:.1f} seconds.",
            flush=True,
        )
    return build_candidate_evaluation(
        model_name="transductive high-confidence pseudo-label XGBoost",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )

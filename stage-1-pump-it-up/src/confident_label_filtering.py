"""Evaluate nested confident-disagreement filtering for likely label errors."""

from __future__ import annotations

import time as _time
from typing import Callable as _Callable

import numpy as _np
import pandas as _pd
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold

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
CLASS_TO_INTEGER = {
    label: position for position, label in enumerate(CLASS_LABELS)
}
INNER_FOLDS = 3
INNER_SEED = 20260822
ALTERNATIVE_CONFIDENCE = 0.90
OBSERVED_LABEL_MAXIMUM = 0.05


def select_confident_label_disagreements(
    y_observed: _pd.Series,
    probabilities: _np.ndarray,
) -> _np.ndarray:
    """Return rows whose nested prediction strongly contradicts the label."""

    expected_shape = (len(y_observed), len(CLASS_LABELS))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected disagreement probability shape {expected_shape}, "
            f"found {probabilities.shape}."
        )
    if not _np.isfinite(probabilities).all():
        raise ValueError("Disagreement probabilities are not all finite.")
    if not _np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Disagreement probabilities do not sum to one.")
    try:
        observed_positions = _np.asarray(
            [CLASS_TO_INTEGER[label] for label in y_observed]
        )
    except KeyError as error:
        raise ValueError(f"Unexpected observed label: {error.args[0]!r}.") from error
    predicted_positions = probabilities.argmax(axis=1)
    alternative_confidence = probabilities.max(axis=1)
    observed_confidence = probabilities[
        _np.arange(len(probabilities)),
        observed_positions,
    ]
    return (
        (predicted_positions != observed_positions)
        & (alternative_confidence >= ALTERNATIVE_CONFIDENCE)
        & (observed_confidence <= OBSERVED_LABEL_MAXIMUM)
    )


def evaluate_confident_filter_xgboost(
    partitioned_data: PartitionedData,
    accepted_xgboost: CandidateEvaluation,
    cross_validation: object,
    *,
    gpu_fitter: _Callable = fit_gpu_candidate_probabilities,
) -> CandidateEvaluation:
    """Cross-fit disagreement evidence inside each outer-training partition."""

    if (
        accepted_xgboost.cross_validation_fingerprint
        != partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError("Confident-filter teacher uses different frozen folds.")
    probability_values = _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostic_rows = []
    spec = make_xgboost_spec(variant="depth 8 child 1")
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        fold_started = _time.perf_counter()
        X_training = partitioned_data.X_development.iloc[
            training_positions
        ].reset_index(drop=True)
        y_training = partitioned_data.y_development.iloc[
            training_positions
        ].reset_index(drop=True)
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        iterations = int(
            accepted_xgboost.diagnostics.loc[
                fold_number,
                "selected_iterations",
            ]
        )
        nested_probabilities = _cross_fit_teacher_probabilities(
            X_training,
            y_training,
            iterations=iterations,
            spec=spec,
            gpu_fitter=gpu_fitter,
        )
        remove = select_confident_label_disagreements(
            y_training,
            nested_probabilities,
        )
        if not remove.any():
            raise ValueError("The confident-disagreement filter selected no rows.")
        X_filtered = X_training.loc[~remove].reset_index(drop=True)
        y_filtered = y_training.loc[~remove].reset_index(drop=True)
        validation_probabilities, student_seconds = gpu_fitter(
            spec,
            X_filtered,
            y_filtered,
            X_validation,
            iterations=iterations,
        )
        probability_values[validation_positions] = validation_probabilities
        removed_labels = y_training.loc[remove].value_counts()
        predicted_labels = _np.asarray(CLASS_LABELS)[
            nested_probabilities[remove].argmax(axis=1)
        ]
        predicted_counts = _pd.Series(predicted_labels).value_counts()
        diagnostic_rows.append(
            {
                "validation_fold": fold_number,
                "selected_iterations": iterations,
                "removed_rows": int(remove.sum()),
                "removed_fraction": float(remove.mean()),
                "removed_observed_functional": int(
                    removed_labels.get("functional", 0)
                ),
                "removed_observed_repair": int(
                    removed_labels.get("functional needs repair", 0)
                ),
                "removed_observed_non_functional": int(
                    removed_labels.get("non functional", 0)
                ),
                "predicted_functional": int(
                    predicted_counts.get("functional", 0)
                ),
                "predicted_repair": int(
                    predicted_counts.get("functional needs repair", 0)
                ),
                "predicted_non_functional": int(
                    predicted_counts.get("non functional", 0)
                ),
                "student_seconds": student_seconds,
                "total_seconds": _time.perf_counter() - fold_started,
            }
        )
        print(
            f"Completed confident-filter fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS}: removed {int(remove.sum()):,} rows in "
            f"{diagnostic_rows[-1]['total_seconds']:.1f} seconds.",
            flush=True,
        )
    return build_candidate_evaluation(
        model_name="nested confident-disagreement filtered XGBoost",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostic_rows,
    )


def _cross_fit_teacher_probabilities(
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    *,
    iterations: int,
    spec: object,
    gpu_fitter: _Callable,
) -> _np.ndarray:
    splitter = _StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=INNER_SEED,
    )
    probabilities = _np.full(
        (len(y_training), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    for inner_training, inner_validation in splitter.split(X_training, y_training):
        fold_probabilities, _ = gpu_fitter(
            spec,
            X_training.iloc[inner_training],
            y_training.iloc[inner_training],
            X_training.iloc[inner_validation],
            iterations=iterations,
        )
        probabilities[inner_validation] = fold_probabilities
    if not _np.isfinite(probabilities).all():
        raise ValueError("Nested teacher did not predict every training row.")
    return probabilities

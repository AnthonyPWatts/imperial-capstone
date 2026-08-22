"""Evaluate two cumulative accepted-spec XGBoost class boundaries."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import time as _time

import numpy as _np

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from gpu_model_evaluation import CLASS_LABELS
from gpu_model_evaluation import INNER_STOP_SEED
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from one_vs_rest_evaluation import fit_binary_xgboost_fold


FUNCTIONAL_LABEL = "functional"
NON_FUNCTIONAL_LABEL = "non functional"


@_dataclass(frozen=True)
class OrdinalXGBoostTrial:
    """OOF evidence for two fixed cumulative XGBoost boundaries."""

    evaluation: CandidateEvaluation
    class_labels: tuple[str, ...]


def reconstruct_cumulative_memberships(
    repair_or_worse: _np.ndarray,
    non_functional: _np.ndarray,
) -> _np.ndarray:
    """Project cumulative probabilities to coherence and recover classes."""

    first = _np.asarray(repair_or_worse, dtype="float64")
    second = _np.asarray(non_functional, dtype="float64")
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("Cumulative boundaries require aligned one-dimensional rows.")
    if (
        not _np.isfinite(first).all()
        or not _np.isfinite(second).all()
        or (first < 0).any()
        or (first > 1).any()
        or (second < 0).any()
        or (second > 1).any()
    ):
        raise ValueError("Cumulative probabilities must be finite within zero to one.")
    crossing = second > first
    midpoint = (first + second) / 2.0
    coherent_first = _np.where(crossing, midpoint, first)
    coherent_second = _np.where(crossing, midpoint, second)
    memberships = _np.column_stack(
        (
            1.0 - coherent_first,
            coherent_first - coherent_second,
            coherent_second,
        )
    )
    if (memberships < -1e-12).any() or not _np.allclose(
        memberships.sum(axis=1),
        1.0,
        atol=1e-10,
    ):
        raise ValueError("Reconstructed ordinal memberships are invalid.")
    return _np.clip(memberships, 0.0, 1.0)


def evaluate_ordinal_xgboost(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> OrdinalXGBoostTrial:
    """Fit the two cumulative boundaries inside every outer fold."""

    probability_values = _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostics = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        fold_started = _time.perf_counter()
        first, first_iterations = fit_binary_xgboost_fold(
            X_training,
            y_training.ne(FUNCTIONAL_LABEL).astype("int8"),
            X_validation,
            split_seed=INNER_STOP_SEED + 10 + fold_number,
        )
        second, second_iterations = fit_binary_xgboost_fold(
            X_training,
            y_training.eq(NON_FUNCTIONAL_LABEL).astype("int8"),
            X_validation,
            split_seed=INNER_STOP_SEED + 20 + fold_number,
        )
        crossing_share = float(_np.mean(second > first))
        probability_values[validation_positions] = reconstruct_cumulative_memberships(
            first,
            second,
        )
        diagnostics.append(
            {
                "validation_fold": fold_number,
                "inner_training_rows": len(training_positions),
                "validation_rows": len(validation_positions),
                "repair_or_worse_selected_iterations": first_iterations,
                "non_functional_selected_iterations": second_iterations,
                "crossing_share": crossing_share,
                "total_seconds": _time.perf_counter() - fold_started,
            }
        )
        print(
            f"Completed cumulative XGBoost fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS} in "
            f"{diagnostics[-1]['total_seconds']:.1f} seconds."
        )
    evaluation = build_candidate_evaluation(
        model_name="Cumulative ordinal XGBoost depth 8 child 1",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )
    return OrdinalXGBoostTrial(evaluation=evaluation, class_labels=CLASS_LABELS)

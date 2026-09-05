"""Confirm the two-seed deep bag inside the exact frozen archive ensemble."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd

from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from deep_xgboost_variance_hedge import build_variance_hedge_probabilities
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from locked_screen_evidence import recompute_and_validate_oof_evaluation
from model_evaluation import build_candidate_evaluation
from model_evaluation import CandidateEvaluation


CONFIRMATION_CACHE_VERSION = 1
ENSEMBLE_MEAN_ACCURACY_GAIN_GATE = 0.0005
ENSEMBLE_FOLD_WIN_GATE = 3
ENSEMBLE_WORST_FOLD_DELTA_GATE = -0.001
ENSEMBLE_REPAIR_RECALL_DELTA_GATE = -0.01
REPAIR_METRIC = "recall: functional needs repair"
CANDIDATE_NAME = "50:50 seed-20260824/seed-20260905 deep-slot probability bag"
INCUMBENT_NAME = "seed-20260824 deep archive"


@_dataclass(frozen=True)
class DeepFrozenEnsembleComparison:
    """Replayed incumbent and one exact deep-slot challenger."""

    deep_seed_average: CandidateEvaluation
    incumbent: CandidateEvaluation
    candidate: CandidateEvaluation
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame


def compare_deep_seed_average_ensemble(
    partitioned_data: PartitionedData,
    incumbent_components: _Mapping[str, _np.ndarray],
    seed_20260824: CandidateEvaluation,
    seed_20260905: CandidateEvaluation,
) -> DeepFrozenEnsembleComparison:
    """Score only the predeclared average at the unchanged 0.33 slot."""

    cross_validation = make_cross_validation(partitioned_data)
    seed_24 = recompute_and_validate_oof_evaluation(
        seed_20260824,
        partitioned_data,
        cross_validation,
        candidate="seed-20260824 deep-XGBoost",
    )
    seed_05 = recompute_and_validate_oof_evaluation(
        seed_20260905,
        partitioned_data,
        cross_validation,
        candidate="seed-20260905 deep-XGBoost",
    )
    incumbent_deep = _np.asarray(incumbent_components.get("deep_xgboost"))
    seed_24_probabilities = seed_24.out_of_fold_probabilities.to_numpy(
        dtype="float64"
    )
    if not _np.array_equal(incumbent_deep, seed_24_probabilities):
        raise ValueError(
            "The incumbent deep slot is not the pinned seed-20260824 OOF evidence."
        )
    hedge = build_variance_hedge_probabilities(
        incumbent_components,
        seed_24_probabilities,
        seed_05.out_of_fold_probabilities.to_numpy(dtype="float64"),
    )
    deep_seed_average = _build_evaluation(
        "50:50 seed-20260824/seed-20260905 deep-XGBoost component",
        partitioned_data,
        cross_validation,
        hedge.deep_seed_average,
    )
    incumbent = _build_evaluation(
        INCUMBENT_NAME,
        partitioned_data,
        cross_validation,
        hedge.incumbent_ensemble,
    )
    candidate = _build_evaluation(
        CANDIDATE_NAME,
        partitioned_data,
        cross_validation,
        hedge.candidate_ensemble,
    )
    folds = _paired_fold_deltas(partitioned_data, incumbent, candidate)
    summary = _summarise(incumbent, candidate, folds)
    return DeepFrozenEnsembleComparison(
        deep_seed_average=deep_seed_average,
        incumbent=incumbent,
        candidate=candidate,
        summary=summary,
        fold_deltas=folds,
    )


def passes_deep_frozen_ensemble_gate(
    *,
    mean_accuracy_delta: float,
    fold_wins: int,
    worst_fold_delta: float,
    repair_recall_delta: float,
) -> bool:
    """Apply the locked exact-ensemble gate used by the RF confirmation."""

    return (
        _at_least(mean_accuracy_delta, ENSEMBLE_MEAN_ACCURACY_GAIN_GATE)
        and fold_wins >= ENSEMBLE_FOLD_WIN_GATE
        and _at_least(worst_fold_delta, ENSEMBLE_WORST_FOLD_DELTA_GATE)
        and _at_least(
            repair_recall_delta,
            ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
        )
    )


def validate_seed_fold_cache(
    payload: object,
    expected_metadata: _Mapping[str, object],
    expected_validation_ids: _np.ndarray,
) -> dict[str, object]:
    """Validate one resumable seed-20260905 frozen-fold cache."""

    required = {
        "cache_version",
        "metadata",
        "validation_ids",
        "probabilities",
        "fit_and_predict_seconds",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Seed-20260905 fold cache schema changed.")
    if payload["cache_version"] != CONFIRMATION_CACHE_VERSION:
        raise ValueError("Seed-20260905 fold cache version changed.")
    if payload["metadata"] != dict(expected_metadata):
        raise ValueError("Seed-20260905 fold cache metadata is stale.")
    if not _np.array_equal(payload["validation_ids"], expected_validation_ids):
        raise ValueError("Seed-20260905 fold validation IDs are misaligned.")
    validate_probabilities(
        payload["probabilities"],
        len(expected_validation_ids),
    )
    seconds = payload["fit_and_predict_seconds"]
    if (
        not isinstance(seconds, (int, float))
        or not _np.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("Seed-20260905 fold timing is invalid.")
    return payload


def validate_seed_evaluation_cache(
    payload: object,
    expected_metadata: _Mapping[str, object],
    partitioned_data: PartitionedData,
) -> CandidateEvaluation:
    """Validate aggregate cache metadata and replay all OOF metrics."""

    required = {"cache_version", "metadata", "evaluation"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Seed-20260905 aggregate cache schema changed.")
    if payload["cache_version"] != CONFIRMATION_CACHE_VERSION:
        raise ValueError("Seed-20260905 aggregate cache version changed.")
    if payload["metadata"] != dict(expected_metadata):
        raise ValueError("Seed-20260905 aggregate cache metadata is stale.")
    evaluation = payload["evaluation"]
    if not isinstance(evaluation, CandidateEvaluation):
        raise ValueError("Seed-20260905 aggregate cache has no evaluation.")
    return recompute_and_validate_oof_evaluation(
        evaluation,
        partitioned_data,
        make_cross_validation(partitioned_data),
        candidate="seed-20260905 deep-XGBoost aggregate",
    )


def validate_deep_average_component_cache(
    payload: object,
    expected_metadata: _Mapping[str, object],
    partitioned_data: PartitionedData,
) -> CandidateEvaluation:
    """Validate the cache exposed for later cache-only factorial checks."""

    required = {"cache_version", "metadata", "evaluation"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Deep seed-average component cache schema changed.")
    if payload["cache_version"] != CONFIRMATION_CACHE_VERSION:
        raise ValueError("Deep seed-average component cache version changed.")
    if payload["metadata"] != dict(expected_metadata):
        raise ValueError("Deep seed-average component cache metadata is stale.")
    evaluation = payload["evaluation"]
    if not isinstance(evaluation, CandidateEvaluation):
        raise ValueError("Deep seed-average component cache has no evaluation.")
    return recompute_and_validate_oof_evaluation(
        evaluation,
        partitioned_data,
        make_cross_validation(partitioned_data),
        candidate="50:50 deep seed-average component",
    )


def _build_evaluation(
    name: str,
    partitioned_data: PartitionedData,
    cross_validation: object,
    probabilities: _np.ndarray,
) -> CandidateEvaluation:
    validate_probabilities(probabilities, len(partitioned_data.y_development))
    return build_candidate_evaluation(
        model_name=name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=[
            {"validation_fold": fold}
            for fold in range(1, 6)
        ],
    )


def _paired_fold_deltas(
    partitioned_data: PartitionedData,
    incumbent: CandidateEvaluation,
    candidate: CandidateEvaluation,
) -> _pd.DataFrame:
    incumbent_predictions = incumbent.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    candidate_predictions = candidate.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    rows = []
    for fold, (_, positions) in enumerate(
        make_cross_validation(partitioned_data).split(),
        start=1,
    ):
        actual = partitioned_data.y_development.iloc[positions].to_numpy()
        incumbent_correct = incumbent_predictions[positions] == actual
        candidate_correct = candidate_predictions[positions] == actual
        gained = int((candidate_correct & ~incumbent_correct).sum())
        lost = int((~candidate_correct & incumbent_correct).sum())
        rows.append(
            {
                "validation_fold": fold,
                "rows": len(positions),
                "incumbent_accuracy": float(incumbent_correct.mean()),
                "candidate_accuracy": float(candidate_correct.mean()),
                "accuracy_delta": float(
                    candidate_correct.mean() - incumbent_correct.mean()
                ),
                "incumbent_repair_recall": float(
                    incumbent.fold_metrics.loc[fold, REPAIR_METRIC]
                ),
                "candidate_repair_recall": float(
                    candidate.fold_metrics.loc[fold, REPAIR_METRIC]
                ),
                "repair_recall_delta": float(
                    candidate.fold_metrics.loc[fold, REPAIR_METRIC]
                    - incumbent.fold_metrics.loc[fold, REPAIR_METRIC]
                ),
                "prediction_disagreements": int(
                    (candidate_predictions[positions]
                     != incumbent_predictions[positions]).sum()
                ),
                "gained_correct": gained,
                "lost_correct": lost,
                "net_additional_correct": gained - lost,
            }
        )
    return _pd.DataFrame(rows)


def _summarise(
    incumbent: CandidateEvaluation,
    candidate: CandidateEvaluation,
    folds: _pd.DataFrame,
) -> _pd.DataFrame:
    incumbent_accuracy = float(
        incumbent.metric_summary.loc["accuracy", "mean"]
    )
    candidate_accuracy = float(
        candidate.metric_summary.loc["accuracy", "mean"]
    )
    repair_delta = float(
        candidate.metric_summary.loc[REPAIR_METRIC, "mean"]
        - incumbent.metric_summary.loc[REPAIR_METRIC, "mean"]
    )
    mean_delta = candidate_accuracy - incumbent_accuracy
    fold_wins = int(folds["accuracy_delta"].gt(0).sum())
    worst_fold_delta = float(folds["accuracy_delta"].min())
    row = {
        "candidate": "deep_seed_probability_average_at_0.33",
        "incumbent_mean_accuracy": incumbent_accuracy,
        "candidate_mean_accuracy": candidate_accuracy,
        "mean_accuracy_delta": mean_delta,
        "fold_wins": fold_wins,
        "fold_losses": int(folds["accuracy_delta"].lt(0).sum()),
        "worst_fold_delta": worst_fold_delta,
        "best_fold_delta": float(folds["accuracy_delta"].max()),
        "incumbent_repair_recall": float(
            incumbent.metric_summary.loc[REPAIR_METRIC, "mean"]
        ),
        "candidate_repair_recall": float(
            candidate.metric_summary.loc[REPAIR_METRIC, "mean"]
        ),
        "repair_recall_delta": repair_delta,
        "prediction_disagreements": int(
            folds["prediction_disagreements"].sum()
        ),
        "gained_correct": int(folds["gained_correct"].sum()),
        "lost_correct": int(folds["lost_correct"].sum()),
        "net_additional_correct": int(
            folds["net_additional_correct"].sum()
        ),
    }
    row["passes_gate"] = passes_deep_frozen_ensemble_gate(
        mean_accuracy_delta=mean_delta,
        fold_wins=fold_wins,
        worst_fold_delta=worst_fold_delta,
        repair_recall_delta=repair_delta,
    )
    return _pd.DataFrame([row]).set_index("candidate")


def _at_least(actual: float, threshold: float) -> bool:
    return float(actual) >= threshold - 1e-12

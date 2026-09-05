"""Replay one fixed complete-identity/spatial-grid CatBoost probability bag."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

import numpy as _np
import pandas as _pd

from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from final_model import validate_probabilities
from catboost_identity_evaluation import CatBoostIdentityTrial
from deep_archive_confirmation import blend_deep_archive
from locked_architecture_candidates import LockedReplacementDecision
from locked_architecture_candidates import SPATIAL_SUBSTITUTION
from locked_architecture_candidates import build_locked_substitution_probabilities
from locked_rf_ensemble_confirmation import passes_locked_rf_ensemble_gate
from locked_screen_evidence import recompute_and_validate_oof_evaluation
from model_evaluation import build_candidate_evaluation
from model_evaluation import CandidateEvaluation
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_ACCURACY
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_FOLD_WINS
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_WORST_FOLD


CATBOOST_BAG_CANDIDATE = "complete_identity_spatial_grid_50_50_bag"
CATBOOST_OLD_FOLD_CACHE_VERSION = 1


@_dataclass(frozen=True)
class FreshCatBoostBagComparison:
    """Fresh-fold metrics for the single predeclared probability bag."""

    identity_evaluation: CandidateEvaluation
    bag_evaluation: CandidateEvaluation
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame


@_dataclass(frozen=True)
class LockedCatBoostBagEnsembleComparison:
    """Old-fold evidence for the fixed bag inside the locked 20% slot."""

    incumbent_evaluation: CandidateEvaluation
    candidate_evaluation: CandidateEvaluation
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame


def make_equal_catboost_probability_bag(
    complete_identity: _np.ndarray,
    spatial_grid: _np.ndarray,
) -> _np.ndarray:
    """Return the sole locked 50:50 probability bag."""

    rows = len(complete_identity)
    validate_probabilities(complete_identity, rows)
    validate_probabilities(spatial_grid, rows)
    bag = 0.5 * complete_identity + 0.5 * spatial_grid
    validate_probabilities(bag, rows)
    return bag


def compare_fresh_catboost_probability_bag(
    partitioned_data: PartitionedData,
    complete_identity: CandidateEvaluation,
    spatial_grid: CandidateEvaluation,
) -> FreshCatBoostBagComparison:
    """Recompute the fixed bag and apply the unchanged fresh spatial gate."""

    cross_validation = make_cross_validation(partitioned_data)
    identity = recompute_and_validate_oof_evaluation(
        complete_identity,
        partitioned_data,
        cross_validation,
        candidate="complete_identity_catboost",
    )
    grid = recompute_and_validate_oof_evaluation(
        spatial_grid,
        partitioned_data,
        cross_validation,
        candidate="spatial_grid_catboost",
    )
    bag_probabilities = make_equal_catboost_probability_bag(
        identity.out_of_fold_probabilities.to_numpy(),
        grid.out_of_fold_probabilities.to_numpy(),
    )
    bag = build_candidate_evaluation(
        model_name=CATBOOST_BAG_CANDIDATE,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=bag_probabilities,
        diagnostic_rows=[
            {"validation_fold": fold_number}
            for fold_number in range(1, 6)
        ],
    )
    folds = _paired_folds(partitioned_data, identity, bag)
    accuracy_delta = float(
        bag.metric_summary.loc["accuracy", "mean"]
        - identity.metric_summary.loc["accuracy", "mean"]
    )
    repair_metric = "recall: functional needs repair"
    repair_delta = float(
        bag.metric_summary.loc[repair_metric, "mean"]
        - identity.metric_summary.loc[repair_metric, "mean"]
    )
    fold_wins = int(folds["accuracy_delta"].gt(0).sum())
    worst_fold = float(folds["accuracy_delta"].min())
    summary = _pd.DataFrame(
        [
            {
                "candidate": CATBOOST_BAG_CANDIDATE,
                "identity_mean_accuracy": float(
                    identity.metric_summary.loc["accuracy", "mean"]
                ),
                "bag_mean_accuracy": float(
                    bag.metric_summary.loc["accuracy", "mean"]
                ),
                "mean_accuracy_delta": accuracy_delta,
                "fold_wins": fold_wins,
                "worst_fold_delta": worst_fold,
                "identity_repair_recall": float(
                    identity.metric_summary.loc[repair_metric, "mean"]
                ),
                "bag_repair_recall": float(
                    bag.metric_summary.loc[repair_metric, "mean"]
                ),
                "repair_recall_delta": repair_delta,
                "net_additional_correct": int(
                    folds["net_additional_correct"].sum()
                ),
                "passes_gate": passes_fresh_catboost_bag_gate(
                    mean_accuracy_delta=accuracy_delta,
                    fold_wins=fold_wins,
                    worst_fold_delta=worst_fold,
                ),
            }
        ]
    ).set_index("candidate")
    return FreshCatBoostBagComparison(
        identity_evaluation=identity,
        bag_evaluation=bag,
        summary=summary,
        fold_deltas=folds,
    )


def passes_fresh_catboost_bag_gate(
    *,
    mean_accuracy_delta: float,
    fold_wins: int,
    worst_fold_delta: float,
) -> bool:
    """Apply the existing spatial-grid fresh gate without adding criteria."""

    return (
        _at_least(mean_accuracy_delta, SPATIAL_GRID_GATE_ACCURACY)
        and fold_wins >= SPATIAL_GRID_GATE_FOLD_WINS
        and _at_least(worst_fold_delta, SPATIAL_GRID_GATE_WORST_FOLD)
    )


def compare_locked_catboost_bag_ensemble(
    partitioned_data: PartitionedData,
    incumbent_components: dict[str, _np.ndarray],
    spatial_grid: _np.ndarray,
) -> LockedCatBoostBagEnsembleComparison:
    """Place the fixed CatBoost bag into the unchanged 20% archive slot."""

    identity_grid_bag = make_equal_catboost_probability_bag(
        incumbent_components["identity_catboost"],
        spatial_grid,
    )
    incumbent_probabilities = blend_deep_archive(incumbent_components)
    candidate_probabilities = build_locked_substitution_probabilities(
        incumbent_components,
        LockedReplacementDecision(True, None),
        spatial_grid_catboost=identity_grid_bag,
    )[SPATIAL_SUBSTITUTION]
    cross_validation = make_cross_validation(partitioned_data)
    incumbent = _build_evaluation(
        "seed-20260824 deep archive",
        partitioned_data,
        cross_validation,
        incumbent_probabilities,
    )
    candidate = _build_evaluation(
        CATBOOST_BAG_CANDIDATE,
        partitioned_data,
        cross_validation,
        candidate_probabilities,
    )
    folds = _paired_folds(partitioned_data, incumbent, candidate).rename(
        columns={
            "identity_accuracy": "incumbent_accuracy",
            "bag_accuracy": "candidate_accuracy",
            "identity_repair_recall": "incumbent_repair_recall",
            "bag_repair_recall": "candidate_repair_recall",
        }
    )
    repair_metric = "recall: functional needs repair"
    accuracy_delta = float(
        candidate.metric_summary.loc["accuracy", "mean"]
        - incumbent.metric_summary.loc["accuracy", "mean"]
    )
    repair_delta = float(
        candidate.metric_summary.loc[repair_metric, "mean"]
        - incumbent.metric_summary.loc[repair_metric, "mean"]
    )
    fold_wins = int(folds["accuracy_delta"].gt(0).sum())
    worst_fold = float(folds["accuracy_delta"].min())
    summary = _pd.DataFrame(
        [
            {
                "candidate": CATBOOST_BAG_CANDIDATE,
                "incumbent_mean_accuracy": float(
                    incumbent.metric_summary.loc["accuracy", "mean"]
                ),
                "candidate_mean_accuracy": float(
                    candidate.metric_summary.loc["accuracy", "mean"]
                ),
                "mean_accuracy_delta": accuracy_delta,
                "fold_wins": fold_wins,
                "worst_fold_delta": worst_fold,
                "incumbent_repair_recall": float(
                    incumbent.metric_summary.loc[repair_metric, "mean"]
                ),
                "candidate_repair_recall": float(
                    candidate.metric_summary.loc[repair_metric, "mean"]
                ),
                "repair_recall_delta": repair_delta,
                "net_additional_correct": int(
                    folds["net_additional_correct"].sum()
                ),
                "passes_gate": passes_locked_rf_ensemble_gate(
                    mean_accuracy_delta=accuracy_delta,
                    fold_wins=fold_wins,
                    worst_fold_delta=worst_fold,
                    repair_recall_delta=repair_delta,
                ),
            }
        ]
    ).set_index("candidate")
    return LockedCatBoostBagEnsembleComparison(
        incumbent_evaluation=incumbent,
        candidate_evaluation=candidate,
        summary=summary,
        fold_deltas=folds,
    )


def validate_old_fold_spatial_grid_cache(
    payload: object,
    expected_metadata: dict[str, object],
    partitioned_data: PartitionedData,
) -> CandidateEvaluation:
    """Reject a stale spatial-grid recipe/cache and replay its OOF metrics."""

    if not isinstance(payload, dict) or set(payload) != {
        "cache_version",
        "metadata",
        "trial",
    }:
        raise ValueError("Old-fold spatial-grid cache schema changed.")
    if payload["cache_version"] != CATBOOST_OLD_FOLD_CACHE_VERSION:
        raise ValueError("Old-fold spatial-grid cache version changed.")
    if payload["metadata"] != expected_metadata:
        raise ValueError("Old-fold spatial-grid cache metadata is stale.")
    trial = payload["trial"]
    if not isinstance(trial, CatBoostIdentityTrial):
        raise ValueError("Old-fold spatial-grid cache has no trial.")
    return recompute_and_validate_oof_evaluation(
        trial.evaluation,
        partitioned_data,
        make_cross_validation(partitioned_data),
        candidate="old-fold spatial_grid_catboost",
    )


def _build_evaluation(
    name: str,
    partitioned_data: PartitionedData,
    cross_validation,
    probabilities: _np.ndarray,
) -> CandidateEvaluation:
    validate_probabilities(probabilities, len(partitioned_data.y_development))
    return build_candidate_evaluation(
        model_name=name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=[
            {"validation_fold": fold_number}
            for fold_number in range(1, 6)
        ],
    )


def _paired_folds(
    partitioned_data: PartitionedData,
    identity: CandidateEvaluation,
    bag: CandidateEvaluation,
) -> _pd.DataFrame:
    identity_predictions = identity.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    bag_predictions = bag.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    rows = []
    for fold_number, (_, positions) in enumerate(
        make_cross_validation(partitioned_data).split(),
        start=1,
    ):
        actual = partitioned_data.y_development.iloc[positions].to_numpy()
        identity_correct = identity_predictions[positions] == actual
        bag_correct = bag_predictions[positions] == actual
        rows.append(
            {
                "validation_fold": fold_number,
                "rows": len(positions),
                "identity_accuracy": identity.fold_metrics.loc[
                    fold_number, "accuracy"
                ],
                "bag_accuracy": bag.fold_metrics.loc[fold_number, "accuracy"],
                "accuracy_delta": (
                    bag.fold_metrics.loc[fold_number, "accuracy"]
                    - identity.fold_metrics.loc[fold_number, "accuracy"]
                ),
                "identity_repair_recall": identity.fold_metrics.loc[
                    fold_number, "recall: functional needs repair"
                ],
                "bag_repair_recall": bag.fold_metrics.loc[
                    fold_number, "recall: functional needs repair"
                ],
                "repair_recall_delta": (
                    bag.fold_metrics.loc[
                        fold_number, "recall: functional needs repair"
                    ]
                    - identity.fold_metrics.loc[
                        fold_number, "recall: functional needs repair"
                    ]
                ),
                "gained_correct": int(
                    (bag_correct & ~identity_correct).sum()
                ),
                "lost_correct": int(
                    (~bag_correct & identity_correct).sum()
                ),
                "net_additional_correct": int(
                    bag_correct.sum() - identity_correct.sum()
                ),
            }
        )
    return _pd.DataFrame(rows).set_index("validation_fold")


def _at_least(actual: float, threshold: float) -> bool:
    return float(actual) >= threshold - 1e-12

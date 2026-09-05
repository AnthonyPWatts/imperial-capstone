"""Evaluate the single predeclared equal ten-seed deep-XGBoost bag."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd

from data_partitioning import PartitionedData
from deep_xgboost_seed_average import FOLD_WIN_GATE
from deep_xgboost_seed_average import MEAN_ACCURACY_GAIN_GATE
from deep_xgboost_seed_average import REPAIR_RECALL_DELTA_GATE
from deep_xgboost_seed_average import WORST_FOLD_DELTA_GATE
from deep_xgboost_seed_average import rebuild_and_validate_evaluation
from final_model import validate_probabilities
from model_evaluation import build_candidate_evaluation
from model_evaluation import CandidateEvaluation


LOCKED_SEEDS = (
    20260822,
    20260823,
    20260824,
    20260905,
    20260906,
    20260907,
    20260908,
    20260909,
    20260910,
    20260911,
)
FIT_SEEDS = tuple(seed for seed in LOCKED_SEEDS if seed not in {20260824, 20260905})
SEED_20260824_KEY = "seed_20260824"
SEED_20260905_KEY = "seed_20260905"
TEN_BAG_NAME = "Equal 10-seed raw top-50 deep-XGBoost bag"
TWO_BAG_NAME = "Fixed 50:50 seed-20260824/seed-20260905 bag"
REPAIR_METRIC = "recall: functional needs repair"


@_dataclass(frozen=True)
class TenSeedBagConfirmation:
    """The fixed baselines, ten-seed bag and two paired comparisons."""

    fixed_two_seed_bag: CandidateEvaluation
    ten_seed_bag: CandidateEvaluation
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame
    passes_gate_against_both: bool


def confirm_equal_ten_seed_bag(
    partitioned_data: PartitionedData,
    cross_validation: object,
    seed_evaluations: _Mapping[int, CandidateEvaluation],
    cached_two_seed_bag: CandidateEvaluation,
) -> TenSeedBagConfirmation:
    """Evaluate only the equal ten-bag against seed24 and the fixed two-bag."""

    if set(seed_evaluations) != set(LOCKED_SEEDS):
        raise ValueError("Ten-seed evidence set changed from the locked list.")
    rebuilt = {
        seed: rebuild_and_validate_evaluation(
            evaluation,
            partitioned_data,
            cross_validation,
            evidence_name=f"seed_{seed}",
        )
        for seed, evaluation in seed_evaluations.items()
    }
    two_seed_probabilities = 0.5 * (
        rebuilt[20260824].out_of_fold_probabilities.to_numpy(dtype="float64")
        + rebuilt[20260905].out_of_fold_probabilities.to_numpy(dtype="float64")
    )
    validate_probabilities(two_seed_probabilities, len(partitioned_data.y_development))
    cached_two_seed = rebuild_and_validate_evaluation(
        cached_two_seed_bag,
        partitioned_data,
        cross_validation,
        evidence_name="cached_fixed_two_seed_bag",
    )
    if not _np.array_equal(
        cached_two_seed.out_of_fold_probabilities.to_numpy(),
        two_seed_probabilities,
    ):
        raise ValueError("Cached fixed two-seed bag differs from its seed inputs.")
    fixed_two_seed = _build_evaluation(
        TWO_BAG_NAME,
        partitioned_data,
        cross_validation,
        two_seed_probabilities,
    )

    probability_sum = _np.zeros_like(two_seed_probabilities, dtype="float64")
    for seed in LOCKED_SEEDS:
        probability_sum += rebuilt[seed].out_of_fold_probabilities.to_numpy(
            dtype="float64"
        )
    ten_seed_probabilities = probability_sum / len(LOCKED_SEEDS)
    validate_probabilities(ten_seed_probabilities, len(partitioned_data.y_development))
    ten_seed = _build_evaluation(
        TEN_BAG_NAME,
        partitioned_data,
        cross_validation,
        ten_seed_probabilities,
    )
    baselines = {
        SEED_20260824_KEY: rebuilt[20260824],
        "fixed_two_seed_bag": fixed_two_seed,
    }
    fold_deltas = _paired_fold_deltas(
        partitioned_data,
        cross_validation,
        baselines,
        ten_seed,
    )
    summary = _summarise(baselines, ten_seed, fold_deltas)
    return TenSeedBagConfirmation(
        fixed_two_seed_bag=fixed_two_seed,
        ten_seed_bag=ten_seed,
        summary=summary,
        fold_deltas=fold_deltas,
        passes_gate_against_both=bool(summary["passes_gate"].all()),
    )


def passes_ten_seed_gate(
    *,
    mean_accuracy_delta: float,
    fold_wins: int,
    worst_fold_delta: float,
    repair_recall_delta: float,
) -> bool:
    """Apply the standard four-part fresh-screen promotion gate."""

    return (
        _at_least(mean_accuracy_delta, MEAN_ACCURACY_GAIN_GATE)
        and fold_wins >= FOLD_WIN_GATE
        and _at_least(worst_fold_delta, WORST_FOLD_DELTA_GATE)
        and _at_least(repair_recall_delta, REPAIR_RECALL_DELTA_GATE)
    )


def _build_evaluation(
    name: str,
    partitioned_data: PartitionedData,
    cross_validation: object,
    probabilities: _np.ndarray,
) -> CandidateEvaluation:
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
    cross_validation: object,
    baselines: _Mapping[str, CandidateEvaluation],
    ten_seed: CandidateEvaluation,
) -> _pd.DataFrame:
    ten_predictions = ten_seed.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    rows = []
    for baseline_name, baseline in baselines.items():
        baseline_predictions = baseline.out_of_fold_probabilities.idxmax(
            axis="columns"
        ).to_numpy()
        for fold, (_, positions) in enumerate(cross_validation.split(), start=1):
            actual = partitioned_data.y_development.iloc[positions].to_numpy()
            baseline_correct = baseline_predictions[positions] == actual
            ten_correct = ten_predictions[positions] == actual
            gained = int((ten_correct & ~baseline_correct).sum())
            lost = int((~ten_correct & baseline_correct).sum())
            rows.append(
                {
                    "comparison": f"ten_seed_bag_vs_{baseline_name}",
                    "baseline": baseline_name,
                    "validation_fold": fold,
                    "rows": len(positions),
                    "baseline_accuracy": float(baseline_correct.mean()),
                    "ten_seed_bag_accuracy": float(ten_correct.mean()),
                    "accuracy_delta": float(
                        ten_correct.mean() - baseline_correct.mean()
                    ),
                    "baseline_repair_recall": float(
                        baseline.fold_metrics.loc[fold, REPAIR_METRIC]
                    ),
                    "ten_seed_bag_repair_recall": float(
                        ten_seed.fold_metrics.loc[fold, REPAIR_METRIC]
                    ),
                    "repair_recall_delta": float(
                        ten_seed.fold_metrics.loc[fold, REPAIR_METRIC]
                        - baseline.fold_metrics.loc[fold, REPAIR_METRIC]
                    ),
                    "prediction_disagreements": int(
                        (ten_predictions[positions]
                         != baseline_predictions[positions]).sum()
                    ),
                    "gained_correct": gained,
                    "lost_correct": lost,
                    "net_additional_correct": gained - lost,
                }
            )
    return _pd.DataFrame(rows)


def _summarise(
    baselines: _Mapping[str, CandidateEvaluation],
    ten_seed: CandidateEvaluation,
    fold_deltas: _pd.DataFrame,
) -> _pd.DataFrame:
    rows = []
    for baseline_name, baseline in baselines.items():
        comparison = f"ten_seed_bag_vs_{baseline_name}"
        folds = fold_deltas.loc[fold_deltas["comparison"].eq(comparison)]
        accuracy_delta = float(
            ten_seed.metric_summary.loc["accuracy", "mean"]
            - baseline.metric_summary.loc["accuracy", "mean"]
        )
        repair_delta = float(
            ten_seed.metric_summary.loc[REPAIR_METRIC, "mean"]
            - baseline.metric_summary.loc[REPAIR_METRIC, "mean"]
        )
        fold_wins = int(folds["accuracy_delta"].gt(0).sum())
        worst_fold_delta = float(folds["accuracy_delta"].min())
        rows.append(
            {
                "comparison": comparison,
                "baseline": baseline_name,
                "baseline_mean_accuracy": float(
                    baseline.metric_summary.loc["accuracy", "mean"]
                ),
                "ten_seed_bag_mean_accuracy": float(
                    ten_seed.metric_summary.loc["accuracy", "mean"]
                ),
                "mean_accuracy_delta": accuracy_delta,
                "fold_wins": fold_wins,
                "fold_losses": int(folds["accuracy_delta"].lt(0).sum()),
                "worst_fold_delta": worst_fold_delta,
                "best_fold_delta": float(folds["accuracy_delta"].max()),
                "baseline_repair_recall": float(
                    baseline.metric_summary.loc[REPAIR_METRIC, "mean"]
                ),
                "ten_seed_bag_repair_recall": float(
                    ten_seed.metric_summary.loc[REPAIR_METRIC, "mean"]
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
                "passes_gate": passes_ten_seed_gate(
                    mean_accuracy_delta=accuracy_delta,
                    fold_wins=fold_wins,
                    worst_fold_delta=worst_fold_delta,
                    repair_recall_delta=repair_delta,
                ),
            }
        )
    return _pd.DataFrame(rows).set_index("comparison")


def _at_least(actual: float, threshold: float) -> bool:
    return float(actual) >= threshold - 1e-12

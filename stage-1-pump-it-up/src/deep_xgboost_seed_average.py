"""Evaluate the predeclared two-seed deep-XGBoost probability average."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd

from data_partitioning import PartitionedData
from final_model import validate_probabilities
from gpu_model_evaluation import CLASS_LABELS
from model_evaluation import build_candidate_evaluation
from model_evaluation import CandidateEvaluation


SEED_EVIDENCE_KEYS = ("seed_20260824", "seed_20260905")
AVERAGE_MODEL_NAME = "50:50 raw top-50 deep-XGBoost seed probability average"
MEAN_ACCURACY_GAIN_GATE = 0.001
FOLD_WIN_GATE = 3
WORST_FOLD_DELTA_GATE = -0.0025
REPAIR_RECALL_DELTA_GATE = -0.02
REPAIR_METRIC = "recall: functional needs repair"


@_dataclass(frozen=True)
class SeedAverageConfirmation:
    """Recomputed average, paired evidence and the locked decision."""

    average: CandidateEvaluation
    seed_evaluations: dict[str, CandidateEvaluation]
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame
    gate_failures: dict[str, tuple[str, ...]]
    passes_gate_against_both: bool
    stable_positive_near_miss: bool
    verdict: str


def confirm_seed_average(
    partitioned_data: PartitionedData,
    cross_validation: object,
    seed_evaluations: _Mapping[str, CandidateEvaluation],
) -> SeedAverageConfirmation:
    """Evaluate only the exact 50:50 average against its two seed inputs."""

    if set(seed_evaluations) != set(SEED_EVIDENCE_KEYS):
        raise ValueError(
            "Two-seed confirmation requires exactly "
            f"{SEED_EVIDENCE_KEYS!r}."
        )
    rebuilt = {
        key: rebuild_and_validate_evaluation(
            evaluation,
            partitioned_data,
            cross_validation,
            evidence_name=key,
        )
        for key, evaluation in seed_evaluations.items()
    }
    probability_values = 0.5 * (
        rebuilt[SEED_EVIDENCE_KEYS[0]].out_of_fold_probabilities.to_numpy()
        + rebuilt[SEED_EVIDENCE_KEYS[1]].out_of_fold_probabilities.to_numpy()
    )
    validate_probabilities(probability_values, len(partitioned_data.y_development))
    average = build_candidate_evaluation(
        model_name=AVERAGE_MODEL_NAME,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=[
            {
                "validation_fold": fold,
                "seed_20260824_weight": 0.5,
                "seed_20260905_weight": 0.5,
            }
            for fold in range(1, 6)
        ],
    )
    fold_deltas = _paired_fold_deltas(
        partitioned_data,
        cross_validation,
        rebuilt,
        average,
    )
    summary = _summarise(rebuilt, average, fold_deltas)
    (
        gate_failures,
        passes_both,
        stable_near_miss,
        verdict,
    ) = classify_seed_average_summary(summary)
    return SeedAverageConfirmation(
        average=average,
        seed_evaluations=rebuilt,
        summary=summary,
        fold_deltas=fold_deltas,
        gate_failures=gate_failures,
        passes_gate_against_both=passes_both,
        stable_positive_near_miss=stable_near_miss,
        verdict=verdict,
    )


def classify_seed_average_summary(
    summary: _pd.DataFrame,
) -> tuple[dict[str, tuple[str, ...]], bool, bool, str]:
    """Classify the locked two-baseline gate and positive near-miss case."""

    expected = {f"average_vs_{key}" for key in SEED_EVIDENCE_KEYS}
    if set(summary.index) != expected:
        raise ValueError("Seed-average summary comparisons changed.")
    required = {
        "mean_accuracy_delta",
        "fold_wins",
        "worst_fold_delta",
        "repair_recall_delta",
        "passes_gate",
    }
    if not required.issubset(summary.columns):
        raise ValueError("Seed-average summary metrics are incomplete.")
    gate_failures = {
        comparison: _gate_failures(row)
        for comparison, row in summary.iterrows()
    }
    recorded_passes = {
        comparison: bool(row["passes_gate"])
        for comparison, row in summary.iterrows()
    }
    recomputed_passes = {
        comparison: not failures
        for comparison, failures in gate_failures.items()
    }
    if recorded_passes != recomputed_passes:
        raise ValueError("Seed-average recorded gate decisions are inconsistent.")
    passes_both = all(recomputed_passes.values())
    stable_near_miss = (
        not passes_both
        and bool(summary["mean_accuracy_delta"].gt(0).all())
        and all(
            set(failures).issubset({"mean_accuracy_gain"})
            for failures in gate_failures.values()
        )
    )
    if passes_both:
        verdict = "passes_gate_against_both_seeds"
    elif stable_near_miss:
        verdict = "stable_positive_near_miss"
    else:
        verdict = "fails_confirmation"
    return gate_failures, passes_both, stable_near_miss, verdict


def rebuild_and_validate_evaluation(
    evaluation: CandidateEvaluation,
    partitioned_data: PartitionedData,
    cross_validation: object,
    *,
    evidence_name: str,
) -> CandidateEvaluation:
    """Rebuild all scored metrics from aligned OOF probabilities."""

    probabilities = evaluation.out_of_fold_probabilities
    if evaluation.cross_validation_fingerprint != (
        partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError(f"{evidence_name} fold fingerprint is misaligned.")
    if not probabilities.index.equals(partitioned_data.y_development.index):
        raise ValueError(f"{evidence_name} OOF row index is misaligned.")
    if tuple(probabilities.columns) != tuple(CLASS_LABELS):
        raise ValueError(f"{evidence_name} probability classes are misaligned.")
    validate_probabilities(
        probabilities.to_numpy(),
        len(partitioned_data.y_development),
    )
    recomputed = build_candidate_evaluation(
        model_name=evaluation.model_name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probabilities.to_numpy(),
        diagnostic_rows=[
            {"validation_fold": fold}
            for fold in range(1, 6)
        ],
    )
    for field in (
        "fold_metrics",
        "metric_summary",
        "confusion_counts",
        "confusion_recall",
    ):
        if not getattr(evaluation, field).equals(getattr(recomputed, field)):
            raise ValueError(
                f"{evidence_name} cached {field} disagrees with its OOF rows."
            )
    return recomputed


def _paired_fold_deltas(
    partitioned_data: PartitionedData,
    cross_validation: object,
    seeds: _Mapping[str, CandidateEvaluation],
    average: CandidateEvaluation,
) -> _pd.DataFrame:
    average_predictions = average.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    rows = []
    for seed_key in SEED_EVIDENCE_KEYS:
        baseline = seeds[seed_key]
        baseline_predictions = baseline.out_of_fold_probabilities.idxmax(
            axis="columns"
        ).to_numpy()
        for fold, (_, positions) in enumerate(
            cross_validation.split(),
            start=1,
        ):
            actual = partitioned_data.y_development.iloc[positions].to_numpy()
            baseline_fold_predictions = baseline_predictions[positions]
            average_fold_predictions = average_predictions[positions]
            baseline_correct = baseline_fold_predictions == actual
            average_correct = average_fold_predictions == actual
            gained = int((average_correct & ~baseline_correct).sum())
            lost = int((~average_correct & baseline_correct).sum())
            rows.append(
                {
                    "comparison": f"average_vs_{seed_key}",
                    "baseline": seed_key,
                    "validation_fold": fold,
                    "rows": len(positions),
                    "baseline_accuracy": float(baseline_correct.mean()),
                    "average_accuracy": float(average_correct.mean()),
                    "accuracy_delta": float(
                        average_correct.mean() - baseline_correct.mean()
                    ),
                    "baseline_repair_recall": float(
                        baseline.fold_metrics.loc[fold, REPAIR_METRIC]
                    ),
                    "average_repair_recall": float(
                        average.fold_metrics.loc[fold, REPAIR_METRIC]
                    ),
                    "repair_recall_delta": float(
                        average.fold_metrics.loc[fold, REPAIR_METRIC]
                        - baseline.fold_metrics.loc[fold, REPAIR_METRIC]
                    ),
                    "prediction_disagreements": int(
                        (average_fold_predictions != baseline_fold_predictions).sum()
                    ),
                    "gained_correct": gained,
                    "lost_correct": lost,
                    "net_additional_correct": gained - lost,
                }
            )
    return _pd.DataFrame(rows)


def _summarise(
    seeds: _Mapping[str, CandidateEvaluation],
    average: CandidateEvaluation,
    fold_deltas: _pd.DataFrame,
) -> _pd.DataFrame:
    rows = []
    for seed_key in SEED_EVIDENCE_KEYS:
        baseline = seeds[seed_key]
        comparison = f"average_vs_{seed_key}"
        folds = fold_deltas.loc[fold_deltas["comparison"].eq(comparison)]
        mean_delta = float(
            average.metric_summary.loc["accuracy", "mean"]
            - baseline.metric_summary.loc["accuracy", "mean"]
        )
        repair_delta = float(
            average.metric_summary.loc[REPAIR_METRIC, "mean"]
            - baseline.metric_summary.loc[REPAIR_METRIC, "mean"]
        )
        row = {
            "comparison": comparison,
            "baseline": seed_key,
            "baseline_mean_accuracy": float(
                baseline.metric_summary.loc["accuracy", "mean"]
            ),
            "average_mean_accuracy": float(
                average.metric_summary.loc["accuracy", "mean"]
            ),
            "mean_accuracy_delta": mean_delta,
            "fold_wins": int(folds["accuracy_delta"].gt(0).sum()),
            "fold_losses": int(folds["accuracy_delta"].lt(0).sum()),
            "worst_fold_delta": float(folds["accuracy_delta"].min()),
            "best_fold_delta": float(folds["accuracy_delta"].max()),
            "baseline_repair_recall": float(
                baseline.metric_summary.loc[REPAIR_METRIC, "mean"]
            ),
            "average_repair_recall": float(
                average.metric_summary.loc[REPAIR_METRIC, "mean"]
            ),
            "repair_recall_delta": repair_delta,
            "gained_correct": int(folds["gained_correct"].sum()),
            "lost_correct": int(folds["lost_correct"].sum()),
            "net_additional_correct": int(
                folds["net_additional_correct"].sum()
            ),
        }
        row["passes_gate"] = not _gate_failures(_pd.Series(row))
        rows.append(row)
    return _pd.DataFrame(rows).set_index("comparison")


def _gate_failures(row: _pd.Series) -> tuple[str, ...]:
    failures = []
    if not _at_least(row["mean_accuracy_delta"], MEAN_ACCURACY_GAIN_GATE):
        failures.append("mean_accuracy_gain")
    if int(row["fold_wins"]) < FOLD_WIN_GATE:
        failures.append("fold_wins")
    if not _at_least(row["worst_fold_delta"], WORST_FOLD_DELTA_GATE):
        failures.append("worst_fold_delta")
    if not _at_least(row["repair_recall_delta"], REPAIR_RECALL_DELTA_GATE):
        failures.append("repair_recall_delta")
    return tuple(failures)


def _at_least(value: float, threshold: float) -> bool:
    return float(value) >= threshold - 1e-12

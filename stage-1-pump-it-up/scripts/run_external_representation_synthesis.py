"""Combine the two fixed archive-derived representation candidates once."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "external-representation-synthesis"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import compare_candidate_diversity
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data


# Equal averaging of the already fixed frequency-forest and spatial-height
# representation votes. The resulting component weights are algebraic, not
# fitted or searched against the OOF labels.
ARCHIVE_SYNTHESIS_WEIGHTS = (0.33, 0.11, 0.18, 0.09, 0.09, 0.20)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned = partition_modelling_data(
        prepare_modelling_data(
            pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
            pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
            pd.read_csv(DATA_DIR / "TestSetValues.csv"),
        )
    )
    baseline = joblib.load(
        PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
    )
    identity = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "catboost-identity-screen"
        / "complete-deferred-identities.joblib"
    ).evaluation
    frequency_forest = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "categorical-frequency-forest-screen"
        / "all-categorical-occurrence-counts.joblib"
    ).random_forest
    spatial = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "spatial-height-imputation-screen"
        / "ten-neighbour-height.joblib"
    )
    age = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "age-cohort-screen"
        / "fixed-age-cohort.joblib"
    )
    age_identity = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "age-cohort-screen"
        / "identity-pump-age-cohort.joblib"
    ).evaluation
    _validate_evidence(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity,
        frequency_forest,
        spatial.xgboost,
        spatial.random_forest,
        age.xgboost,
        age.random_forest,
        age_identity,
    )

    leader = _vote(
        partitioned,
        (baseline.xgboost, baseline.random_forest, identity),
        (0.44, 0.36, 0.20),
        "Promoted complete-identity vote",
    )
    archive_synthesis = _vote(
        partitioned,
        (
            baseline.xgboost,
            spatial.xgboost,
            baseline.random_forest,
            frequency_forest,
            spatial.random_forest,
            identity,
        ),
        ARCHIVE_SYNTHESIS_WEIGHTS,
        "Equal archive-derived representation synthesis",
    )
    archive_age_synthesis = _vote(
        partitioned,
        (
            baseline.xgboost,
            spatial.xgboost,
            baseline.random_forest,
            frequency_forest,
            spatial.random_forest,
            identity,
            age.xgboost,
            age.random_forest,
            age_identity,
        ),
        (0.165, 0.055, 0.09, 0.045, 0.045, 0.10, 0.22, 0.18, 0.10),
        "Equal archive-synthesis and age-cohort vote",
    )
    candidates = {
        "promoted_identity_vote": leader,
        "archive_representation_synthesis": archive_synthesis,
        "archive_plus_age_synthesis": archive_age_synthesis,
    }
    summary = _summarise(candidates, leader)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    diversity = compare_candidate_diversity(
        partitioned,
        leader,
        archive_synthesis,
        archive_age_synthesis,
    )
    diversity.to_csv(OUTPUT_DIR / "diversity.csv")
    result = {
        "archive_synthesis_weights": list(ARCHIVE_SYNTHESIS_WEIGHTS),
        "archive_synthesis_accuracy": float(
            summary.loc["archive_representation_synthesis", "mean_accuracy"]
        ),
        "change_vs_promoted_identity_vote": float(
            summary.loc["archive_representation_synthesis", "leader_change"]
        ),
        "passes_gate": bool(
            summary.loc["archive_representation_synthesis", "passes_gate"]
        ),
        "age_follow_up_passes_gate": bool(
            summary.loc["archive_plus_age_synthesis", "passes_gate"]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)
    print("\nDiversity:\n", diversity.to_string(), flush=True)


def _vote(partitioned, evaluations, weights, name):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        *evaluations,
        weights=weights,
        model_name=name,
    )


def _validate_evidence(partitioned, *evaluations) -> None:
    for evaluation in evaluations:
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Synthesis evidence uses different folds.")


def _summarise(candidates, leader):
    leader_accuracy = float(leader.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        leader.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = []
    for key, candidate in candidates.items():
        fold_change = candidate.fold_metrics.accuracy - leader.fold_metrics.accuracy
        accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        change = accuracy - leader_accuracy
        repair_change = repair - leader_repair
        rows.append(
            {
                "candidate": key,
                "mean_accuracy": accuracy,
                "leader_change": change,
                "fold_wins_vs_leader": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall_change": repair_change,
                "passes_gate": key == "archive_representation_synthesis" and (
                    change >= 0.001
                    and int(fold_change.gt(0).sum()) >= 3
                    and float(fold_change.min()) >= -0.0025
                    and repair_change >= -0.02
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("candidate")
        .sort_values(["mean_accuracy", "candidate"], ascending=[False, True])
    )


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in (
        "mean_accuracy",
        "leader_change",
        "worst_fold_change",
        "repair_recall_change",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


if __name__ == "__main__":
    main()

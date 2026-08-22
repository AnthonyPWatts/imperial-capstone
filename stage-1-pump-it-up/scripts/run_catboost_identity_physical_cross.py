"""Cross the promoted identity CatBoost with source-plus-class components."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "catboost-identity-physical-cross"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
IDENTITY_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "catboost-identity-screen"
    / "complete-deferred-identities.joblib"
)
SOURCE_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "physical-hierarchy-screen"
    / "source-source_plus_class.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned = partition_modelling_data(
        prepare_modelling_data(
            pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
            pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
            pd.read_csv(DATA_DIR / "TestSetValues.csv"),
        )
    )
    baseline = joblib.load(BASELINE_PATH)
    identity = joblib.load(IDENTITY_PATH).evaluation
    source = joblib.load(SOURCE_PATH)
    _validate_evidence(partitioned, baseline, identity, source)
    incumbent = _vote(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity,
        "Promoted complete-identity vote",
    )
    source_cross = _vote(
        partitioned,
        source.xgboost,
        source.random_forest,
        identity,
        "Source-plus-class vote with complete-identity CatBoost",
    )
    summary = _summarise(
        incumbent,
        source.blend,
        identity,
        source_cross,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result = {
        "candidate_accuracy": float(
            summary.loc["source_plus_class_identity_cross", "mean_accuracy"]
        ),
        "change_vs_promoted_identity_vote": float(
            summary.loc["source_plus_class_identity_cross", "leader_change"]
        ),
        "passes_gate": bool(
            summary.loc["source_plus_class_identity_cross", "passes_gate"]
        ),
        "models_refitted": False,
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


def _vote(partitioned, xgboost, forest, catboost, name):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        xgboost,
        forest,
        catboost,
        weights=[0.44, 0.36, 0.20],
        model_name=name,
    )


def _validate_evidence(partitioned, baseline, identity, source) -> None:
    for evaluation in (
        baseline.xgboost,
        baseline.random_forest,
        baseline.blend,
        identity,
        source.xgboost,
        source.random_forest,
        source.blend,
    ):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Identity/physical cross uses different folds.")


def _summarise(leader, source_blend, identity, source_cross):
    leader_accuracy = float(leader.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        leader.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = []
    for key, candidate in {
        "promoted_identity_vote": leader,
        "source_plus_class_vote": source_blend,
        "complete_identity_catboost": identity,
        "source_plus_class_identity_cross": source_cross,
    }.items():
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
                "passes_gate": key == "source_plus_class_identity_cross" and (
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

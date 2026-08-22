"""Run the fixed explicit recording-time feature screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "temporal-feature-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
IDENTITY_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "catboost-identity-screen"
    / "complete-deferred-identities.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from temporal_feature_evaluation import evaluate_temporal_feature_trial


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
    _validate_evidence(partitioned, baseline, identity)
    cache_path = OUTPUT_DIR / "explicit-recording-time.joblib"
    if cache_path.exists():
        temporal = joblib.load(cache_path)
        for evaluation in (temporal.xgboost, temporal.random_forest):
            if (
                evaluation.cross_validation_fingerprint
                != partitioned.cross_validation_fingerprint
            ):
                raise ValueError("Cached temporal trial uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        temporal = evaluate_temporal_feature_trial(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(temporal, cache_path)

    leader = _vote(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity,
        "Promoted complete-identity vote",
    )
    temporal_vote = _vote(
        partitioned,
        temporal.xgboost,
        temporal.random_forest,
        identity,
        "Explicit recording-time identity vote",
    )
    temporal_two_component = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        temporal.xgboost,
        temporal.random_forest,
        weights=[0.55, 0.45],
        model_name="Explicit recording-time 55:45 vote",
    )
    summary = _summarise(
        leader,
        baseline.xgboost,
        baseline.random_forest,
        temporal,
        temporal_two_component,
        temporal_vote,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result = {
        "engineered_features": temporal.engineered_features,
        "transformed_features_fold_1": temporal.transformed_features_fold_1,
        "temporal_identity_vote_accuracy": float(
            summary.loc["temporal_identity_vote", "mean_accuracy"]
        ),
        "change_vs_promoted_identity_vote": float(
            summary.loc["temporal_identity_vote", "leader_change"]
        ),
        "passes_gate": bool(summary.loc["temporal_identity_vote", "passes_gate"]),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


def _vote(partitioned, xgboost, forest, identity, name):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        xgboost,
        forest,
        identity,
        weights=[0.44, 0.36, 0.20],
        model_name=name,
    )


def _validate_evidence(partitioned, baseline, identity) -> None:
    for evaluation in (
        baseline.xgboost,
        baseline.random_forest,
        identity,
    ):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Temporal comparison uses different frozen folds.")


def _summarise(
    leader,
    accepted_xgboost,
    accepted_forest,
    temporal,
    temporal_two_component,
    temporal_vote,
):
    leader_accuracy = float(leader.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        leader.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = []
    for key, candidate in {
        "promoted_identity_vote": leader,
        "accepted_xgboost": accepted_xgboost,
        "temporal_xgboost": temporal.xgboost,
        "accepted_random_forest": accepted_forest,
        "temporal_random_forest": temporal.random_forest,
        "temporal_two_component_vote": temporal_two_component,
        "temporal_identity_vote": temporal_vote,
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
                "passes_gate": key == "temporal_identity_vote" and (
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

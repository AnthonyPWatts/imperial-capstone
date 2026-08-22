"""Run the fixed target-free deferred-identity hash screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "identity-hash-screen"
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
from identity_hash_evaluation import evaluate_identity_hash_trial
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
    _validate_evidence(partitioned, baseline, identity)
    cache_path = OUTPUT_DIR / "shared-4096-hash-xgboost.joblib"
    if cache_path.exists():
        hashed = joblib.load(cache_path)
        if (
            hashed.xgboost.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Cached identity hash uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        hashed = evaluate_identity_hash_trial(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(hashed, cache_path)

    leader = _vote(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity,
        "Promoted complete-identity vote",
    )
    hash_vote = _vote(
        partitioned,
        hashed.xgboost,
        baseline.random_forest,
        identity,
        "Hashed-XGBoost identity vote",
    )
    summary = _summarise(leader, baseline.xgboost, hashed.xgboost, hash_vote)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result = {
        "hash_features": hashed.hash_features,
        "transformed_features_fold_1": hashed.transformed_features_fold_1,
        "maximum_hash_nonzeros_per_row": hashed.maximum_hash_nonzeros_per_row,
        "hash_vote_accuracy": float(
            summary.loc["hashed_xgboost_identity_vote", "mean_accuracy"]
        ),
        "change_vs_promoted_identity_vote": float(
            summary.loc["hashed_xgboost_identity_vote", "leader_change"]
        ),
        "passes_gate": bool(
            summary.loc["hashed_xgboost_identity_vote", "passes_gate"]
        ),
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
    for evaluation in (baseline.xgboost, baseline.random_forest, identity):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Identity hash comparison uses different folds.")


def _summarise(leader, accepted_xgboost, hashed_xgboost, hash_vote):
    leader_accuracy = float(leader.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        leader.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = []
    for key, candidate in {
        "promoted_identity_vote": leader,
        "accepted_xgboost": accepted_xgboost,
        "hashed_xgboost": hashed_xgboost,
        "hashed_xgboost_identity_vote": hash_vote,
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
                "passes_gate": key == "hashed_xgboost_identity_vote" and (
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

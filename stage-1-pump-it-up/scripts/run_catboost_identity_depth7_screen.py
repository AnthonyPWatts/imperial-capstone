"""Run one diversity-motivated depth-7 complete-identity CatBoost screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "catboost-identity-depth7-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
DEPTH8_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "catboost-identity-screen"
    / "complete-deferred-identities.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_evaluation import (
    evaluate_depth7_complete_identity_catboost,
)
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
    depth8 = joblib.load(DEPTH8_PATH)
    _validate_evidence(partitioned, baseline, depth8.evaluation)
    cache_path = OUTPUT_DIR / "depth7-complete-identities.joblib"
    if cache_path.exists():
        depth7 = joblib.load(cache_path)
        if (
            depth7.evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Cached depth-7 identity CatBoost uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        depth7 = evaluate_depth7_complete_identity_catboost(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(depth7, cache_path)

    depth8_vote = _vote(partitioned, baseline, depth8.evaluation, "depth 8")
    depth7_vote = _vote(partitioned, baseline, depth7.evaluation, "depth 7")
    summary = _summarise(
        depth8.evaluation,
        depth8_vote,
        depth7.evaluation,
        depth7_vote,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result = {
        "depth7_vote_accuracy": float(
            summary.loc["depth7_identity_vote_20%", "mean_accuracy"]
        ),
        "change_vs_depth8_identity_vote": float(
            summary.loc["depth7_identity_vote_20%", "leader_change"]
        ),
        "passes_gate": bool(
            summary.loc["depth7_identity_vote_20%", "passes_gate"]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


def _vote(partitioned, baseline, identity, depth):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        baseline.random_forest,
        identity,
        weights=[0.44, 0.36, 0.20],
        model_name=f"Accepted ensemble plus 20% {depth} identity CatBoost",
    )


def _validate_evidence(partitioned, baseline, depth8) -> None:
    for evaluation in (
        baseline.xgboost,
        baseline.random_forest,
        baseline.blend,
        depth8,
    ):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Depth-7 identity comparison uses different folds.")


def _summarise(depth8, depth8_vote, depth7, depth7_vote):
    leader_accuracy = float(depth8_vote.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        depth8_vote.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    rows = []
    for key, candidate in {
        "depth8_identity_vote_20%": depth8_vote,
        "depth7_identity_vote_20%": depth7_vote,
        "depth8_identity_catboost": depth8,
        "depth7_identity_catboost": depth7,
    }.items():
        fold_change = candidate.fold_metrics.accuracy - depth8_vote.fold_metrics.accuracy
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
                "passes_gate": key == "depth7_identity_vote_20%" and (
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

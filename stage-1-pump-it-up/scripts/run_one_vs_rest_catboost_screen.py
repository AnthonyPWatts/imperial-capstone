"""Run complete-identity one-vs-rest CatBoost and one equal boundary bag."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "one-vs-rest-catboost-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
IDENTITY_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "catboost-identity-screen"
    / "complete-deferred-identities.joblib"
)
OVR_XGBOOST_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "one-vs-rest-screen"
    / "independent-xgboost-boundaries.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_equal_weight_soft_vote
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from one_vs_rest_catboost_evaluation import (
    evaluate_one_vs_rest_identity_catboost,
)


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
    identity = joblib.load(IDENTITY_PATH)
    ovr_xgboost = joblib.load(OVR_XGBOOST_PATH).evaluation
    _validate_evidence(partitioned, baseline, identity.evaluation, ovr_xgboost)
    cache_path = OUTPUT_DIR / "complete-identity-independent-boundaries.joblib"
    if cache_path.exists():
        trial = joblib.load(cache_path)
        if (
            trial.evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Cached identity OvR trial uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        trial = evaluate_one_vs_rest_identity_catboost(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(trial, cache_path)

    boundary_bag = evaluate_equal_weight_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        identity.evaluation,
        trial.evaluation,
        model_name="Equal multiclass and OvR identity CatBoost bag",
    )
    xgboost_boundary_bag = evaluate_equal_weight_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        ovr_xgboost,
        model_name="Equal multiclass and OvR XGBoost bag",
    )
    leader = _vote(
        partitioned,
        baseline,
        identity.evaluation,
        "Promoted complete-identity vote",
    )
    ovr_vote = _vote(
        partitioned,
        baseline,
        trial.evaluation,
        "OvR-identity-CatBoost vote",
    )
    boundary_bag_vote = _vote(
        partitioned,
        baseline,
        boundary_bag,
        "Equal-CatBoost-boundary-bag vote",
    )
    combined_boundary_vote = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        xgboost_boundary_bag,
        baseline.random_forest,
        boundary_bag,
        weights=[0.44, 0.36, 0.20],
        model_name="Combined XGBoost and CatBoost boundary-bag vote",
    )
    summary = _summarise(
        leader,
        identity.evaluation,
        trial.evaluation,
        ovr_vote,
        boundary_bag,
        boundary_bag_vote,
        combined_boundary_vote,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result = {
        "class_labels": list(trial.class_labels),
        "engineered_features": trial.engineered_features,
        "categorical_features": trial.categorical_features,
        "combined_boundary_vote_accuracy": float(
            summary.loc["combined_boundary_vote", "mean_accuracy"]
        ),
        "change_vs_promoted_identity_vote": float(
            summary.loc["combined_boundary_vote", "leader_change"]
        ),
        "passes_gate": bool(
            summary.loc["combined_boundary_vote", "passes_gate"]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


def _vote(partitioned, baseline, identity_component, name):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        baseline.random_forest,
        identity_component,
        weights=[0.44, 0.36, 0.20],
        model_name=name,
    )


def _validate_evidence(partitioned, baseline, identity, ovr_xgboost) -> None:
    for evaluation in (
        baseline.xgboost,
        baseline.random_forest,
        identity,
        ovr_xgboost,
    ):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Identity OvR comparison uses different folds.")


def _summarise(
    leader,
    identity,
    ovr_identity,
    ovr_vote,
    boundary_bag,
    boundary_bag_vote,
    combined_boundary_vote,
):
    leader_accuracy = float(leader.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        leader.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = []
    for key, candidate in {
        "promoted_identity_vote": leader,
        "multiclass_identity_catboost": identity,
        "one_vs_rest_identity_catboost": ovr_identity,
        "ovr_identity_catboost_vote": ovr_vote,
        "equal_identity_boundary_bag": boundary_bag,
        "boundary_bag_identity_vote": boundary_bag_vote,
        "combined_boundary_vote": combined_boundary_vote,
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
                "passes_gate": key == "combined_boundary_vote" and (
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

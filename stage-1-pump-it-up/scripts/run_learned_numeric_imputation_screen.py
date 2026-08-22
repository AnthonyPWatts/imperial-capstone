"""Run the fixed spatial-height plus learned-amount imputation screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "learned-numeric-imputation-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from learned_numeric_imputation_evaluation import evaluate_learned_numeric_imputation_trial
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
    baseline = joblib.load(
        PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
    )
    identity = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "catboost-identity-screen"
        / "complete-deferred-identities.joblib"
    ).evaluation
    spatial = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "spatial-height-imputation-screen"
        / "ten-neighbour-height.joblib"
    )
    _validate_evidence(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity,
        spatial.xgboost,
        spatial.random_forest,
    )

    cache_path = OUTPUT_DIR / "spatial-height-plus-log-amount.joblib"
    if cache_path.exists():
        learned = joblib.load(cache_path)
        for evaluation in (learned.xgboost, learned.random_forest):
            if (
                evaluation.cross_validation_fingerprint
                != partitioned.cross_validation_fingerprint
            ):
                raise ValueError("Cached learned-imputation trial uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        learned = evaluate_learned_numeric_imputation_trial(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(learned, cache_path)

    leader = _identity_vote(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity,
        "Promoted complete-identity vote",
    )
    learned_vote = _identity_vote(
        partitioned,
        learned.xgboost,
        learned.random_forest,
        identity,
        "Learned-numeric identity vote",
    )
    learned_xgboost_vote = _identity_vote(
        partitioned,
        learned.xgboost,
        baseline.random_forest,
        identity,
        "Learned-numeric XGBoost identity vote",
    )
    learned_forest_vote = _identity_vote(
        partitioned,
        baseline.xgboost,
        learned.random_forest,
        identity,
        "Learned-numeric Random Forest identity vote",
    )
    representation_bag = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        learned.xgboost,
        baseline.random_forest,
        learned.random_forest,
        identity,
        weights=(0.22, 0.22, 0.18, 0.18, 0.20),
        model_name="Equal original/learned-numeric representation bag",
    )
    spatial_representation_bag = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        spatial.xgboost,
        baseline.random_forest,
        spatial.random_forest,
        identity,
        weights=(0.22, 0.22, 0.18, 0.18, 0.20),
        model_name="Equal original/spatial-height representation bag",
    )
    candidates = {
        "promoted_identity_vote": leader,
        "accepted_xgboost": baseline.xgboost,
        "learned_numeric_xgboost": learned.xgboost,
        "accepted_random_forest": baseline.random_forest,
        "learned_numeric_random_forest": learned.random_forest,
        "learned_numeric_full_vote": learned_vote,
        "learned_numeric_xgboost_vote": learned_xgboost_vote,
        "learned_numeric_forest_vote": learned_forest_vote,
        "learned_numeric_representation_bag": representation_bag,
        "spatial_height_representation_bag": spatial_representation_bag,
    }
    summary = _summarise(candidates, leader)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result = {
        "engineered_features": learned.engineered_features,
        "transformed_features_fold_1": learned.transformed_features_fold_1,
        "learned_numeric_bag_accuracy": float(
            summary.loc["learned_numeric_representation_bag", "mean_accuracy"]
        ),
        "change_vs_promoted_identity_vote": float(
            summary.loc["learned_numeric_representation_bag", "leader_change"]
        ),
        "change_vs_spatial_height_bag": float(
            summary.loc["learned_numeric_representation_bag", "mean_accuracy"]
            - summary.loc["spatial_height_representation_bag", "mean_accuracy"]
        ),
        "passes_gate": bool(
            summary.loc["learned_numeric_representation_bag", "passes_gate"]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
        "external_evidence": "https://github.com/drivendataorg/pump-it-up/tree/master/mrbeer",
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


def _identity_vote(partitioned, xgboost, forest, identity, name):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        xgboost,
        forest,
        identity,
        weights=(0.44, 0.36, 0.20),
        model_name=name,
    )


def _validate_evidence(partitioned, *evaluations) -> None:
    for evaluation in evaluations:
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Learned-imputation evidence uses different folds.")


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
                "passes_gate": key == "learned_numeric_representation_bag" and (
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

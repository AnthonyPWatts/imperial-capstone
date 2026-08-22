"""Run the fixed pump-age cohort screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "age-cohort-screen"
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

from age_cohort_evaluation import evaluate_age_cohort_trial
from catboost_identity_evaluation import evaluate_age_cohort_identity_catboost
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
    _validate_evidence(partitioned, baseline, identity)
    cache_path = OUTPUT_DIR / "fixed-age-cohort.joblib"
    if cache_path.exists():
        cohort = joblib.load(cache_path)
        for evaluation in (cohort.xgboost, cohort.random_forest):
            if (
                evaluation.cross_validation_fingerprint
                != partitioned.cross_validation_fingerprint
            ):
                raise ValueError("Cached age-cohort trial uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        cohort = evaluate_age_cohort_trial(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(cohort, cache_path)
    identity_cohort_cache_path = OUTPUT_DIR / "identity-pump-age-cohort.joblib"
    if identity_cohort_cache_path.exists():
        identity_cohort = joblib.load(identity_cohort_cache_path)
        if (
            identity_cohort.evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Cached identity age cohort uses different folds.")
        print(f"Loaded {identity_cohort_cache_path}.", flush=True)
    else:
        identity_cohort = evaluate_age_cohort_identity_catboost(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(identity_cohort, identity_cohort_cache_path)

    leader = _identity_vote(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity,
        "Promoted complete-identity vote",
    )
    cohort_identity_vote = _identity_vote(
        partitioned,
        cohort.xgboost,
        cohort.random_forest,
        identity,
        "Pump-age-cohort identity vote",
    )
    identity_cohort_vote = _identity_vote(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity_cohort.evaluation,
        "Identity-CatBoost pump-age-cohort vote",
    )
    all_cohort_vote = _identity_vote(
        partitioned,
        cohort.xgboost,
        cohort.random_forest,
        identity_cohort.evaluation,
        "All-component pump-age-cohort vote",
    )
    cohort_two_component = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        cohort.xgboost,
        cohort.random_forest,
        weights=[0.55, 0.45],
        model_name="Pump-age-cohort 55:45 vote",
    )
    summary = _summarise(
        leader,
        baseline,
        cohort,
        cohort_two_component,
        cohort_identity_vote,
        identity,
        identity_cohort.evaluation,
        identity_cohort_vote,
        all_cohort_vote,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    screened_keys = [
        "cohort_identity_vote",
        "identity_cohort_vote",
        "all_cohort_vote",
    ]
    best_key = summary.loc[screened_keys, "mean_accuracy"].idxmax()
    result = {
        "engineered_features": cohort.engineered_features,
        "transformed_features_fold_1": cohort.transformed_features_fold_1,
        "identity_catboost_engineered_features": (
            identity_cohort.engineered_features
        ),
        "identity_catboost_categorical_features": (
            identity_cohort.categorical_features
        ),
        "best_candidate": best_key,
        "best_candidate_accuracy": float(summary.loc[best_key, "mean_accuracy"]),
        "change_vs_promoted_identity_vote": float(
            summary.loc[best_key, "leader_change"]
        ),
        "passes_gate": bool(summary.loc[best_key, "passes_gate"]),
        "local_test_opened": False,
        "competition_predictions_generated": False,
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
        weights=[0.44, 0.36, 0.20],
        model_name=name,
    )


def _validate_evidence(partitioned, baseline, identity) -> None:
    for evaluation in (baseline.xgboost, baseline.random_forest, identity):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Age-cohort comparison uses different folds.")


def _summarise(
    leader,
    baseline,
    cohort,
    cohort_two_component,
    cohort_identity_vote,
    identity,
    identity_cohort,
    identity_cohort_vote,
    all_cohort_vote,
):
    leader_accuracy = float(leader.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        leader.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = []
    for key, candidate in {
        "promoted_identity_vote": leader,
        "accepted_xgboost": baseline.xgboost,
        "cohort_xgboost": cohort.xgboost,
        "accepted_random_forest": baseline.random_forest,
        "cohort_random_forest": cohort.random_forest,
        "cohort_two_component_vote": cohort_two_component,
        "cohort_identity_vote": cohort_identity_vote,
        "complete_identity_catboost": identity,
        "cohort_identity_catboost": identity_cohort,
        "identity_cohort_vote": identity_cohort_vote,
        "all_cohort_vote": all_cohort_vote,
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
                "passes_gate": key in {
                    "cohort_identity_vote",
                    "identity_cohort_vote",
                    "all_cohort_vote",
                } and (
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

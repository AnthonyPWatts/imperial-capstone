"""Run the fixed context-qualified native-identity CatBoost screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "catboost-identity-context-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
COMPLETE_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "catboost-identity-screen"
    / "complete-deferred-identities.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_evaluation import evaluate_context_identity_catboost
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
    complete = joblib.load(COMPLETE_PATH)
    _validate_evidence(partitioned, baseline, complete)
    cache_path = OUTPUT_DIR / "context-deferred-identities.joblib"
    if cache_path.exists():
        context = joblib.load(cache_path)
        if (
            context.evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Cached context CatBoost uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        context = evaluate_context_identity_catboost(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(context, cache_path)

    complete_vote = _identity_vote(partitioned, baseline, complete.evaluation)
    context_vote = _identity_vote(partitioned, baseline, context.evaluation)
    summary = _summarise(
        baseline.blend,
        complete.evaluation,
        complete_vote,
        context.evaluation,
        context_vote,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result = {
        "engineered_features": context.engineered_features,
        "categorical_features": context.categorical_features,
        "context_vote_accuracy": float(
            summary.loc["context_identity_vote_20%", "mean_accuracy"]
        ),
        "change_vs_complete_identity_vote": float(
            summary.loc["context_identity_vote_20%", "leader_change"]
        ),
        "passes_gate_vs_complete_identity_vote": bool(
            summary.loc["context_identity_vote_20%", "passes_gate"]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


def _identity_vote(partitioned, baseline, catboost):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        baseline.random_forest,
        catboost,
        weights=[0.44, 0.36, 0.20],
        model_name="44% XGBoost + 36% Random Forest + 20% identity CatBoost",
    )


def _validate_evidence(partitioned, baseline, complete) -> None:
    for evaluation in (
        baseline.xgboost,
        baseline.random_forest,
        baseline.blend,
        complete.evaluation,
    ):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Context CatBoost comparison uses different folds.")


def _summarise(accepted, complete, complete_vote, context, context_vote):
    leader_accuracy = float(complete_vote.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        complete_vote.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    rows = []
    for key, candidate in {
        "accepted_baseline": accepted,
        "complete_identity_catboost": complete,
        "complete_identity_vote_20%": complete_vote,
        "context_identity_catboost": context,
        "context_identity_vote_20%": context_vote,
    }.items():
        fold_change = (
            candidate.fold_metrics.accuracy
            - complete_vote.fold_metrics.accuracy
        )
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
                "passes_gate": key == "context_identity_vote_20%" and (
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

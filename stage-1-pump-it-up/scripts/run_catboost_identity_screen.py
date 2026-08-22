"""Run the complete deferred-identity CatBoost screen."""

from __future__ import annotations

import argparse
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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "catboost-identity-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
CURRENT_CATBOOST_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "gpu-model-screen"
    / "catboost-d8-current-features.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_evaluation import evaluate_complete_identity_catboost
from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned_data = partition_modelling_data(_load_modelling_data())
    baseline = joblib.load(BASELINE_PATH)
    current_catboost = _load_current_catboost()
    _validate_evidence(partitioned_data, baseline, current_catboost)
    path = OUTPUT_DIR / "complete-deferred-identities.joblib"
    if path.exists() and not args.force:
        trial = joblib.load(path)
        if (
            trial.evaluation.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Cached identity CatBoost uses different folds.")
        print(f"Loaded {path}.", flush=True)
    else:
        trial = evaluate_complete_identity_catboost(
            partitioned_data,
            make_cross_validation(partitioned_data),
        )
        joblib.dump(trial, path)

    contributions = {}
    cross_validation = make_cross_validation(partitioned_data)
    for weight in (0.05, 0.10, 0.15, 0.20):
        contributions[f"identity_catboost_{weight:.0%}"] = (
            evaluate_weighted_soft_vote(
                partitioned_data,
                cross_validation,
                baseline.xgboost,
                baseline.random_forest,
                trial.evaluation,
                weights=[
                    0.55 * (1.0 - weight),
                    0.45 * (1.0 - weight),
                    weight,
                ],
                model_name=(
                    f"{weight:.0%} identity CatBoost + accepted ensemble"
                ),
            )
        )
    summary = _summarise(
        baseline.blend,
        current_catboost,
        trial.evaluation,
        contributions,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result = {
        "engineered_features": trial.engineered_features,
        "categorical_features": trial.categorical_features,
        "best_candidate": str(summary.index[0]),
        "best_accuracy": float(summary.iloc[0]["mean_accuracy"]),
        "selection_gate_passers": int(summary["passes_gate"].sum()),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


def _load_modelling_data():
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _load_current_catboost():
    cached = joblib.load(CURRENT_CATBOOST_PATH)
    if not isinstance(cached, dict) or "evaluation" not in cached:
        raise ValueError("Current CatBoost cache has an unexpected structure.")
    return cached["evaluation"]


def _validate_evidence(partitioned_data, baseline, current_catboost) -> None:
    for evaluation in (
        baseline.xgboost,
        baseline.random_forest,
        baseline.blend,
        current_catboost,
    ):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("CatBoost comparison uses different frozen folds.")


def _summarise(accepted, current_catboost, identity_catboost, contributions):
    baseline_accuracy = float(accepted.metric_summary.loc["accuracy", "mean"])
    baseline_repair = float(
        accepted.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    candidates = {
        "accepted_baseline": accepted,
        "current_catboost_d8": current_catboost,
        "complete_identity_catboost_d8": identity_catboost,
        **contributions,
    }
    rows = []
    for key, candidate in candidates.items():
        fold_change = candidate.fold_metrics.accuracy - accepted.fold_metrics.accuracy
        accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        change = accuracy - baseline_accuracy
        repair_change = repair - baseline_repair
        rows.append(
            {
                "candidate": key,
                "mean_accuracy": accuracy,
                "accuracy_change": change,
                "fold_wins": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall_change": repair_change,
                "passes_gate": key != "accepted_baseline" and (
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
        "accuracy_change",
        "worst_fold_change",
        "repair_recall_change",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Run and persist the cross-fitted local spatial-outcome screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "spatial-outcome-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from spatial_outcome_evaluation import evaluate_spatial_outcomes
from spatial_outcome_evaluation import summarise_spatial_outcomes


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned_data = partition_modelling_data(_load_modelling_data())
    baseline = joblib.load(BASELINE_PATH)
    _validate_baseline(partitioned_data, baseline)
    path = OUTPUT_DIR / "spatial-k10-s20.joblib"
    if path.exists() and not args.force:
        trial = joblib.load(path)
        if (
            trial.blend.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Cached spatial trial uses different frozen folds.")
        print(f"Loaded {path}.", flush=True)
    else:
        trial = evaluate_spatial_outcomes(
            partitioned_data,
            make_cross_validation(partitioned_data),
            baseline.blend,
        )
        joblib.dump(trial, path)

    summary = summarise_spatial_outcomes(baseline.blend, trial)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    crosses = _evaluate_component_crosses(partitioned_data, baseline, trial)
    cross_summary = _summarise_crosses(baseline.blend, crosses)
    cross_summary.to_csv(OUTPUT_DIR / "component-cross-summary.csv")
    result = {
        "selection_gate_passers": int(summary["passes_gate"].sum()),
        "best_candidate": str(summary.index[0]),
        "best_accuracy": float(summary.iloc[0]["mean_accuracy"]),
        "best_component_cross": str(cross_summary.index[0]),
        "best_component_cross_accuracy": float(
            cross_summary.iloc[0]["mean_accuracy"]
        ),
        "transformed_features_fold_1": trial.transformed_features_fold_1,
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)
    print("\nFixed component crosses:", flush=True)
    print(_format_summary(cross_summary), flush=True)
    print(
        "\nThe local test remains closed and no competition prediction was made.",
        flush=True,
    )


def _load_modelling_data():
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _validate_baseline(partitioned_data, baseline) -> None:
    for evaluation in (baseline.xgboost, baseline.random_forest, baseline.blend):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Accepted baseline uses different frozen folds.")


def _evaluate_component_crosses(partitioned_data, baseline, trial):
    cross_validation = make_cross_validation(partitioned_data)
    return {
        "spatial_xgboost__accepted_random_forest": evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            trial.xgboost,
            baseline.random_forest,
            weights=[0.55, 0.45],
            model_name="Spatial XGBoost + accepted Random Forest",
        ),
        "accepted_xgboost__spatial_random_forest": evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            baseline.xgboost,
            trial.random_forest,
            weights=[0.55, 0.45],
            model_name="Accepted XGBoost + spatial Random Forest",
        ),
    }


def _summarise_crosses(accepted, crosses) -> pd.DataFrame:
    baseline_accuracy = float(accepted.metric_summary.loc["accuracy", "mean"])
    baseline_repair = float(
        accepted.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = []
    for key, candidate in crosses.items():
        fold_change = (
            candidate.fold_metrics["accuracy"]
            - accepted.fold_metrics["accuracy"]
        )
        accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        accuracy_change = accuracy - baseline_accuracy
        repair_change = repair - baseline_repair
        rows.append(
            {
                "candidate": key,
                "model_name": candidate.model_name,
                "mean_accuracy": accuracy,
                "accuracy_change": accuracy_change,
                "fold_wins": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall": repair,
                "repair_recall_change": repair_change,
                "passes_gate": (
                    accuracy_change >= 0.001
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
    display = summary.loc[
        :,
        [
            "mean_accuracy",
            "accuracy_change",
            "fold_wins",
            "worst_fold_change",
            "repair_recall_change",
            "passes_gate",
        ],
    ].copy()
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

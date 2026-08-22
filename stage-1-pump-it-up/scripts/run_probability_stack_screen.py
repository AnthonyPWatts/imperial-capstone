"""Run one constrained meta-model over the six archive components."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "probability-stack-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from probability_stack_evaluation import evaluate_cross_fitted_probability_stack


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
    frequency = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "categorical-frequency-forest-screen"
        / "all-categorical-occurrence-counts.joblib"
    ).random_forest
    spatial = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "spatial-height-imputation-screen"
        / "ten-neighbour-height.joblib"
    )
    components = (
        baseline.xgboost,
        spatial.xgboost,
        baseline.random_forest,
        frequency,
        spatial.random_forest,
        identity,
    )
    cross_validation = make_cross_validation(partitioned)
    archive = evaluate_weighted_soft_vote(
        partitioned,
        cross_validation,
        *components,
        weights=(0.33, 0.11, 0.18, 0.09, 0.09, 0.20),
        model_name="Existing archive synthesis",
    )
    stack = evaluate_cross_fitted_probability_stack(
        partitioned,
        cross_validation,
        *components,
    )
    summary = _summarise(archive, stack)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    stack.diagnostics.to_csv(OUTPUT_DIR / "fold-diagnostics.csv")
    result = {
        "components": [component.model_name for component in components],
        "meta_features": int(stack.diagnostics["meta_features"].iloc[0]),
        "stack_accuracy": float(summary.loc["probability_stack", "mean_accuracy"]),
        "change_vs_archive": float(
            summary.loc["probability_stack", "incumbent_change"]
        ),
        "fold_wins": int(
            summary.loc["probability_stack", "fold_wins_vs_incumbent"]
        ),
        "passes_gate": bool(summary.loc["probability_stack", "passes_gate"]),
        "strict_nested_base_refit_run": False,
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)
    print("\nDiagnostics:\n", stack.diagnostics.to_string(), flush=True)


def _summarise(archive, stack):
    incumbent_accuracy = float(archive.metric_summary.loc["accuracy", "mean"])
    incumbent_repair = float(
        archive.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    rows = []
    for key, candidate in (
        ("existing_archive_synthesis", archive),
        ("probability_stack", stack),
    ):
        fold_change = candidate.fold_metrics.accuracy - archive.fold_metrics.accuracy
        accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        change = accuracy - incumbent_accuracy
        repair_change = repair - incumbent_repair
        rows.append(
            {
                "candidate": key,
                "mean_accuracy": accuracy,
                "incumbent_change": change,
                "fold_wins_vs_incumbent": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall_change": repair_change,
                "passes_gate": key == "probability_stack"
                and change >= 0.001
                and int(fold_change.gt(0).sum()) >= 3
                and float(fold_change.min()) >= -0.0025
                and repair_change >= -0.02,
            }
        )
    return pd.DataFrame(rows).set_index("candidate")


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in (
        "mean_accuracy",
        "incumbent_change",
        "worst_fold_change",
        "repair_recall_change",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


if __name__ == "__main__":
    main()

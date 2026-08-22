"""Run two cumulative accepted-spec XGBoost boundaries and archive crosses."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "ordinal-xgboost-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import compare_candidate_diversity
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from ordinal_xgboost_evaluation import evaluate_ordinal_xgboost


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
    _validate_evidence(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity,
        frequency,
        spatial.xgboost,
        spatial.random_forest,
    )

    cache_path = OUTPUT_DIR / "cumulative-xgboost-boundaries.joblib"
    if cache_path.exists():
        trial = joblib.load(cache_path)
        _validate_evidence(partitioned, trial.evaluation)
        print(f"Loaded {cache_path}.", flush=True)
    else:
        trial = evaluate_ordinal_xgboost(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(trial, cache_path)

    archive = _archive_vote(
        partitioned,
        baseline,
        baseline.xgboost,
        frequency,
        spatial,
        identity,
        "Existing archive synthesis",
    )
    archive_substitution = _archive_vote(
        partitioned,
        baseline,
        trial.evaluation,
        frequency,
        spatial,
        identity,
        "Cumulative XGBoost archive substitution",
    )
    archive_bag = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        trial.evaluation,
        spatial.xgboost,
        baseline.random_forest,
        frequency,
        spatial.random_forest,
        identity,
        weights=(0.165, 0.165, 0.11, 0.18, 0.09, 0.09, 0.20),
        model_name="Equal multiclass/cumulative XGBoost archive bag",
    )
    candidates = {
        "accepted_multiclass_xgboost": baseline.xgboost,
        "cumulative_xgboost": trial.evaluation,
        "existing_archive_synthesis": archive,
        "ordinal_archive_substitution": archive_substitution,
        "ordinal_archive_bag": archive_bag,
    }
    summary = _summarise(candidates, archive)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    trial.evaluation.diagnostics.to_csv(OUTPUT_DIR / "fold-diagnostics.csv")
    diversity = compare_candidate_diversity(
        partitioned,
        baseline.xgboost,
        trial.evaluation,
        archive,
        archive_substitution,
        archive_bag,
    )
    diversity.to_csv(OUTPUT_DIR / "diversity.csv")
    result = {
        "class_labels": list(trial.class_labels),
        "cumulative_xgboost_accuracy": float(
            summary.loc["cumulative_xgboost", "mean_accuracy"]
        ),
        "best_archive_accuracy": float(
            summary.loc[
                ["ordinal_archive_substitution", "ordinal_archive_bag"],
                "mean_accuracy",
            ].max()
        ),
        "best_change_vs_archive": float(
            summary.loc[
                ["ordinal_archive_substitution", "ordinal_archive_bag"],
                "incumbent_change",
            ].max()
        ),
        "passes_gate": bool(summary["passes_gate"].any()),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)
    print("\nDiagnostics:\n", trial.evaluation.diagnostics.to_string(), flush=True)
    print("\nDiversity:\n", diversity.to_string(), flush=True)


def _archive_vote(
    partitioned,
    baseline,
    primary_xgboost,
    frequency,
    spatial,
    identity,
    name,
):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        primary_xgboost,
        spatial.xgboost,
        baseline.random_forest,
        frequency,
        spatial.random_forest,
        identity,
        weights=(0.33, 0.11, 0.18, 0.09, 0.09, 0.20),
        model_name=name,
    )


def _validate_evidence(partitioned, *evaluations) -> None:
    for evaluation in evaluations:
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Ordinal XGBoost evidence uses different folds.")


def _summarise(candidates, incumbent):
    incumbent_accuracy = float(incumbent.metric_summary.loc["accuracy", "mean"])
    incumbent_repair = float(
        incumbent.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    eligible = {"ordinal_archive_substitution", "ordinal_archive_bag"}
    rows = []
    for key, candidate in candidates.items():
        fold_change = (
            candidate.fold_metrics.accuracy - incumbent.fold_metrics.accuracy
        )
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
                "passes_gate": key in eligible
                and change >= 0.001
                and int(fold_change.gt(0).sum()) >= 3
                and float(fold_change.min()) >= -0.0025
                and repair_change >= -0.02,
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
        "incumbent_change",
        "worst_fold_change",
        "repair_recall_change",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


if __name__ == "__main__":
    main()

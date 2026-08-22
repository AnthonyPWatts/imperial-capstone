"""Run fixed target-free spatial density in archived deep XGBoost."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "spatial-density-screen"
DENSITY_SEED = 20260822
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import compare_candidate_diversity
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from top_common_identity_evaluation import evaluate_archived_deep_xgboost
from top_common_identity_evaluation import make_top_common_spatial_density_preprocessor


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
    deep = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "archived-deep-xgboost-screen"
        / f"depth-17-seed-{DENSITY_SEED}.joblib"
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
        deep,
        identity,
        frequency,
        spatial.xgboost,
        spatial.random_forest,
    )

    cache_path = OUTPUT_DIR / f"deep-density-seed-{DENSITY_SEED}.joblib"
    if cache_path.exists():
        density = joblib.load(cache_path)
        _validate_evidence(partitioned, density)
        print(f"Loaded {cache_path}.", flush=True)
    else:
        density = evaluate_archived_deep_xgboost(
            partitioned,
            make_cross_validation(partitioned),
            seed=DENSITY_SEED,
            preprocessor_factory=make_top_common_spatial_density_preprocessor,
            representation_name="top-50 identities plus spatial density",
            feature_policy="accepted_plus_top_50_identities_and_spatial_density",
        )
        joblib.dump(density, cache_path)

    archive = _archive_vote(
        partitioned,
        baseline,
        baseline.xgboost,
        frequency,
        spatial,
        identity,
        "Existing archive synthesis",
    )
    deep_archive = _archive_vote(
        partitioned,
        baseline,
        deep,
        frequency,
        spatial,
        identity,
        "Existing archived deep XGBoost substitution",
    )
    density_archive = _archive_vote(
        partitioned,
        baseline,
        density,
        frequency,
        spatial,
        identity,
        "Spatial-density deep XGBoost substitution",
    )
    candidates = {
        "archived_deep_xgboost": deep,
        "spatial_density_deep_xgboost": density,
        "existing_archive_synthesis": archive,
        "existing_deep_archive_substitution": deep_archive,
        "spatial_density_archive_substitution": density_archive,
    }
    summary = _summarise(candidates, archive)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    density.diagnostics.to_csv(OUTPUT_DIR / "fold-diagnostics.csv")
    diversity = compare_candidate_diversity(
        partitioned,
        deep,
        density,
        archive,
        deep_archive,
        density_archive,
    )
    diversity.to_csv(OUTPUT_DIR / "diversity.csv")
    result = {
        "neighbour_ranks": [1, 10],
        "spatial_density_deep_accuracy": float(
            summary.loc["spatial_density_deep_xgboost", "mean_accuracy"]
        ),
        "archive_accuracy": float(
            summary.loc["spatial_density_archive_substitution", "mean_accuracy"]
        ),
        "archive_change": float(
            summary.loc[
                "spatial_density_archive_substitution",
                "incumbent_change",
            ]
        ),
        "passes_gate": bool(
            summary.loc["spatial_density_archive_substitution", "passes_gate"]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)
    print("\nDiagnostics:\n", density.diagnostics.to_string(), flush=True)
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
            raise ValueError("Spatial-density evidence uses different folds.")


def _summarise(candidates, incumbent):
    incumbent_accuracy = float(incumbent.metric_summary.loc["accuracy", "mean"])
    incumbent_repair = float(
        incumbent.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
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
        eligible = key == "spatial_density_archive_substitution"
        rows.append(
            {
                "candidate": key,
                "mean_accuracy": accuracy,
                "incumbent_change": change,
                "fold_wins_vs_incumbent": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall_change": repair_change,
                "passes_gate": eligible
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

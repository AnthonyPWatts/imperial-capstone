"""Run the archived depth-17 XGBoost specification on top-50 identities."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "archived-deep-xgboost-screen"
ARCHIVED_SEEDS = (20260822, 20260823, 20260824)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import compare_candidate_diversity
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from top_common_identity_evaluation import evaluate_archived_deep_xgboost


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
    top_common = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "top-common-identity-screen"
        / "top-50-deferred-identities.joblib"
    ).xgboost
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
        top_common,
        identity,
        frequency,
        spatial.xgboost,
        spatial.random_forest,
    )

    deep_evaluations = []
    for seed in ARCHIVED_SEEDS:
        cache_path = OUTPUT_DIR / f"depth-17-seed-{seed}.joblib"
        if cache_path.exists():
            deep = joblib.load(cache_path)
            _validate_evidence(partitioned, deep)
            print(f"Loaded {cache_path}.", flush=True)
        else:
            deep = evaluate_archived_deep_xgboost(
                partitioned,
                make_cross_validation(partitioned),
                seed=seed,
            )
            joblib.dump(deep, cache_path)
        deep_evaluations.append(deep)
    deep = deep_evaluations[0]
    deep_seed_bag = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        *deep_evaluations,
        weights=(1 / 3, 1 / 3, 1 / 3),
        model_name="Three-seed archived depth-17 XGBoost bag",
    )

    archive = _archive_vote(
        partitioned,
        baseline,
        baseline.xgboost,
        frequency,
        spatial,
        identity,
        "Existing archive synthesis",
    )
    substitution = _archive_vote(
        partitioned,
        baseline,
        deep,
        frequency,
        spatial,
        identity,
        "Archived deep XGBoost archive substitution",
    )
    seed_bag_substitution = _archive_vote(
        partitioned,
        baseline,
        deep_seed_bag,
        frequency,
        spatial,
        identity,
        "Three-seed deep XGBoost archive substitution",
    )
    top_common_bag = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        top_common,
        spatial.xgboost,
        baseline.random_forest,
        frequency,
        spatial.random_forest,
        identity,
        weights=(0.165, 0.165, 0.11, 0.18, 0.09, 0.09, 0.20),
        model_name="Existing equal accepted/top-50 XGBoost archive bag",
    )
    deep_bag = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        deep,
        spatial.xgboost,
        baseline.random_forest,
        frequency,
        spatial.random_forest,
        identity,
        weights=(0.165, 0.165, 0.11, 0.18, 0.09, 0.09, 0.20),
        model_name="Equal accepted/deep XGBoost archive bag",
    )
    candidates = {
        "accepted_xgboost": baseline.xgboost,
        "top_50_depth_8_xgboost": top_common,
        "archived_depth_17_seed_20260822": deep_evaluations[0],
        "archived_depth_17_seed_20260823": deep_evaluations[1],
        "archived_depth_17_seed_20260824": deep_evaluations[2],
        "archived_depth_17_seed_bag": deep_seed_bag,
        "existing_archive_synthesis": archive,
        "existing_top_50_archive_bag": top_common_bag,
        "deep_archive_substitution": substitution,
        "deep_seed_bag_archive_substitution": seed_bag_substitution,
        "deep_archive_bag": deep_bag,
    }
    summary = _summarise(candidates, archive)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    deep.diagnostics.to_csv(OUTPUT_DIR / "fold-diagnostics.csv")
    diversity = compare_candidate_diversity(
        partitioned,
        baseline.xgboost,
        top_common,
        *deep_evaluations,
        deep_seed_bag,
        archive,
        substitution,
        seed_bag_substitution,
        deep_bag,
    )
    diversity.to_csv(OUTPUT_DIR / "diversity.csv")
    result = {
        "seeds": list(ARCHIVED_SEEDS),
        "iterations": 600,
        "archived_depth_17_seed_bag_accuracy": float(
            summary.loc["archived_depth_17_seed_bag", "mean_accuracy"]
        ),
        "best_archive_accuracy": float(
            summary.loc[
                [
                    "deep_archive_substitution",
                    "deep_seed_bag_archive_substitution",
                    "deep_archive_bag",
                ],
                "mean_accuracy",
            ].max()
        ),
        "best_change_vs_archive": float(
            summary.loc[
                [
                    "deep_archive_substitution",
                    "deep_seed_bag_archive_substitution",
                    "deep_archive_bag",
                ],
                "incumbent_change",
            ].max()
        ),
        "passes_gate": bool(summary["passes_gate"].any()),
        "seed_bag_run": True,
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)
    print("\nDiagnostics:\n", deep.diagnostics.to_string(), flush=True)
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
            raise ValueError("Archived deep XGBoost evidence uses different folds.")


def _summarise(candidates, incumbent):
    incumbent_accuracy = float(incumbent.metric_summary.loc["accuracy", "mean"])
    incumbent_repair = float(
        incumbent.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    eligible = {
        "deep_archive_substitution",
        "deep_seed_bag_archive_substitution",
        "deep_archive_bag",
    }
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

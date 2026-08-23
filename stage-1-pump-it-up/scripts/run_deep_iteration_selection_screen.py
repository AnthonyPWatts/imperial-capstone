"""Select deep-XGBoost tree counts inside each frozen development fold."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "deep-iteration-selection-screen"
CACHE_PATH = OUTPUT_DIR / "inner-stopped-deep-seed-20260822.joblib"
SEED = 20260822
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from top_common_identity_evaluation import make_top_common_identity_preprocessor


ARCHIVE_WEIGHTS = (0.33, 0.11, 0.18, 0.09, 0.09, 0.20)


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
    fixed_deep = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "archived-deep-xgboost-screen"
        / f"depth-17-seed-{SEED}.joblib"
    )
    _validate_evidence(
        partitioned,
        baseline.random_forest,
        identity,
        frequency,
        spatial.xgboost,
        spatial.random_forest,
        fixed_deep,
    )

    if CACHE_PATH.exists():
        selected_deep = joblib.load(CACHE_PATH)
        _validate_evidence(partitioned, selected_deep)
        print(f"Loaded {CACHE_PATH}.", flush=True)
    else:
        spec = make_xgboost_spec(variant="archived depth 17", seed=SEED)
        selected_deep = evaluate_gpu_candidate(
            spec,
            partitioned,
            make_cross_validation(partitioned),
            preprocessor_factory=make_top_common_identity_preprocessor,
        )
        joblib.dump(selected_deep, CACHE_PATH)

    incumbent = _archive_vote(
        partitioned,
        baseline,
        fixed_deep,
        frequency,
        spatial,
        identity,
        "Existing fixed-600 deep archive substitution",
    )
    candidate = _archive_vote(
        partitioned,
        baseline,
        selected_deep,
        frequency,
        spatial,
        identity,
        "Inner-stopped deep archive substitution",
    )
    summary = _summarise(
        {
            "fixed_600_deep_xgboost": fixed_deep,
            "inner_stopped_deep_xgboost": selected_deep,
            "fixed_600_deep_archive": incumbent,
            "inner_stopped_deep_archive": candidate,
        },
        incumbent,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    selected_deep.diagnostics.to_csv(OUTPUT_DIR / "fold-diagnostics.csv")
    result = {
        "seed": SEED,
        "selected_iterations": [
            int(value) for value in selected_deep.diagnostics.selected_iterations
        ],
        "candidate_accuracy": float(
            summary.loc["inner_stopped_deep_archive", "mean_accuracy"]
        ),
        "change_vs_fixed_600": float(
            summary.loc["inner_stopped_deep_archive", "incumbent_change"]
        ),
        "passes_gate": bool(
            summary.loc["inner_stopped_deep_archive", "passes_gate"]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


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
        weights=ARCHIVE_WEIGHTS,
        model_name=name,
    )


def _validate_evidence(partitioned, *evaluations) -> None:
    for evaluation in evaluations:
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Deep iteration evidence uses different folds.")


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
        rows.append(
            {
                "candidate": key,
                "mean_accuracy": accuracy,
                "incumbent_change": change,
                "fold_wins_vs_incumbent": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall_change": repair_change,
                "passes_gate": key == "inner_stopped_deep_archive"
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

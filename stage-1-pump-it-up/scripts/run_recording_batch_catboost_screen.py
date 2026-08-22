"""Run one exact recording-date native CatBoost screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "recording-batch-catboost-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from recording_batch_catboost_evaluation import evaluate_recording_batch_catboost


ARCHIVE_SYNTHESIS_WEIGHTS = (0.33, 0.11, 0.18, 0.09, 0.09, 0.20)


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
    frequency_forest = joblib.load(
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
        frequency_forest,
        spatial.xgboost,
        spatial.random_forest,
    )

    cache_path = OUTPUT_DIR / "exact-recording-batch.joblib"
    if cache_path.exists():
        batch = joblib.load(cache_path)
        if (
            batch.evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Cached recording-batch CatBoost uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        batch = evaluate_recording_batch_catboost(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(batch, cache_path)

    leader = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        baseline.random_forest,
        identity,
        weights=(0.44, 0.36, 0.20),
        model_name="Promoted complete-identity vote",
    )
    batch_vote = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        baseline.random_forest,
        batch.evaluation,
        weights=(0.44, 0.36, 0.20),
        model_name="Recording-batch identity vote",
    )
    identity_batch_bag = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        baseline.random_forest,
        identity,
        batch.evaluation,
        weights=(0.44, 0.36, 0.10, 0.10),
        model_name="Equal identity/recording-batch CatBoost bag",
    )
    archive_identity_vote = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        spatial.xgboost,
        baseline.random_forest,
        frequency_forest,
        spatial.random_forest,
        identity,
        weights=ARCHIVE_SYNTHESIS_WEIGHTS,
        model_name="Existing archive-derived synthesis",
    )
    archive_batch_vote = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        spatial.xgboost,
        baseline.random_forest,
        frequency_forest,
        spatial.random_forest,
        batch.evaluation,
        weights=ARCHIVE_SYNTHESIS_WEIGHTS,
        model_name="Archive synthesis with recording-batch CatBoost",
    )
    archive_identity_batch_bag = evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        spatial.xgboost,
        baseline.random_forest,
        frequency_forest,
        spatial.random_forest,
        identity,
        batch.evaluation,
        weights=(0.33, 0.11, 0.18, 0.09, 0.09, 0.10, 0.10),
        model_name="Archive synthesis with equal identity/batch CatBoost bag",
    )
    candidates = {
        "promoted_identity_vote": leader,
        "complete_identity_catboost": identity,
        "recording_batch_catboost": batch.evaluation,
        "recording_batch_vote": batch_vote,
        "identity_recording_batch_bag": identity_batch_bag,
        "archive_identity_vote": archive_identity_vote,
        "archive_recording_batch_vote": archive_batch_vote,
        "archive_identity_recording_batch_bag": archive_identity_batch_bag,
    }
    summary = _summarise(candidates, leader)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    support = _support_audit(partitioned, make_cross_validation(partitioned))
    support.to_csv(OUTPUT_DIR / "fold-support-audit.csv", index=False)
    result = {
        "engineered_features": batch.engineered_features,
        "categorical_features": batch.categorical_features,
        "unique_development_dates": int(
            pd.to_datetime(
                partitioned.X_development["date_recorded"],
                errors="coerce",
            ).nunique()
        ),
        "batch_vote_accuracy": float(
            summary.loc["recording_batch_vote", "mean_accuracy"]
        ),
        "batch_bag_accuracy": float(
            summary.loc["identity_recording_batch_bag", "mean_accuracy"]
        ),
        "best_change_vs_promoted_identity_vote": float(
            summary.loc[
                [
                    "recording_batch_vote",
                    "identity_recording_batch_bag",
                    "archive_recording_batch_vote",
                    "archive_identity_recording_batch_bag",
                ],
                "leader_change",
            ].max()
        ),
        "passes_gate": bool(
            summary.loc[
                [
                    "recording_batch_vote",
                    "identity_recording_batch_bag",
                    "archive_recording_batch_vote",
                    "archive_identity_recording_batch_bag",
                ],
                "passes_gate",
            ].any()
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
        "external_evidence": (
            "https://github.com/drivendataorg/pump-it-up/tree/master/"
            "benedekrozemberczki"
        ),
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


def _support_audit(partitioned, cross_validation) -> pd.DataFrame:
    dates = pd.to_datetime(
        partitioned.X_development["date_recorded"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    rows = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        training = dates.iloc[training_positions]
        validation = dates.iloc[validation_positions]
        support = training.value_counts(dropna=False)
        rows.append(
            {
                "fold": fold_number,
                "training_dates": int(training.nunique(dropna=False)),
                "validation_dates": int(validation.nunique(dropna=False)),
                "validation_unseen_share": float(
                    (~validation.isin(support.index)).mean()
                ),
                "validation_support_below_20_share": float(
                    validation.map(support).fillna(0).lt(20).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_evidence(partitioned, *evaluations) -> None:
    for evaluation in evaluations:
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Recording-batch evidence uses different folds.")


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
                "passes_gate": key in {
                    "recording_batch_vote",
                    "identity_recording_batch_bag",
                    "archive_recording_batch_vote",
                    "archive_identity_recording_batch_bag",
                }
                and (
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

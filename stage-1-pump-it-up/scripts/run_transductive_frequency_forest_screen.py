"""Run the supplied-covariate transductive frequency-forest screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "transductive-frequency-forest-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from categorical_frequency_forest_evaluation import FREQUENCY_MINIMUM_SUPPORT
from categorical_frequency_forest_evaluation import FREQUENCY_SOURCE_FEATURES
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from target_encoding_features import normalise_identity
from transductive_frequency_forest_evaluation import evaluate_transductive_frequency_forest


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = partition_modelling_data(modelling)
    reference_X = pd.concat(
        [partitioned.X_development, modelling.X_competition],
        ignore_index=True,
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
    inductive_frequency = joblib.load(
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
        inductive_frequency,
        spatial.xgboost,
        spatial.random_forest,
    )

    cache_path = OUTPUT_DIR / "development-plus-competition-counts.joblib"
    if cache_path.exists():
        transductive = joblib.load(cache_path)
        if (
            transductive.random_forest.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Cached transductive forest uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        transductive = evaluate_transductive_frequency_forest(
            partitioned,
            make_cross_validation(partitioned),
            reference_X,
        )
        joblib.dump(transductive, cache_path)

    leader = _vote(
        partitioned,
        (baseline.xgboost, baseline.random_forest, identity),
        (0.44, 0.36, 0.20),
        "Promoted complete-identity vote",
    )
    inductive_archive_synthesis = _vote(
        partitioned,
        (
            baseline.xgboost,
            spatial.xgboost,
            baseline.random_forest,
            inductive_frequency,
            spatial.random_forest,
            identity,
        ),
        (0.33, 0.11, 0.18, 0.09, 0.09, 0.20),
        "Inductive archive-derived synthesis",
    )
    transductive_frequency_vote = _vote(
        partitioned,
        (
            baseline.xgboost,
            baseline.random_forest,
            transductive.random_forest,
            identity,
        ),
        (0.44, 0.18, 0.18, 0.20),
        "Transductive frequency-forest representation vote",
    )
    transductive_archive_synthesis = _vote(
        partitioned,
        (
            baseline.xgboost,
            spatial.xgboost,
            baseline.random_forest,
            transductive.random_forest,
            spatial.random_forest,
            identity,
        ),
        (0.33, 0.11, 0.18, 0.09, 0.09, 0.20),
        "Transductive archive-derived synthesis",
    )
    candidates = {
        "promoted_identity_vote": leader,
        "inductive_frequency_forest": inductive_frequency,
        "transductive_frequency_forest": transductive.random_forest,
        "transductive_frequency_vote": transductive_frequency_vote,
        "inductive_archive_synthesis": inductive_archive_synthesis,
        "transductive_archive_synthesis": transductive_archive_synthesis,
    }
    summary = _summarise(candidates, leader)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    rare_crossing = _summarise_rare_threshold_crossing(
        partitioned,
        reference_X,
    )
    rare_crossing.to_csv(OUTPUT_DIR / "rare-threshold-crossing.csv", index=False)
    result = {
        "reference_rows": transductive.reference_rows,
        "engineered_features": transductive.engineered_features,
        "transformed_features_fold_1": transductive.transformed_features_fold_1,
        "transductive_synthesis_accuracy": float(
            summary.loc["transductive_archive_synthesis", "mean_accuracy"]
        ),
        "change_vs_promoted_identity_vote": float(
            summary.loc["transductive_archive_synthesis", "leader_change"]
        ),
        "change_vs_inductive_synthesis": float(
            summary.loc["transductive_archive_synthesis", "mean_accuracy"]
            - summary.loc["inductive_archive_synthesis", "mean_accuracy"]
        ),
        "passes_gate": bool(
            summary.loc["transductive_archive_synthesis", "passes_gate"]
        ),
        "rare_threshold_crossing_share_min": float(rare_crossing["share"].min()),
        "rare_threshold_crossing_share_max": float(rare_crossing["share"].max()),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)


def _summarise_rare_threshold_crossing(partitioned, reference_X):
    rows = []
    reference_counts = {
        source: normalise_identity(reference_X[source]).value_counts()
        for source in FREQUENCY_SOURCE_FEATURES
    }
    for fold_number, (training_positions, validation_positions) in enumerate(
        make_cross_validation(partitioned).split(),
        start=1,
    ):
        any_crossing = np.zeros(len(validation_positions), dtype=bool)
        field_events = 0
        training = partitioned.X_development.iloc[training_positions]
        validation = partitioned.X_development.iloc[validation_positions]
        for source in FREQUENCY_SOURCE_FEATURES:
            training_counts = normalise_identity(training[source]).value_counts()
            values = normalise_identity(validation[source])
            crossing = (
                values.map(training_counts)
                .fillna(0)
                .lt(FREQUENCY_MINIMUM_SUPPORT)
                & values.map(reference_counts[source])
                .fillna(0)
                .ge(FREQUENCY_MINIMUM_SUPPORT)
            )
            any_crossing |= crossing.to_numpy()
            field_events += int(crossing.sum())
        rows.append(
            {
                "validation_fold": fold_number,
                "rows_crossing_any_rare_boundary": int(any_crossing.sum()),
                "share": float(any_crossing.mean()),
                "field_events": field_events,
            }
        )
    return pd.DataFrame(rows)


def _vote(partitioned, evaluations, weights, name):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        *evaluations,
        weights=weights,
        model_name=name,
    )


def _validate_evidence(partitioned, *evaluations) -> None:
    for evaluation in evaluations:
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Transductive-frequency evidence uses different folds.")


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
                "passes_gate": key == "transductive_archive_synthesis" and (
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

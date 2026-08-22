"""Run the fixed all-categorical occurrence-count forest screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "categorical-frequency-forest-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
IDENTITY_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "catboost-identity-screen"
    / "complete-deferred-identities.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from categorical_frequency_forest_evaluation import evaluate_categorical_frequency_forest
from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import compare_candidate_diversity
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
    identity = joblib.load(IDENTITY_PATH).evaluation
    _validate_evidence(partitioned, baseline, identity)

    cache_path = OUTPUT_DIR / "all-categorical-occurrence-counts.joblib"
    if cache_path.exists():
        frequency = joblib.load(cache_path)
        if (
            frequency.random_forest.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Cached frequency forest uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        frequency = evaluate_categorical_frequency_forest(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(frequency, cache_path)

    leader = _vote(
        partitioned,
        (baseline.xgboost, baseline.random_forest, identity),
        (0.44, 0.36, 0.20),
        "Promoted complete-identity vote",
    )
    replacement_vote = _vote(
        partitioned,
        (baseline.xgboost, frequency.random_forest, identity),
        (0.44, 0.36, 0.20),
        "Frequency-forest replacement vote",
    )
    forest_bag_vote = _vote(
        partitioned,
        (
            baseline.xgboost,
            baseline.random_forest,
            frequency.random_forest,
            identity,
        ),
        (0.44, 0.18, 0.18, 0.20),
        "Equal Random Forest representation bag vote",
    )
    candidates = {
        "promoted_identity_vote": leader,
        "accepted_random_forest": baseline.random_forest,
        "frequency_random_forest": frequency.random_forest,
        "frequency_replacement_vote": replacement_vote,
        "forest_representation_bag_vote": forest_bag_vote,
    }
    summary = _summarise(candidates, leader)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    diversity = compare_candidate_diversity(
        partitioned,
        baseline.random_forest,
        frequency.random_forest,
        leader,
    )
    diversity.to_csv(OUTPUT_DIR / "diversity.csv")
    result = {
        "engineered_features": frequency.engineered_features,
        "transformed_features_fold_1": frequency.transformed_features_fold_1,
        "frequency_forest_accuracy": float(
            summary.loc["frequency_random_forest", "mean_accuracy"]
        ),
        "forest_bag_accuracy": float(
            summary.loc["forest_representation_bag_vote", "mean_accuracy"]
        ),
        "change_vs_promoted_identity_vote": float(
            summary.loc[
                "forest_representation_bag_vote",
                "leader_change",
            ]
        ),
        "passes_gate": bool(
            summary.loc[
                "forest_representation_bag_vote",
                "passes_gate",
            ]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
        "external_evidence": "https://github.com/drivendataorg/pump-it-up/tree/master/madRid",
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)
    print("\nDiversity:\n", diversity.to_string(), flush=True)


def _vote(partitioned, evaluations, weights, name):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        *evaluations,
        weights=weights,
        model_name=name,
    )


def _validate_evidence(partitioned, baseline, identity) -> None:
    for evaluation in (baseline.xgboost, baseline.random_forest, identity):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Categorical-frequency comparison uses different folds.")


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
                "passes_gate": key == "forest_representation_bag_vote" and (
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

"""Run the shared region-by-physical interaction-layer screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "regional-interaction-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from regional_interaction_evaluation import evaluate_regional_interaction_trial


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
    _validate_evidence(
        partitioned,
        baseline.xgboost,
        baseline.random_forest,
        identity,
    )

    cache_path = OUTPUT_DIR / "three-region-physical-composites.joblib"
    if cache_path.exists():
        trial = joblib.load(cache_path)
        for evaluation in (trial.xgboost, trial.random_forest, trial.catboost):
            if (
                evaluation.cross_validation_fingerprint
                != partitioned.cross_validation_fingerprint
            ):
                raise ValueError("Cached regional interaction trial uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        trial = evaluate_regional_interaction_trial(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(trial, cache_path)

    leader = _vote(
        partitioned,
        (baseline.xgboost, baseline.random_forest, identity),
        (0.44, 0.36, 0.20),
        "Promoted complete-identity vote",
    )
    interaction_two_component = _vote(
        partitioned,
        (trial.xgboost, trial.random_forest),
        (0.55, 0.45),
        "Regional-interaction 55:45 vote",
    )
    interaction_global_vote = _vote(
        partitioned,
        (trial.xgboost, trial.random_forest, identity),
        (0.44, 0.36, 0.20),
        "Regional interactions in global components",
    )
    interaction_catboost_vote = _vote(
        partitioned,
        (baseline.xgboost, baseline.random_forest, trial.catboost),
        (0.44, 0.36, 0.20),
        "Regional interactions in identity CatBoost",
    )
    interaction_full_vote = _vote(
        partitioned,
        (trial.xgboost, trial.random_forest, trial.catboost),
        (0.44, 0.36, 0.20),
        "Regional-interaction full vote",
    )
    interaction_representation_bag = _vote(
        partitioned,
        (
            baseline.xgboost,
            trial.xgboost,
            baseline.random_forest,
            trial.random_forest,
            identity,
            trial.catboost,
        ),
        (0.22, 0.22, 0.18, 0.18, 0.10, 0.10),
        "Equal original/regional-interaction representation bag",
    )
    candidates = {
        "promoted_identity_vote": leader,
        "regional_interaction_xgboost": trial.xgboost,
        "regional_interaction_random_forest": trial.random_forest,
        "regional_interaction_catboost": trial.catboost,
        "regional_interaction_two_component_vote": interaction_two_component,
        "regional_interaction_global_vote": interaction_global_vote,
        "regional_interaction_catboost_vote": interaction_catboost_vote,
        "regional_interaction_full_vote": interaction_full_vote,
        "regional_interaction_representation_bag": (
            interaction_representation_bag
        ),
    }
    summary = _summarise(candidates, leader)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    support = _support_audit(partitioned)
    support.to_csv(OUTPUT_DIR / "interaction-support-audit.csv", index=False)
    result = {
        "engineered_features": trial.engineered_features,
        "transformed_features_fold_1": trial.transformed_features_fold_1,
        "categorical_features": trial.categorical_features,
        "full_vote_accuracy": float(
            summary.loc["regional_interaction_full_vote", "mean_accuracy"]
        ),
        "representation_bag_accuracy": float(
            summary.loc[
                "regional_interaction_representation_bag",
                "mean_accuracy",
            ]
        ),
        "best_change_vs_promoted_identity_vote": float(
            summary.loc[
                [
                    "regional_interaction_global_vote",
                    "regional_interaction_catboost_vote",
                    "regional_interaction_full_vote",
                    "regional_interaction_representation_bag",
                ],
                "leader_change",
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


def _vote(partitioned, evaluations, weights, name):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        *evaluations,
        weights=weights,
        model_name=name,
    )


def _support_audit(partitioned) -> pd.DataFrame:
    rows = []
    region = partitioned.X_development["region"].astype("string").str.strip().str.lower()
    for source in ("extraction_type", "source", "waterpoint_type"):
        physical = (
            partitioned.X_development[source]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        composite = region.str.cat(physical, sep="::")
        counts = composite.value_counts(dropna=False)
        rows.append(
            {
                "source": source,
                "levels": int(composite.nunique(dropna=False)),
                "singleton_row_share": float(composite.map(counts).eq(1).mean()),
                "rows_in_groups_at_least_20": float(
                    composite.map(counts).ge(20).mean()
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
            raise ValueError("Regional interaction evidence uses different folds.")


def _summarise(candidates, leader):
    leader_accuracy = float(leader.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        leader.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    eligible = {
        "regional_interaction_global_vote",
        "regional_interaction_catboost_vote",
        "regional_interaction_full_vote",
        "regional_interaction_representation_bag",
    }
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
                "passes_gate": key in eligible
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

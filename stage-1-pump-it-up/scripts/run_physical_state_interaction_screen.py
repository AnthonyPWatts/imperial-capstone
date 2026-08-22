"""Run the fixed quantity-by-physical CatBoost interaction screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "physical-state-interaction-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from physical_state_interaction_evaluation import (
    PHYSICAL_STATE_INTERACTION_SOURCES,
)
from physical_state_interaction_evaluation import (
    evaluate_physical_state_interaction_trial,
)


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

    cache_path = OUTPUT_DIR / "three-quantity-physical-composites.joblib"
    if cache_path.exists():
        trial = joblib.load(cache_path)
        _validate_evidence(partitioned, trial.catboost)
        print(f"Loaded {cache_path}.", flush=True)
    else:
        trial = evaluate_physical_state_interaction_trial(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(trial, cache_path)

    promoted = _vote(
        partitioned,
        (baseline.xgboost, baseline.random_forest, identity),
        (0.44, 0.36, 0.20),
        "Promoted complete-identity vote",
    )
    direct = _vote(
        partitioned,
        (baseline.xgboost, baseline.random_forest, trial.catboost),
        (0.44, 0.36, 0.20),
        "Physical-state interaction CatBoost vote",
    )
    direct_bag = _vote(
        partitioned,
        (baseline.xgboost, baseline.random_forest, identity, trial.catboost),
        (0.44, 0.36, 0.10, 0.10),
        "Equal identity/physical-state CatBoost bag",
    )
    archive = _vote(
        partitioned,
        (
            baseline.xgboost,
            spatial.xgboost,
            baseline.random_forest,
            frequency,
            spatial.random_forest,
            identity,
        ),
        (0.33, 0.11, 0.18, 0.09, 0.09, 0.20),
        "Existing archive synthesis",
    )
    archive_replacement = _vote(
        partitioned,
        (
            baseline.xgboost,
            spatial.xgboost,
            baseline.random_forest,
            frequency,
            spatial.random_forest,
            trial.catboost,
        ),
        (0.33, 0.11, 0.18, 0.09, 0.09, 0.20),
        "Physical-state CatBoost archive substitution",
    )
    archive_bag = _vote(
        partitioned,
        (
            baseline.xgboost,
            spatial.xgboost,
            baseline.random_forest,
            frequency,
            spatial.random_forest,
            identity,
            trial.catboost,
        ),
        (0.33, 0.11, 0.18, 0.09, 0.09, 0.10, 0.10),
        "Equal CatBoost-subfamily archive bag",
    )
    candidates = {
        "promoted_identity_vote": promoted,
        "interaction_catboost": trial.catboost,
        "interaction_direct_vote": direct,
        "interaction_direct_bag": direct_bag,
        "existing_archive_synthesis": archive,
        "interaction_archive_substitution": archive_replacement,
        "interaction_archive_bag": archive_bag,
    }
    summary = _summarise(candidates, archive)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    support = _support_audit(partitioned)
    support.to_csv(OUTPUT_DIR / "interaction-support-audit.csv", index=False)
    result = {
        "engineered_features": trial.engineered_features,
        "categorical_features": trial.categorical_features,
        "archive_substitution_accuracy": float(
            summary.loc["interaction_archive_substitution", "mean_accuracy"]
        ),
        "archive_bag_accuracy": float(
            summary.loc["interaction_archive_bag", "mean_accuracy"]
        ),
        "best_change_vs_archive": float(
            summary.loc[
                ["interaction_archive_substitution", "interaction_archive_bag"],
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
    print("\nSupport:\n", support.to_string(index=False), flush=True)


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
    quantity = (
        partitioned.X_development["quantity"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    for source in PHYSICAL_STATE_INTERACTION_SOURCES:
        physical = (
            partitioned.X_development[source]
            .astype("string")
            .str.strip()
            .str.lower()
        )
        composite = quantity.str.cat(physical, sep="::")
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
            raise ValueError("Physical-state interaction evidence uses different folds.")


def _summarise(candidates, incumbent):
    incumbent_accuracy = float(incumbent.metric_summary.loc["accuracy", "mean"])
    incumbent_repair = float(
        incumbent.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    eligible = {"interaction_archive_substitution", "interaction_archive_bag"}
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

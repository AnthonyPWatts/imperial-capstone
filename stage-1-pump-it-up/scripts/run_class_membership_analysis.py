"""Analyse and export accepted-model three-class membership probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "class-membership-analysis"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
COMPONENT_CACHE_PATH = OUTPUT_DIR / "accepted-competition-probabilities.joblib"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import partition_modelling_data
from final_model import ordered_probabilities, validate_probabilities
from geography_evaluation import RANDOM_FOREST_WEIGHT, XGBOOST_WEIGHT
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import median_selected_iterations
from model_evaluation import make_random_forest_pipeline
from modelling_data import prepare_modelling_data
from probability_diagnostics import class_membership_summary
from probability_diagnostics import class_membership_table
from probability_diagnostics import classwise_probability_scores
from probability_diagnostics import component_disagreement_summary
from probability_diagnostics import membership_ambiguity_summary
from probability_diagnostics import probability_quality_summary
from probability_diagnostics import top_label_reliability
from probability_diagnostics import top_two_membership_summary
from probability_diagnostics import write_class_membership_table


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = _load_modelling_data()
    partitioned_data = partition_modelling_data(modelling_data)
    baseline_trial = joblib.load(BASELINE_PATH)
    accepted = baseline_trial.blend
    if (
        accepted.cross_validation_fingerprint
        != partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError("Accepted membership baseline uses different folds.")

    development_probabilities = accepted.out_of_fold_probabilities.to_numpy()
    development_table = class_membership_table(
        partitioned_data.development_ids,
        development_probabilities,
        actual=partitioned_data.y_development,
    )
    development_summary = class_membership_summary(development_table)
    development_ambiguity = membership_ambiguity_summary(development_table)
    development_boundaries = top_two_membership_summary(development_table)
    probability_quality = pd.Series(
        probability_quality_summary(
            partitioned_data.y_development,
            development_probabilities,
        ),
        name="value",
    )
    classwise_quality = classwise_probability_scores(
        partitioned_data.y_development,
        development_probabilities,
    )
    reliability = top_label_reliability(
        partitioned_data.y_development,
        development_probabilities,
    )
    disagreement = pd.Series(
        component_disagreement_summary(
            {
                "XGBoost": baseline_trial.xgboost.out_of_fold_probabilities,
                "Random Forest": (
                    baseline_trial.random_forest.out_of_fold_probabilities
                ),
            }
        ),
        name="value",
    )

    component_payload = _load_or_fit_competition_probabilities(
        modelling_data,
        baseline_trial,
        force=args.force,
    )
    competition_table = class_membership_table(
        modelling_data.competition_ids,
        component_payload["blend_probabilities"],
    )
    competition_summary = class_membership_summary(competition_table)
    competition_ambiguity = membership_ambiguity_summary(competition_table)
    competition_boundaries = top_two_membership_summary(competition_table)

    development_path = write_class_membership_table(
        development_table,
        OUTPUT_DIR / "accepted-development-memberships.csv",
    )
    competition_path = write_class_membership_table(
        competition_table,
        OUTPUT_DIR / "accepted-competition-memberships.csv",
    )
    development_summary.to_csv(OUTPUT_DIR / "development-class-summary.csv")
    competition_summary.to_csv(OUTPUT_DIR / "competition-class-summary.csv")
    development_ambiguity.to_frame().to_csv(
        OUTPUT_DIR / "development-ambiguity-summary.csv"
    )
    competition_ambiguity.to_frame().to_csv(
        OUTPUT_DIR / "competition-ambiguity-summary.csv"
    )
    development_boundaries.to_csv(
        OUTPUT_DIR / "development-top-two-boundaries.csv"
    )
    competition_boundaries.to_csv(
        OUTPUT_DIR / "competition-top-two-boundaries.csv"
    )
    probability_quality.to_frame().to_csv(
        OUTPUT_DIR / "development-probability-quality.csv"
    )
    classwise_quality.to_csv(OUTPUT_DIR / "development-classwise-quality.csv")
    reliability.to_csv(OUTPUT_DIR / "development-top-label-reliability.csv")
    disagreement.to_frame().to_csv(
        OUTPUT_DIR / "development-component-disagreement.csv"
    )

    result = {
        "accepted_evaluation": accepted,
        "development_table_path": development_path,
        "competition_table_path": competition_path,
        "development_summary": development_summary,
        "competition_summary": competition_summary,
        "development_ambiguity": development_ambiguity,
        "competition_ambiguity": competition_ambiguity,
        "development_boundaries": development_boundaries,
        "competition_boundaries": competition_boundaries,
        "probability_quality": probability_quality,
        "classwise_quality": classwise_quality,
        "reliability": reliability,
        "component_disagreement": disagreement,
        "xgboost_iterations": component_payload["xgboost_iterations"],
        "component_seconds": component_payload["component_seconds"],
    }
    joblib.dump(result, OUTPUT_DIR / "class-membership-analysis.joblib")
    payload = {
        "development_rows": len(development_table),
        "competition_rows": len(competition_table),
        "xgboost_iterations": int(component_payload["xgboost_iterations"]),
        "development_accuracy": float(probability_quality["accuracy"]),
        "development_log_loss": float(probability_quality["log_loss"]),
        "development_top_label_ece": float(
            probability_quality["top_label_ece"]
        ),
        "development_close_membership_share": float(
            development_ambiguity["close_membership_share"]
        ),
        "competition_close_membership_share": float(
            competition_ambiguity["close_membership_share"]
        ),
        "development_no_majority_share": float(
            development_ambiguity["no_majority_share"]
        ),
        "competition_no_majority_share": float(
            competition_ambiguity["no_majority_share"]
        ),
        "competition_membership_csv": str(competition_path),
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nDevelopment probability quality", flush=True)
    print(probability_quality.to_string(), flush=True)
    print("\nDevelopment class memberships", flush=True)
    print(_format_class_summary(development_summary), flush=True)
    print("\nCompetition class memberships", flush=True)
    print(_format_class_summary(competition_summary), flush=True)
    print("\nAmbiguity comparison", flush=True)
    print(
        pd.concat(
            {
                "development": development_ambiguity,
                "competition": competition_ambiguity,
            },
            axis=1,
        ).to_string(),
        flush=True,
    )
    print(f"\nMembership CSV: {competition_path}", flush=True)


def _load_or_fit_competition_probabilities(
    modelling_data,
    baseline_trial,
    *,
    force: bool,
) -> dict:
    if COMPONENT_CACHE_PATH.exists() and not force:
        payload = joblib.load(COMPONENT_CACHE_PATH)
        if not payload["competition_ids"].reset_index(drop=True).equals(
            modelling_data.competition_ids.reset_index(drop=True)
        ):
            raise ValueError("Cached competition memberships use different IDs.")
        for key in (
            "xgboost_probabilities",
            "random_forest_probabilities",
            "blend_probabilities",
        ):
            validate_probabilities(payload[key], len(modelling_data.competition_ids))
        print("Loaded accepted competition probability components.", flush=True)
        return payload

    xgboost_iterations = median_selected_iterations(baseline_trial.xgboost)
    xgboost_spec = make_xgboost_spec(variant="depth 8 child 1")
    xgboost_probabilities, xgboost_seconds = fit_gpu_candidate_probabilities(
        xgboost_spec,
        modelling_data.X_original,
        modelling_data.y_original,
        modelling_data.X_competition,
        iterations=xgboost_iterations,
    )
    forest = make_random_forest_pipeline()
    forest_started = time.perf_counter()
    forest.fit(modelling_data.X_original, modelling_data.y_original)
    random_forest_probabilities = ordered_probabilities(
        forest,
        modelling_data.X_competition,
    )
    forest_seconds = time.perf_counter() - forest_started
    blend_probabilities = (
        XGBOOST_WEIGHT * xgboost_probabilities
        + RANDOM_FOREST_WEIGHT * random_forest_probabilities
    )
    for values in (
        xgboost_probabilities,
        random_forest_probabilities,
        blend_probabilities,
    ):
        validate_probabilities(values, len(modelling_data.competition_ids))
    payload = {
        "competition_ids": modelling_data.competition_ids.copy(),
        "xgboost_probabilities": xgboost_probabilities,
        "random_forest_probabilities": random_forest_probabilities,
        "blend_probabilities": blend_probabilities,
        "xgboost_iterations": xgboost_iterations,
        "component_seconds": pd.Series(
            {
                "XGBoost": xgboost_seconds,
                "Random Forest": forest_seconds,
            },
            name="fit and predict seconds",
        ),
    }
    joblib.dump(payload, COMPONENT_CACHE_PATH)
    return payload


def _load_modelling_data():
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _format_class_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in (
        "mean_membership",
        "predicted_share",
        "membership_at_least_25_share",
        "membership_at_least_50_share",
        "actual_share",
        "mean_membership_when_actual",
    ):
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.3%}")
    return display.to_string()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit accepted components instead of reusing probability caches.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

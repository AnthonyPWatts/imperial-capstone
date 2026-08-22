"""Refit and write the fixed archive-derived synthesis candidate."""

from __future__ import annotations

import hashlib
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
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "archive-synthesis-competition"
COMPONENT_CACHE_PATH = RUNTIME_DIR / "archive-components.joblib"
OUTPUT_DIR = STAGE_DIR / "submissions" / "2026-08-22-archive-synthesis"
CANDIDATE_NAME = (
    "01-33-xgboost-11-spatial-xgboost-18-random-forest-"
    "09-frequency-forest-09-spatial-forest-20-identity-catboost.csv"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archive_synthesis_confirmation import ARCHIVE_SYNTHESIS_WEIGHTS
from archive_synthesis_confirmation import blend_archive_synthesis
from categorical_frequency_forest_evaluation import (
    make_categorical_frequency_preprocessor,
)
from final_model import CLASS_LABELS
from final_model import build_competition_prediction
from final_model import ordered_probabilities
from final_model import validate_probabilities
from final_model import write_validated_submission
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import median_selected_iterations
from model_evaluation import make_random_forest_pipeline
from modelling_data import prepare_modelling_data
from spatial_height_imputation_evaluation import make_spatial_height_preprocessor


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    submission_template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")
    accepted = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "class-membership-analysis"
        / "accepted-competition-probabilities.joblib"
    )
    identity = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "catboost-identity-competition"
        / "complete-identity-catboost.joblib"
    )
    spatial_trial = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "spatial-height-imputation-screen"
        / "ten-neighbour-height.joblib"
    )
    frequency_trial = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "categorical-frequency-forest-screen"
        / "all-categorical-occurrence-counts.joblib"
    )
    _validate_component_cache(accepted, modelling_data, "accepted")
    _validate_component_cache(identity, modelling_data, "identity")
    archive_components = _load_or_fit_archive_components(
        modelling_data,
        spatial_trial,
        frequency_trial,
    )
    components = {
        "xgboost": accepted["xgboost_probabilities"],
        "spatial_xgboost": archive_components["spatial_xgboost"],
        "random_forest": accepted["random_forest_probabilities"],
        "frequency_random_forest": archive_components[
            "frequency_random_forest"
        ],
        "spatial_random_forest": archive_components["spatial_random_forest"],
        "identity_catboost": identity["probabilities"],
    }
    probabilities = blend_archive_synthesis(components)
    accepted_seconds = accepted["component_seconds"]
    new_seconds = archive_components["component_seconds"]
    seconds = pd.Series(
        {
            "XGBoost": float(accepted_seconds.loc["XGBoost"]),
            "spatial-height XGBoost": float(
                new_seconds.loc["spatial-height XGBoost"]
            ),
            "Random Forest": float(accepted_seconds.loc["Random Forest"]),
            "frequency Random Forest": float(
                new_seconds.loc["frequency Random Forest"]
            ),
            "spatial-height Random Forest": float(
                new_seconds.loc["spatial-height Random Forest"]
            ),
            "complete-identity CatBoost": float(
                identity["fit_and_predict_seconds"]
            ),
        },
        name="fit and predict seconds",
    )
    prediction = build_competition_prediction(
        modelling_data,
        submission_template,
        probabilities,
        seconds,
    )
    destination = OUTPUT_DIR / CANDIDATE_NAME
    write_validated_submission(prediction, destination)
    reloaded = pd.read_csv(destination)
    _validate_written_submission(
        reloaded,
        submission_template,
        expected_rows=len(modelling_data.X_competition),
    )
    ready_identity_path = (
        STAGE_DIR
        / "submissions"
        / "2026-08-22-catboost-identities"
        / (
            "01-44-xgboost-depth-8-child-1-36-random-forest-"
            "20-catboost-complete-identities.csv"
        )
    )
    ready_identity = pd.read_csv(ready_identity_path)
    manifest = {
        "date": "2026-08-22",
        "status": "prepared_not_uploaded",
        "selection": {
            "development_accuracy": 0.8179292929292929,
            "development_change_vs_ready_identity": 0.0005260942760942,
            "fold_wins_vs_ready_identity": 5,
            "local_test_accuracy": 0.8118686868686869,
            "local_test_change_vs_ready_identity": 0.0010101010101010,
            "local_test_net_additional_correct": 12,
        },
        "weights": ARCHIVE_SYNTHESIS_WEIGHTS,
        "iterations": {
            "xgboost": int(accepted["xgboost_iterations"]),
            "spatial_height_xgboost": int(
                archive_components["spatial_xgboost_iterations"]
            ),
            "complete_identity_catboost": int(identity["iterations"]),
        },
        "competition_prediction_disagreement_vs_ready_identity": float(
            reloaded["status_group"].ne(ready_identity["status_group"]).mean()
        ),
        "competition_class_shares": {
            label: float(prediction.class_shares.loc[label])
            for label in CLASS_LABELS
        },
        "fit_and_predict_seconds": {
            key: float(value) for key, value in seconds.items()
        },
        "csv": destination.name,
        "rows": len(reloaded),
        "unique_ids": int(reloaded["id"].nunique()),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "daily_submission_allowance": "3/3 already used; do not upload",
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def _load_or_fit_archive_components(modelling_data, spatial_trial, frequency_trial):
    spatial_iterations = median_selected_iterations(spatial_trial.xgboost)
    if COMPONENT_CACHE_PATH.exists():
        payload = joblib.load(COMPONENT_CACHE_PATH)
        _validate_component_cache(payload, modelling_data, "archive")
        if payload.get("spatial_xgboost_iterations") != spatial_iterations:
            raise ValueError("Cached spatial XGBoost tree count changed.")
        for key in (
            "spatial_xgboost",
            "spatial_random_forest",
            "frequency_random_forest",
        ):
            validate_probabilities(
                payload[key],
                len(modelling_data.X_competition),
            )
        print(f"Loaded {COMPONENT_CACHE_PATH}.", flush=True)
        return payload

    spatial_xgboost, xgboost_seconds = fit_gpu_candidate_probabilities(
        make_xgboost_spec(variant="depth 8 child 1"),
        modelling_data.X_original,
        modelling_data.y_original,
        modelling_data.X_competition,
        iterations=spatial_iterations,
        preprocessor_factory=make_spatial_height_preprocessor,
    )
    spatial_forest, spatial_forest_seconds = _fit_forest(
        modelling_data,
        make_spatial_height_preprocessor,
    )
    frequency_forest, frequency_forest_seconds = _fit_forest(
        modelling_data,
        make_categorical_frequency_preprocessor,
    )
    payload = {
        "competition_ids": modelling_data.competition_ids.copy(),
        "spatial_xgboost": spatial_xgboost,
        "spatial_random_forest": spatial_forest,
        "frequency_random_forest": frequency_forest,
        "spatial_xgboost_iterations": spatial_iterations,
        "component_seconds": pd.Series(
            {
                "spatial-height XGBoost": xgboost_seconds,
                "spatial-height Random Forest": spatial_forest_seconds,
                "frequency Random Forest": frequency_forest_seconds,
            },
            name="fit and predict seconds",
        ),
    }
    joblib.dump(payload, COMPONENT_CACHE_PATH)
    return payload


def _fit_forest(modelling_data, preprocessor_factory):
    forest = make_random_forest_pipeline(
        preprocessor_factory=preprocessor_factory,
    )
    started = time.perf_counter()
    forest.fit(modelling_data.X_original, modelling_data.y_original)
    probabilities = ordered_probabilities(forest, modelling_data.X_competition)
    elapsed = time.perf_counter() - started
    validate_probabilities(probabilities, len(modelling_data.X_competition))
    return probabilities, elapsed


def _validate_component_cache(payload, modelling_data, name: str) -> None:
    if not payload["competition_ids"].reset_index(drop=True).equals(
        modelling_data.competition_ids.reset_index(drop=True)
    ):
        raise ValueError(f"Cached {name} component IDs changed.")


def _validate_written_submission(submission, template, *, expected_rows: int) -> None:
    if list(submission.columns) != ["id", "status_group"]:
        raise ValueError("Archive synthesis has invalid columns.")
    if len(submission) != expected_rows:
        raise ValueError("Archive synthesis has the wrong row count.")
    if submission["id"].duplicated().any() or submission.isna().any().any():
        raise ValueError("Archive synthesis contains missing or duplicate rows.")
    if not submission["id"].equals(template["id"]):
        raise ValueError("Archive synthesis changed template ID order.")
    if not set(submission["status_group"]).issubset(CLASS_LABELS):
        raise ValueError("Archive synthesis contains an invalid label.")


if __name__ == "__main__":
    main()

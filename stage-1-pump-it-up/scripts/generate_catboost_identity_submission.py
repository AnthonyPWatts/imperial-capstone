"""Refit and write the frozen complete-identity CatBoost candidate."""

from __future__ import annotations

import hashlib
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
IDENTITY_TRIAL_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "catboost-identity-screen"
    / "complete-deferred-identities.joblib"
)
ACCEPTED_COMPONENT_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "class-membership-analysis"
    / "accepted-competition-probabilities.joblib"
)
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "catboost-identity-competition"
CATBOOST_CACHE_PATH = RUNTIME_DIR / "complete-identity-catboost.joblib"
OUTPUT_DIR = STAGE_DIR / "submissions" / "2026-08-22-catboost-identities"
ACCEPTED_SUBMISSION_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-08-21-model-family-screen"
    / "01-55-xgboost-depth-8-child-1-current-one-hot-45-random-forest.csv"
)
CANDIDATE_NAME = (
    "01-44-xgboost-depth-8-child-1-36-random-forest-"
    "20-catboost-complete-identities.csv"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_confirmation import (
    build_frozen_identity_competition_predictions,
)
from catboost_identity_evaluation import (
    engineer_complete_identity_catboost_features,
)
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from final_model import write_validated_submission
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_catboost_spec
from gpu_model_evaluation import median_selected_iterations
from modelling_data import prepare_modelling_data


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    submission_template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")
    accepted = joblib.load(ACCEPTED_COMPONENT_PATH)
    _validate_accepted_cache(accepted, modelling_data)
    identity_trial = joblib.load(IDENTITY_TRIAL_PATH)
    catboost_iterations = median_selected_iterations(identity_trial.evaluation)
    catboost_probabilities, catboost_seconds = _load_or_fit_catboost(
        modelling_data,
        catboost_iterations,
    )
    components = {
        "xgboost": accepted["xgboost_probabilities"],
        "random_forest": accepted["random_forest_probabilities"],
        "identity_catboost": catboost_probabilities,
    }
    accepted_seconds = accepted["component_seconds"]
    seconds = pd.Series(
        {
            "XGBoost": float(accepted_seconds.loc["XGBoost"]),
            "Random Forest": float(accepted_seconds.loc["Random Forest"]),
            "complete-identity CatBoost": catboost_seconds,
        },
        name="fit and predict seconds",
    )
    predictions = build_frozen_identity_competition_predictions(
        modelling_data,
        submission_template,
        components,
        seconds,
    )
    _verify_accepted_reconstruction(
        predictions["accepted_55_45"].submission,
    )
    candidate = predictions["identity_candidate_44_36_20"]
    destination = OUTPUT_DIR / CANDIDATE_NAME
    write_validated_submission(candidate, destination)
    reloaded = pd.read_csv(destination)
    accepted_labels = predictions["accepted_55_45"].submission["status_group"]
    candidate_labels = reloaded["status_group"]
    transition = pd.crosstab(
        accepted_labels,
        candidate_labels,
        rownames=["accepted prediction"],
        colnames=["candidate prediction"],
        dropna=False,
    ).reindex(index=CLASS_LABELS, columns=CLASS_LABELS, fill_value=0)
    transition.to_csv(OUTPUT_DIR / "accepted-to-candidate-transitions.csv")
    record = {
        "date": "2026-08-22",
        "status": "prepared_not_uploaded",
        "selection": {
            "development_accuracy": 0.8174031986531987,
            "development_change": 0.0011574074074073293,
            "fold_wins": 4,
            "local_test_accuracy": 0.8108585858585858,
            "local_test_change": 0.003872053872053916,
        },
        "weights": {
            "xgboost": 0.44,
            "random_forest": 0.36,
            "complete_identity_catboost": 0.20,
        },
        "iterations": {
            "xgboost": int(accepted["xgboost_iterations"]),
            "complete_identity_catboost": catboost_iterations,
        },
        "accepted_reconstruction_exact": True,
        "competition_prediction_disagreement": float(
            accepted_labels.ne(candidate_labels).mean()
        ),
        "competition_class_shares": {
            label: float(candidate.class_shares.loc[label])
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
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2), flush=True)


def _load_or_fit_catboost(modelling_data, iterations: int):
    if CATBOOST_CACHE_PATH.exists():
        payload = joblib.load(CATBOOST_CACHE_PATH)
        if payload.get("iterations") != iterations:
            raise ValueError("Cached full-data CatBoost tree count changed.")
        if not payload["competition_ids"].reset_index(drop=True).equals(
            modelling_data.competition_ids.reset_index(drop=True)
        ):
            raise ValueError("Cached full-data CatBoost competition IDs changed.")
        probabilities = payload["probabilities"]
        validate_probabilities(probabilities, len(modelling_data.X_competition))
        print(f"Loaded {CATBOOST_CACHE_PATH}.", flush=True)
        return probabilities, float(payload["fit_and_predict_seconds"])

    probabilities, elapsed = fit_gpu_candidate_probabilities(
        make_catboost_spec(variant="d8"),
        modelling_data.X_original,
        modelling_data.y_original,
        modelling_data.X_competition,
        iterations=iterations,
        catboost_feature_engineer=engineer_complete_identity_catboost_features,
    )
    validate_probabilities(probabilities, len(modelling_data.X_competition))
    joblib.dump(
        {
            "competition_ids": modelling_data.competition_ids.copy(),
            "iterations": iterations,
            "probabilities": probabilities,
            "fit_and_predict_seconds": elapsed,
        },
        CATBOOST_CACHE_PATH,
    )
    return probabilities, elapsed


def _validate_accepted_cache(payload, modelling_data) -> None:
    if not payload["competition_ids"].reset_index(drop=True).equals(
        modelling_data.competition_ids.reset_index(drop=True)
    ):
        raise ValueError("Accepted full-data component IDs changed.")
    for key in ("xgboost_probabilities", "random_forest_probabilities"):
        validate_probabilities(payload[key], len(modelling_data.X_competition))


def _verify_accepted_reconstruction(submission: pd.DataFrame) -> None:
    existing = pd.read_csv(ACCEPTED_SUBMISSION_PATH)
    if not existing.equals(submission.reset_index(drop=True)):
        raise ValueError(
            "Accepted cached components do not recreate the existing submission."
        )


if __name__ == "__main__":
    main()

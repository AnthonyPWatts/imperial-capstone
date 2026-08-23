"""Refit and write an evidence-backed deep-XGBoost archive candidate."""

from __future__ import annotations

import argparse
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
DEFAULT_RUNTIME_DIR = PROJECT_DIR / ".runtime" / "deep-archive-competition"
DEFAULT_OUTPUT_DIR = STAGE_DIR / "submissions" / "2026-08-23-deep-archive"
DEFAULT_CANDIDATE_NAME = (
    "01-33-deep-top50-xgboost-11-spatial-xgboost-18-random-forest-"
    "09-frequency-forest-09-spatial-forest-20-identity-catboost.csv"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from deep_archive_confirmation import DEEP_XGBOOST_SEED
from deep_archive_confirmation import blend_deep_archive
from final_model import CLASS_LABELS
from final_model import build_competition_prediction
from final_model import validate_probabilities
from final_model import write_validated_submission
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from modelling_data import prepare_modelling_data
from top_common_identity_evaluation import ARCHIVED_DEPTH_17_ITERATIONS
from top_common_identity_evaluation import make_top_common_identity_preprocessor


SELECTION_BY_SEED = {
    DEEP_XGBOOST_SEED: {
        "development_accuracy": 0.8187710437710438,
        "development_change_vs_archive": 0.0008417508417508657,
        "development_change_vs_seed_20260822": 0.0,
        "development_net_additional_correct_vs_seed_20260822": 0,
        "fold_wins_vs_archive": 3,
        "worst_fold_change_vs_archive": -0.00021043771043771642,
        "local_test_accuracy": 0.8135521885521886,
        "local_test_change_vs_archive": 0.001683501683501731,
        "local_test_change_vs_seed_20260822": 0.0,
        "local_test_net_additional_correct_vs_archive": 20,
        "local_test_net_additional_correct_vs_seed_20260822": 0,
    },
    20260824: {
        "development_accuracy": 0.8189814814814815,
        "development_change_vs_archive": 0.001052188552188582,
        "development_change_vs_seed_20260822": 0.00021043771043771642,
        "development_net_additional_correct_vs_seed_20260822": 10,
        "fold_wins_vs_archive": 3,
        "worst_fold_change_vs_archive": -0.00042087542087543284,
        "local_test_accuracy": 0.813973063973064,
        "local_test_change_vs_archive": 0.0021043771043771305,
        "local_test_change_vs_seed_20260822": 0.00042087542087543284,
        "local_test_net_additional_correct_vs_archive": 25,
        "local_test_net_additional_correct_vs_seed_20260822": 5,
    },
}


def main(*, seed: int = DEEP_XGBOOST_SEED) -> None:
    runtime_dir, output_dir, candidate_name = _destinations(seed)
    component_cache_path = runtime_dir / "deep-xgboost.joblib"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    archive = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "archive-synthesis-competition"
        / "archive-components.joblib"
    )
    for name, payload in (
        ("accepted", accepted),
        ("identity", identity),
        ("archive", archive),
    ):
        _validate_component_cache(payload, modelling_data, name)
    deep = _load_or_fit_deep_component(
        modelling_data,
        seed=seed,
        cache_path=component_cache_path,
    )
    components = {
        "deep_xgboost": deep["probabilities"],
        "spatial_xgboost": archive["spatial_xgboost"],
        "random_forest": accepted["random_forest_probabilities"],
        "frequency_random_forest": archive["frequency_random_forest"],
        "spatial_random_forest": archive["spatial_random_forest"],
        "identity_catboost": identity["probabilities"],
    }
    probabilities = blend_deep_archive(components)
    seconds = pd.Series(
        {
            "deep top-50 XGBoost": float(deep["fit_and_predict_seconds"]),
            "spatial-height XGBoost": float(
                archive["component_seconds"].loc["spatial-height XGBoost"]
            ),
            "Random Forest": float(
                accepted["component_seconds"].loc["Random Forest"]
            ),
            "frequency Random Forest": float(
                archive["component_seconds"].loc["frequency Random Forest"]
            ),
            "spatial-height Random Forest": float(
                archive["component_seconds"].loc[
                    "spatial-height Random Forest"
                ]
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
    destination = output_dir / candidate_name
    write_validated_submission(prediction, destination)
    reloaded = pd.read_csv(destination)
    _validate_written_submission(
        reloaded,
        submission_template,
        expected_rows=len(modelling_data.X_competition),
    )
    archive_submission = pd.read_csv(
        STAGE_DIR
        / "submissions"
        / "2026-08-22-archive-synthesis"
        / (
            "01-33-xgboost-11-spatial-xgboost-18-random-forest-"
            "09-frequency-forest-09-spatial-forest-20-identity-catboost.csv"
        )
    )
    manifest = {
        "date": "2026-08-23",
        "status": "prepared_not_uploaded",
        "selection": SELECTION_BY_SEED[seed],
        "weights": DEEP_ARCHIVE_WEIGHTS,
        "seed": seed,
        "iterations": {
            "deep_top_50_xgboost": ARCHIVED_DEPTH_17_ITERATIONS,
            "spatial_height_xgboost": int(
                archive["spatial_xgboost_iterations"]
            ),
            "complete_identity_catboost": int(identity["iterations"]),
        },
        "competition_prediction_disagreement_vs_archive": float(
            reloaded["status_group"].ne(archive_submission["status_group"]).mean()
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
        "daily_submission_allowance": "2/3 used; candidate held for final slot",
    }
    if seed != DEEP_XGBOOST_SEED:
        incumbent_submission = pd.read_csv(
            DEFAULT_OUTPUT_DIR / DEFAULT_CANDIDATE_NAME
        )
        manifest["competition_prediction_disagreement_vs_seed_20260822"] = float(
            reloaded["status_group"]
            .ne(incumbent_submission["status_group"])
            .mean()
        )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def _load_or_fit_deep_component(modelling_data, *, seed: int, cache_path: Path):
    if cache_path.exists():
        payload = joblib.load(cache_path)
        _validate_component_cache(payload, modelling_data, "deep")
        if payload.get("seed") != seed:
            raise ValueError("Cached deep XGBoost seed changed.")
        if payload.get("iterations") != ARCHIVED_DEPTH_17_ITERATIONS:
            raise ValueError("Cached deep XGBoost tree count changed.")
        validate_probabilities(
            payload["probabilities"],
            len(modelling_data.X_competition),
        )
        print(f"Loaded {cache_path}.", flush=True)
        return payload

    probabilities, seconds = fit_gpu_candidate_probabilities(
        make_xgboost_spec(
            variant="archived depth 17",
            seed=seed,
        ),
        modelling_data.X_original,
        modelling_data.y_original,
        modelling_data.X_competition,
        iterations=ARCHIVED_DEPTH_17_ITERATIONS,
        preprocessor_factory=make_top_common_identity_preprocessor,
    )
    payload = {
        "competition_ids": modelling_data.competition_ids.copy(),
        "probabilities": probabilities,
        "seed": seed,
        "iterations": ARCHIVED_DEPTH_17_ITERATIONS,
        "fit_and_predict_seconds": seconds,
    }
    joblib.dump(payload, cache_path)
    return payload


def _destinations(seed: int) -> tuple[Path, Path, str]:
    if seed not in SELECTION_BY_SEED:
        raise ValueError(f"Seed {seed} has no recorded development/local evidence.")
    if seed == DEEP_XGBOOST_SEED:
        return (
            DEFAULT_RUNTIME_DIR,
            DEFAULT_OUTPUT_DIR,
            DEFAULT_CANDIDATE_NAME,
        )
    return (
        PROJECT_DIR / ".runtime" / f"deep-archive-seed-{seed}-competition",
        STAGE_DIR / "submissions" / f"2026-08-23-deep-archive-seed-{seed}",
        f"01-deep-archive-seed-{seed}.csv",
    )


def _validate_component_cache(payload, modelling_data, name: str) -> None:
    if not payload["competition_ids"].reset_index(drop=True).equals(
        modelling_data.competition_ids.reset_index(drop=True)
    ):
        raise ValueError(f"Cached {name} component IDs changed.")


def _validate_written_submission(submission, template, *, expected_rows: int) -> None:
    if list(submission.columns) != ["id", "status_group"]:
        raise ValueError("Deep archive candidate has invalid columns.")
    if len(submission) != expected_rows:
        raise ValueError("Deep archive candidate has the wrong row count.")
    if submission["id"].duplicated().any() or submission.isna().any().any():
        raise ValueError("Deep archive candidate contains missing or duplicate rows.")
    if not submission["id"].equals(template["id"]):
        raise ValueError("Deep archive candidate changed template ID order.")
    if not set(submission["status_group"]).issubset(CLASS_LABELS):
        raise ValueError("Deep archive candidate contains an invalid label.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        choices=tuple(SELECTION_BY_SEED),
        default=DEEP_XGBOOST_SEED,
    )
    main(seed=parser.parse_args().seed)

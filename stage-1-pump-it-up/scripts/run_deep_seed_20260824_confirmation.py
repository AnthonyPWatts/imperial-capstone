"""Confirm the strongest predeclared deep seed on the used local test."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "deep-seed-20260824-confirmation"
CACHE_PATH = OUTPUT_DIR / "local-test-confirmation.joblib"
SEED = 20260824
ITERATIONS = 600
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_confirmation import score_identity_recipes
from catboost_identity_confirmation import summarise_paired_predictions
from data_partitioning import partition_modelling_data
from deep_archive_confirmation import blend_deep_archive
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from modelling_data import prepare_modelling_data
from top_common_identity_evaluation import make_top_common_identity_preprocessor


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned = partition_modelling_data(
        prepare_modelling_data(
            pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
            pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
            pd.read_csv(DATA_DIR / "TestSetValues.csv"),
        )
    )
    deep_payload = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "deep-archive-confirmation"
        / "local-test-confirmation.joblib"
    )
    _validate_payload(deep_payload, partitioned, "Deep archive")
    incumbent_confirmation = deep_payload["confirmation"]
    incumbent = incumbent_confirmation.probabilities[
        "deep_archive_substitution"
    ]

    if CACHE_PATH.exists():
        payload = joblib.load(CACHE_PATH)
        _validate_payload(payload, partitioned, "Deep seed 20260824")
        candidate = payload["candidate"]
        seconds = payload["fit_and_predict_seconds"]
        print(f"Loaded {CACHE_PATH}.", flush=True)
    else:
        spec = make_xgboost_spec(variant="archived depth 17", seed=SEED)
        deep, seconds = fit_gpu_candidate_probabilities(
            spec,
            partitioned.X_development,
            partitioned.y_development,
            partitioned.X_local_test,
            iterations=ITERATIONS,
            preprocessor_factory=make_top_common_identity_preprocessor,
        )
        components = {
            key: value
            for key, value in incumbent_confirmation.probabilities.items()
            if key
            in {
                "spatial_xgboost",
                "random_forest",
                "frequency_random_forest",
                "spatial_random_forest",
                "identity_catboost",
            }
        }
        components["deep_xgboost"] = deep
        candidate = blend_deep_archive(components)
        payload = {
            "local_test_ids": partitioned.local_test_ids.copy(),
            "local_test_fingerprint": partitioned.local_test_fingerprint,
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "candidate": candidate,
            "deep_xgboost": deep,
            "seed": SEED,
            "iterations": ITERATIONS,
            "fit_and_predict_seconds": seconds,
        }
        joblib.dump(payload, CACHE_PATH)

    metrics, confusions = score_identity_recipes(
        partitioned.y_local_test,
        {
            "accepted_55_45": incumbent,
            "identity_candidate_44_36_20": candidate,
        },
    )
    metrics = metrics.rename(
        columns={
            "accepted_55_45": "deep_seed_20260822",
            "identity_candidate_44_36_20": "deep_seed_20260824",
            "candidate_change": "seed_20260824_change",
        }
    )
    metrics.to_csv(OUTPUT_DIR / "local-test-metrics.csv")
    for name, confusion in confusions.items():
        renamed = {
            "accepted_55_45": "deep_seed_20260822",
            "identity_candidate_44_36_20": "deep_seed_20260824",
        }[name]
        confusion.to_csv(OUTPUT_DIR / f"{renamed}-confusion.csv")
    paired = summarise_paired_predictions(
        partitioned.y_local_test,
        {
            "accepted_55_45": incumbent,
            "identity_candidate_44_36_20": candidate,
        },
    )
    paired.to_csv(OUTPUT_DIR / "paired-seed-comparison.csv", header=True)
    result = {
        "seed": SEED,
        "iterations": ITERATIONS,
        "local_test_rows": len(partitioned.y_local_test),
        "local_test_fingerprint": partitioned.local_test_fingerprint,
        "seed_20260822_accuracy": float(
            metrics.loc["accuracy", "deep_seed_20260822"]
        ),
        "seed_20260824_accuracy": float(
            metrics.loc["accuracy", "deep_seed_20260824"]
        ),
        "change_vs_seed_20260822": float(
            metrics.loc["accuracy", "seed_20260824_change"]
        ),
        "net_additional_correct": int(paired["net_additional_correct"]),
        "fit_and_predict_seconds": float(seconds),
        "weights_retuned_after_local_test": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nPaired deep seeds:\n", paired.to_string())


def _validate_payload(payload, partitioned, label: str) -> None:
    if payload.get("local_test_fingerprint") != partitioned.local_test_fingerprint:
        raise ValueError(f"{label} confirmation uses a different local test.")
    if (
        payload.get("cross_validation_fingerprint")
        != partitioned.cross_validation_fingerprint
    ):
        raise ValueError(f"{label} confirmation uses different folds.")
    if not payload["local_test_ids"].reset_index(drop=True).equals(
        partitioned.local_test_ids.reset_index(drop=True)
    ):
        raise ValueError(f"{label} confirmation local-test IDs changed order.")


if __name__ == "__main__":
    main()

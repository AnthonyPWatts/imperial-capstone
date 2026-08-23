"""Confirm the unchanged deep-XGBoost archive substitution on local test."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "deep-archive-confirmation"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_confirmation import score_identity_recipes
from catboost_identity_confirmation import summarise_paired_predictions
from data_partitioning import partition_modelling_data
from deep_archive_confirmation import fit_deep_archive_on_local_test
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
    archive_payload = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "archive-synthesis-confirmation"
        / "local-test-confirmation.joblib"
    )
    _validate_archive_payload(archive_payload, partitioned)
    archive_confirmation = archive_payload["confirmation"]

    cache_path = OUTPUT_DIR / "local-test-confirmation.joblib"
    if cache_path.exists():
        payload = joblib.load(cache_path)
        _validate_cached_payload(payload, partitioned)
        confirmation = payload["confirmation"]
        print(f"Loaded {cache_path}.", flush=True)
    else:
        cached = {
            key: archive_confirmation.probabilities[key]
            for key in (
                "spatial_xgboost",
                "random_forest",
                "frequency_random_forest",
                "spatial_random_forest",
                "identity_catboost",
            )
        }
        confirmation = fit_deep_archive_on_local_test(partitioned, cached)
        payload = {
            "local_test_ids": partitioned.local_test_ids.copy(),
            "local_test_fingerprint": partitioned.local_test_fingerprint,
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "confirmation": confirmation,
        }
        joblib.dump(payload, cache_path)

    archive = archive_confirmation.probabilities["archive_synthesis"]
    candidate = confirmation.probabilities["deep_archive_substitution"]
    metrics, confusions = score_identity_recipes(
        partitioned.y_local_test,
        {
            "accepted_55_45": archive,
            "identity_candidate_44_36_20": candidate,
        },
    )
    metrics = metrics.rename(
        columns={
            "accepted_55_45": "archive_synthesis",
            "identity_candidate_44_36_20": "deep_archive_substitution",
            "candidate_change": "deep_change_vs_archive",
        }
    )
    metrics.to_csv(OUTPUT_DIR / "local-test-metrics.csv")
    for name, confusion in confusions.items():
        renamed = {
            "accepted_55_45": "archive_synthesis",
            "identity_candidate_44_36_20": "deep_archive_substitution",
        }[name]
        confusion.to_csv(OUTPUT_DIR / f"{renamed}-confusion.csv")
    paired = summarise_paired_predictions(
        partitioned.y_local_test,
        {
            "accepted_55_45": archive,
            "identity_candidate_44_36_20": candidate,
        },
    )
    paired.to_csv(OUTPUT_DIR / "paired-archive-vs-deep.csv", header=True)
    result = {
        "local_test_rows": len(partitioned.y_local_test),
        "local_test_fingerprint": partitioned.local_test_fingerprint,
        "archive_synthesis_accuracy": float(
            metrics.loc["accuracy", "archive_synthesis"]
        ),
        "deep_archive_accuracy": float(
            metrics.loc["accuracy", "deep_archive_substitution"]
        ),
        "deep_change_vs_archive": float(
            metrics.loc["accuracy", "deep_change_vs_archive"]
        ),
        "deep_net_additional_correct": int(paired["net_additional_correct"]),
        "deep_mcnemar_p_value": float(paired["exact_mcnemar_p_value"]),
        "deep_change_bootstrap_95": [
            float(paired["accuracy_change_bootstrap_95_low"]),
            float(paired["accuracy_change_bootstrap_95_high"]),
        ],
        "weights_retuned_after_local_test": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nPaired archive vs deep:\n", paired.to_string())


def _validate_archive_payload(payload, partitioned) -> None:
    _validate_payload(payload, partitioned, "Archive")


def _validate_cached_payload(payload, partitioned) -> None:
    _validate_payload(payload, partitioned, "Deep archive")


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

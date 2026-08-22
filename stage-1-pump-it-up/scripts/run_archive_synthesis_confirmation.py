"""Confirm the fixed archive-derived synthesis on the used local test."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "archive-synthesis-confirmation"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archive_synthesis_confirmation import fit_archive_synthesis_on_local_test
from catboost_identity_confirmation import score_identity_recipes
from catboost_identity_confirmation import summarise_paired_predictions
from data_partitioning import partition_modelling_data
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
    identity_payload = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "catboost-identity-confirmation"
        / "local-test-confirmation.joblib"
    )
    _validate_identity_payload(identity_payload, partitioned)
    identity_confirmation = identity_payload["confirmation"]
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

    cache_path = OUTPUT_DIR / "local-test-confirmation.joblib"
    if cache_path.exists():
        payload = joblib.load(cache_path)
        _validate_cached_payload(payload, partitioned)
        confirmation = payload["confirmation"]
        print(f"Loaded {cache_path}.", flush=True)
    else:
        confirmation = fit_archive_synthesis_on_local_test(
            partitioned,
            spatial_trial,
            frequency_trial,
            {
                "xgboost": identity_confirmation.probabilities["xgboost"],
                "random_forest": identity_confirmation.probabilities[
                    "random_forest"
                ],
                "identity_catboost": identity_confirmation.probabilities[
                    "identity_catboost"
                ],
            },
        )
        payload = {
            "local_test_ids": partitioned.local_test_ids.copy(),
            "local_test_fingerprint": partitioned.local_test_fingerprint,
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "confirmation": confirmation,
        }
        joblib.dump(payload, cache_path)

    ready = identity_confirmation.probabilities["identity_candidate_44_36_20"]
    archive = confirmation.probabilities["archive_synthesis"]
    metrics, confusions = score_identity_recipes(
        partitioned.y_local_test,
        {
            "accepted_55_45": ready,
            "identity_candidate_44_36_20": archive,
        },
    )
    metrics = metrics.rename(
        columns={
            "accepted_55_45": "ready_identity_44_36_20",
            "identity_candidate_44_36_20": "archive_synthesis",
            "candidate_change": "archive_change_vs_ready",
        }
    )
    metrics.to_csv(OUTPUT_DIR / "local-test-metrics.csv")
    for name, confusion in confusions.items():
        renamed = {
            "accepted_55_45": "ready_identity_44_36_20",
            "identity_candidate_44_36_20": "archive_synthesis",
        }[name]
        confusion.to_csv(OUTPUT_DIR / f"{renamed}-confusion.csv")
    paired = summarise_paired_predictions(
        partitioned.y_local_test,
        {
            "accepted_55_45": ready,
            "identity_candidate_44_36_20": archive,
        },
    )
    paired.to_csv(OUTPUT_DIR / "paired-ready-vs-archive.csv", header=True)
    result = {
        "local_test_rows": len(partitioned.y_local_test),
        "local_test_fingerprint": partitioned.local_test_fingerprint,
        "ready_identity_accuracy": float(
            metrics.loc["accuracy", "ready_identity_44_36_20"]
        ),
        "archive_synthesis_accuracy": float(
            metrics.loc["accuracy", "archive_synthesis"]
        ),
        "archive_change_vs_ready": float(
            metrics.loc["accuracy", "archive_change_vs_ready"]
        ),
        "archive_net_additional_correct": int(paired["net_additional_correct"]),
        "archive_mcnemar_p_value": float(paired["exact_mcnemar_p_value"]),
        "archive_change_bootstrap_95": [
            float(paired["accuracy_change_bootstrap_95_low"]),
            float(paired["accuracy_change_bootstrap_95_high"]),
        ],
        "development_gate_previously_passed": False,
        "weights_retuned_after_local_test": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nPaired ready identity vs archive:\n", paired.to_string())


def _validate_identity_payload(payload, partitioned) -> None:
    if payload.get("local_test_fingerprint") != partitioned.local_test_fingerprint:
        raise ValueError("Identity confirmation uses a different local test.")
    if (
        payload.get("cross_validation_fingerprint")
        != partitioned.cross_validation_fingerprint
    ):
        raise ValueError("Identity confirmation uses different development folds.")
    if not payload["local_test_ids"].reset_index(drop=True).equals(
        partitioned.local_test_ids.reset_index(drop=True)
    ):
        raise ValueError("Identity confirmation local-test IDs changed order.")


def _validate_cached_payload(payload, partitioned) -> None:
    if payload.get("local_test_fingerprint") != partitioned.local_test_fingerprint:
        raise ValueError("Cached archive confirmation uses a different local test.")
    if (
        payload.get("cross_validation_fingerprint")
        != partitioned.cross_validation_fingerprint
    ):
        raise ValueError("Cached archive confirmation uses different folds.")
    if not payload["local_test_ids"].reset_index(drop=True).equals(
        partitioned.local_test_ids.reset_index(drop=True)
    ):
        raise ValueError("Cached archive confirmation local-test IDs changed order.")


if __name__ == "__main__":
    main()

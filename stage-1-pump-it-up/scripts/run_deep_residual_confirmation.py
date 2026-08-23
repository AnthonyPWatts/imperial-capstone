"""Confirm the fixed deep residual-voter recipe on the used local test."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "deep-residual-confirmation"
CACHE_PATH = OUTPUT_DIR / "local-test-confirmation.joblib"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_confirmation import score_identity_recipes
from catboost_identity_confirmation import summarise_paired_predictions
from data_partitioning import partition_modelling_data
from deep_residual_confirmation import fit_deep_residual_on_local_test
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
    location = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "target-encoding-screen"
        / "frozen-location_identity.joblib"
    )
    grid = joblib.load(
        PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-grid_10km.joblib"
    )
    deep_payload = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "deep-archive-confirmation"
        / "local-test-confirmation.joblib"
    )
    _validate_payload(deep_payload, partitioned, "Deep archive")
    deep_archive = deep_payload["confirmation"].probabilities[
        "deep_archive_substitution"
    ]

    if CACHE_PATH.exists():
        payload = joblib.load(CACHE_PATH)
        _validate_payload(payload, partitioned, "Deep residual")
        confirmation = payload["confirmation"]
        print(f"Loaded {CACHE_PATH}.", flush=True)
    else:
        confirmation = fit_deep_residual_on_local_test(
            partitioned,
            location,
            grid,
            deep_archive,
        )
        payload = {
            "local_test_ids": partitioned.local_test_ids.copy(),
            "local_test_fingerprint": partitioned.local_test_fingerprint,
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "confirmation": confirmation,
        }
        joblib.dump(payload, CACHE_PATH)

    candidate = confirmation.probabilities["deep_residual_candidate"]
    metrics, confusions = score_identity_recipes(
        partitioned.y_local_test,
        {
            "accepted_55_45": deep_archive,
            "identity_candidate_44_36_20": candidate,
        },
    )
    metrics = metrics.rename(
        columns={
            "accepted_55_45": "deep_archive",
            "identity_candidate_44_36_20": "deep_residual_candidate",
            "candidate_change": "residual_change_vs_deep_archive",
        }
    )
    metrics.to_csv(OUTPUT_DIR / "local-test-metrics.csv")
    for name, confusion in confusions.items():
        renamed = {
            "accepted_55_45": "deep_archive",
            "identity_candidate_44_36_20": "deep_residual_candidate",
        }[name]
        confusion.to_csv(OUTPUT_DIR / f"{renamed}-confusion.csv")
    paired = summarise_paired_predictions(
        partitioned.y_local_test,
        {
            "accepted_55_45": deep_archive,
            "identity_candidate_44_36_20": candidate,
        },
    )
    paired.to_csv(OUTPUT_DIR / "paired-deep-vs-residual.csv", header=True)
    result = {
        "local_test_rows": len(partitioned.y_local_test),
        "local_test_fingerprint": partitioned.local_test_fingerprint,
        "deep_archive_accuracy": float(metrics.loc["accuracy", "deep_archive"]),
        "deep_residual_accuracy": float(
            metrics.loc["accuracy", "deep_residual_candidate"]
        ),
        "change_vs_deep_archive": float(
            metrics.loc["accuracy", "residual_change_vs_deep_archive"]
        ),
        "net_additional_correct": int(paired["net_additional_correct"]),
        "exact_mcnemar_p_value": float(paired["exact_mcnemar_p_value"]),
        "weights_retuned_after_local_test": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nPaired deep vs residual:\n", paired.to_string())


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

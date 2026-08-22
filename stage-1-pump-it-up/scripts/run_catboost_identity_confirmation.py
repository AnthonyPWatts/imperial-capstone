"""Confirm the frozen complete-identity CatBoost vote on the local test."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "catboost-identity-confirmation"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
IDENTITY_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "catboost-identity-screen"
    / "complete-deferred-identities.joblib"
)
CONFIRMATION_PATH = OUTPUT_DIR / "local-test-confirmation.joblib"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_confirmation import fit_frozen_identity_recipes_on_local_test
from catboost_identity_confirmation import summarise_paired_predictions
from data_partitioning import partition_modelling_data
from modelling_data import prepare_modelling_data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = partition_modelling_data(modelling_data)
    baseline = joblib.load(BASELINE_PATH)
    identity_trial = joblib.load(IDENTITY_PATH)
    if CONFIRMATION_PATH.exists():
        payload = joblib.load(CONFIRMATION_PATH)
        _validate_cached_payload(payload, partitioned)
        confirmation = payload["confirmation"]
        print(f"Loaded {CONFIRMATION_PATH}.", flush=True)
    else:
        confirmation = fit_frozen_identity_recipes_on_local_test(
            partitioned,
            baseline.xgboost,
            identity_trial.evaluation,
        )
        payload = {
            "local_test_ids": partitioned.local_test_ids.copy(),
            "local_test_fingerprint": partitioned.local_test_fingerprint,
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "confirmation": confirmation,
        }
        joblib.dump(payload, CONFIRMATION_PATH)

    confirmation.metrics.to_csv(OUTPUT_DIR / "local-test-metrics.csv")
    paired = summarise_paired_predictions(
        partitioned.y_local_test,
        {
            "accepted_55_45": confirmation.probabilities["accepted_55_45"],
            "identity_candidate_44_36_20": confirmation.probabilities[
                "identity_candidate_44_36_20"
            ],
        },
    )
    paired.to_csv(OUTPUT_DIR / "paired-summary.csv", header=True)
    for name, counts in confirmation.confusion_counts.items():
        counts.to_csv(OUTPUT_DIR / f"{name}-confusion.csv")
    result = {
        "local_test_rows": len(partitioned.y_local_test),
        "local_test_fingerprint": partitioned.local_test_fingerprint,
        "accepted_accuracy": float(
            confirmation.metrics.loc["accuracy", "accepted_55_45"]
        ),
        "candidate_accuracy": float(
            confirmation.metrics.loc[
                "accuracy",
                "identity_candidate_44_36_20",
            ]
        ),
        "candidate_accuracy_change": float(
            confirmation.metrics.loc["accuracy", "candidate_change"]
        ),
        "net_additional_correct": int(paired["net_additional_correct"]),
        "exact_mcnemar_p_value": float(paired["exact_mcnemar_p_value"]),
        "accuracy_change_bootstrap_95": [
            float(paired["accuracy_change_bootstrap_95_low"]),
            float(paired["accuracy_change_bootstrap_95_high"]),
        ],
        "weights_retuned_after_confirmation": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(confirmation.metrics.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nSelected iterations:", flush=True)
    print(confirmation.selected_iterations.to_string(), flush=True)
    print("\nPaired comparison:", flush=True)
    print(paired.to_string(), flush=True)


def _validate_cached_payload(payload, partitioned) -> None:
    if payload.get("local_test_fingerprint") != partitioned.local_test_fingerprint:
        raise ValueError("Cached confirmation uses a different local test.")
    if (
        payload.get("cross_validation_fingerprint")
        != partitioned.cross_validation_fingerprint
    ):
        raise ValueError("Cached confirmation uses different development folds.")
    if not payload["local_test_ids"].reset_index(drop=True).equals(
        partitioned.local_test_ids.reset_index(drop=True)
    ):
        raise ValueError("Cached confirmation local-test IDs changed order.")


if __name__ == "__main__":
    main()

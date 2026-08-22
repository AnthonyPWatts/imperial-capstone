"""Compare tomorrow's source and identity candidates on the used local test."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "source-identity-confirmation"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_confirmation import score_identity_recipes
from catboost_identity_confirmation import summarise_paired_predictions
from data_partitioning import partition_modelling_data
from modelling_data import prepare_modelling_data
from source_identity_confirmation import fit_source_identity_recipes_on_local_test


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned = partition_modelling_data(
        prepare_modelling_data(
            pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
            pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
            pd.read_csv(DATA_DIR / "TestSetValues.csv"),
        )
    )
    source_trial = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "physical-hierarchy-screen"
        / "source-source_plus_class.joblib"
    )
    identity_payload = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "catboost-identity-confirmation"
        / "local-test-confirmation.joblib"
    )
    _validate_identity_payload(identity_payload, partitioned)
    identity_confirmation = identity_payload["confirmation"]

    cache_path = OUTPUT_DIR / "source-plus-class-local-test.joblib"
    if cache_path.exists():
        payload = joblib.load(cache_path)
        _validate_cached_payload(payload, partitioned)
        source_confirmation = payload["confirmation"]
        print(f"Loaded {cache_path}.", flush=True)
    else:
        source_confirmation = fit_source_identity_recipes_on_local_test(
            partitioned,
            source_trial,
            identity_confirmation.probabilities["identity_catboost"],
        )
        payload = {
            "local_test_ids": partitioned.local_test_ids.copy(),
            "local_test_fingerprint": partitioned.local_test_fingerprint,
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "confirmation": source_confirmation,
        }
        joblib.dump(payload, cache_path)

    source = source_confirmation.probabilities["source_plus_class_55_45"]
    source_identity = source_confirmation.probabilities[
        "source_plus_class_identity_44_36_20"
    ]
    ready_identity = identity_confirmation.probabilities[
        "identity_candidate_44_36_20"
    ]
    source_metrics, source_confusions = score_identity_recipes(
        partitioned.y_local_test,
        {
            "accepted_55_45": source,
            "identity_candidate_44_36_20": source_identity,
        },
    )
    source_metrics = source_metrics.rename(
        columns={
            "accepted_55_45": "source_plus_class_55_45",
            "identity_candidate_44_36_20": (
                "source_plus_class_identity_44_36_20"
            ),
            "candidate_change": "cross_change_vs_source",
        }
    )
    ready_metrics = identity_confirmation.metrics[
        ["identity_candidate_44_36_20"]
    ].rename(
        columns={"identity_candidate_44_36_20": "ready_identity_44_36_20"}
    )
    metrics = pd.concat(
        [
            source_metrics[
                [
                    "source_plus_class_55_45",
                    "source_plus_class_identity_44_36_20",
                    "cross_change_vs_source",
                ]
            ],
            ready_metrics,
        ],
        axis=1,
    )
    metrics["cross_change_vs_ready_identity"] = (
        metrics["source_plus_class_identity_44_36_20"]
        - metrics["ready_identity_44_36_20"]
    )
    metrics.to_csv(OUTPUT_DIR / "local-test-metrics.csv")
    for name, confusion in source_confusions.items():
        renamed = {
            "accepted_55_45": "source_plus_class_55_45",
            "identity_candidate_44_36_20": (
                "source_plus_class_identity_44_36_20"
            ),
        }[name]
        confusion.to_csv(OUTPUT_DIR / f"{renamed}-confusion.csv")
    source_pair = _paired(partitioned.y_local_test, source, source_identity)
    ready_pair = _paired(
        partitioned.y_local_test,
        ready_identity,
        source_identity,
    )
    source_pair.to_csv(OUTPUT_DIR / "paired-source-vs-cross.csv", header=True)
    ready_pair.to_csv(OUTPUT_DIR / "paired-ready-vs-cross.csv", header=True)
    result = {
        "local_test_rows": len(partitioned.y_local_test),
        "local_test_fingerprint": partitioned.local_test_fingerprint,
        "source_plus_class_accuracy": float(
            metrics.loc["accuracy", "source_plus_class_55_45"]
        ),
        "ready_identity_accuracy": float(
            metrics.loc["accuracy", "ready_identity_44_36_20"]
        ),
        "source_identity_accuracy": float(
            metrics.loc[
                "accuracy",
                "source_plus_class_identity_44_36_20",
            ]
        ),
        "source_identity_change_vs_source": float(
            metrics.loc["accuracy", "cross_change_vs_source"]
        ),
        "source_identity_change_vs_ready_identity": float(
            metrics.loc["accuracy", "cross_change_vs_ready_identity"]
        ),
        "source_identity_net_correct_vs_source": int(
            source_pair["net_additional_correct"]
        ),
        "source_identity_net_correct_vs_ready_identity": int(
            ready_pair["net_additional_correct"]
        ),
        "source_identity_mcnemar_vs_source": float(
            source_pair["exact_mcnemar_p_value"]
        ),
        "source_identity_mcnemar_vs_ready_identity": float(
            ready_pair["exact_mcnemar_p_value"]
        ),
        "development_gate_previously_passed": False,
        "weights_retuned_after_local_test": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(metrics.to_string(float_format=lambda value: f"{value:.6f}"))
    print("\nPaired source baseline vs cross:\n", source_pair.to_string())
    print("\nPaired ready identity vs cross:\n", ready_pair.to_string())


def _paired(y_true, accepted, candidate):
    return summarise_paired_predictions(
        y_true,
        {
            "accepted_55_45": accepted,
            "identity_candidate_44_36_20": candidate,
        },
    )


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
        raise ValueError("Cached source confirmation uses a different local test.")
    if (
        payload.get("cross_validation_fingerprint")
        != partitioned.cross_validation_fingerprint
    ):
        raise ValueError("Cached source confirmation uses different folds.")
    if not payload["local_test_ids"].reset_index(drop=True).equals(
        partitioned.local_test_ids.reset_index(drop=True)
    ):
        raise ValueError("Cached source confirmation local-test IDs changed order.")


if __name__ == "__main__":
    main()

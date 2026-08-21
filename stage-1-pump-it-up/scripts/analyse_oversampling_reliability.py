"""Analyse probability reliability and edge cases without fitting new models."""

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
RUNTIME_DIR = PROJECT_DIR / ".runtime"
OUTPUT_DIR = RUNTIME_DIR / "oversampled-reliability"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import partition_modelling_data
from modelling_data import prepare_modelling_data
from oversampled_submission_replay import SUBMISSION_RECIPES
from oversampled_submission_replay import blend_recipe_probabilities
from probability_diagnostics import CLASS_LABELS
from probability_diagnostics import classwise_probability_scores
from probability_diagnostics import classwise_reliability
from probability_diagnostics import component_disagreement_summary
from probability_diagnostics import edge_case_table
from probability_diagnostics import probability_quality_summary
from probability_diagnostics import top_label_reliability


REGIMES = {
    "original training": (
        RUNTIME_DIR
        / "oversampled-submission-replay"
        / "local-test-original-components.joblib"
    ),
    "2.5x repair training": (
        RUNTIME_DIR
        / "oversampled-submission-replay-250x"
        / "local-test-oversampled-components.joblib"
    ),
}


def main() -> None:
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = partition_modelling_data(modelling_data)
    actual_shares = (
        partitioned.y_local_test.value_counts(normalize=True)
        .reindex(CLASS_LABELS, fill_value=0.0)
    )

    records = {}
    for regime, checkpoint in REGIMES.items():
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        fits = joblib.load(checkpoint)
        recipe_records = {}
        for recipe_name, recipe in SUBMISSION_RECIPES.items():
            probabilities = blend_recipe_probabilities(fits, recipe)
            components = {
                component: fits[component].probabilities
                for component, _ in recipe
            }
            recipe_records[recipe_name] = {
                "quality": probability_quality_summary(
                    partitioned.y_local_test,
                    probabilities,
                ),
                "classwise": _frame_records(
                    classwise_probability_scores(
                        partitioned.y_local_test,
                        probabilities,
                    )
                ),
                "top_label_reliability": _frame_records(
                    top_label_reliability(
                        partitioned.y_local_test,
                        probabilities,
                    )
                ),
                "repair_reliability": _frame_records(
                    classwise_reliability(
                        partitioned.y_local_test,
                        probabilities,
                        label="functional needs repair",
                    )
                ),
                "component_disagreement": component_disagreement_summary(
                    components
                ),
                "edge_cases": _frame_records(
                    edge_case_table(
                        partitioned.y_local_test,
                        probabilities,
                    )
                ),
            }
        records[regime] = recipe_records

    payload = {
        "experiment": "10-probability-reliability-and-edge-cases",
        "new_model_fits": 0,
        "multiplier_frozen": 2.5,
        "local_test_rows": len(partitioned.y_local_test),
        "local_test_fingerprint": partitioned.local_test_fingerprint,
        "actual_class_share": {
            label: float(actual_shares[label]) for label in CLASS_LABELS
        },
        "regimes": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "results.json"
    destination.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Recorded reliability diagnostics for {len(REGIMES)} regimes and "
        f"{len(SUBMISSION_RECIPES)} recipes at {destination}."
    )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    reset = frame.reset_index()
    return [
        {
            str(key): (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in row.items()
        }
        for row in reset.to_dict(orient="records")
    ]


if __name__ == "__main__":
    main()

"""Focused tests for generalisation-diagnosis summaries."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generalisation_diagnosis import FOLD_CACHE_VERSION
from generalisation_diagnosis import make_fold_cache_payload
from generalisation_diagnosis import stratified_accuracy_difference_interval
from generalisation_diagnosis import summarise_predictions, wilson_interval
from generalisation_diagnosis import validate_fold_cache_payload


def _predictions() -> pd.DataFrame:
    rows = []
    labels = ("functional", "functional needs repair", "non functional")
    for historical_local in (True, False):
        for label in labels:
            rows.extend(
                [
                    {"actual": label, "predicted": label, "historical_local": historical_local},
                    {"actual": label, "predicted": "functional", "historical_local": historical_local},
                ]
            )
    return pd.DataFrame(rows)


def _fold_cache_values():
    metadata = {
        "source_sha256": {
            "TrainingSetValues.csv": "values-sha",
            "TrainingSetLabels.csv": "labels-sha",
        },
        "model_recipe": {
            "model_seed": 20260905,
            "iterations": 600,
            "class_order": [
                "functional",
                "functional needs repair",
                "non functional",
            ],
        },
        "cross_validation": {
            "folds": 5,
            "seed": 20260905,
            "fingerprint": "fold-sha",
            "validation_fold": 1,
        },
    }
    identifiers = np.asarray([10, 20])
    probabilities = np.asarray(
        [[0.7, 0.1, 0.2], [0.1, 0.2, 0.7]],
        dtype="float64",
    )
    return metadata, identifiers, probabilities


class GeneralisationDiagnosisTests(unittest.TestCase):
    def test_fold_cache_round_trip_accepts_exact_contract(self) -> None:
        metadata, identifiers, probabilities = _fold_cache_values()

        payload = make_fold_cache_payload(
            metadata,
            identifiers,
            probabilities,
            12.5,
        )
        loaded_probabilities, fit_seconds = validate_fold_cache_payload(
            payload,
            metadata,
            identifiers,
        )

        np.testing.assert_array_equal(loaded_probabilities, probabilities)
        self.assertEqual(fit_seconds, 12.5)
        self.assertEqual(payload["cache_version"], FOLD_CACHE_VERSION)

    def test_fold_cache_rejects_missing_or_extra_schema_fields(self) -> None:
        metadata, identifiers, probabilities = _fold_cache_values()
        valid = make_fold_cache_payload(metadata, identifiers, probabilities, 1.0)
        malformed = (
            {key: value for key, value in valid.items() if key != "fit_seconds"},
            {**valid, "unexpected": True},
        )

        for payload in malformed:
            with self.subTest(keys=sorted(payload)):
                with self.assertRaisesRegex(ValueError, "exact schema"):
                    validate_fold_cache_payload(payload, metadata, identifiers)

    def test_fold_cache_rejects_stale_or_non_integer_version(self) -> None:
        metadata, identifiers, probabilities = _fold_cache_values()
        valid = make_fold_cache_payload(metadata, identifiers, probabilities, 1.0)

        for version in (FOLD_CACHE_VERSION + 1, True):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "cache version"):
                    validate_fold_cache_payload(
                        {**valid, "cache_version": version},
                        metadata,
                        identifiers,
                    )

    def test_fold_cache_rejects_stale_source_model_and_fold_metadata(self) -> None:
        metadata, identifiers, probabilities = _fold_cache_values()
        valid = make_fold_cache_payload(metadata, identifiers, probabilities, 1.0)
        stale_values = []
        for path, value in (
            (("source_sha256", "TrainingSetValues.csv"), "changed"),
            (("model_recipe", "model_seed"), 7),
            (("cross_validation", "fingerprint"), "changed"),
        ):
            stale = deepcopy(metadata)
            stale[path[0]][path[1]] = value
            stale_values.append(stale)

        for stale in stale_values:
            with self.subTest(metadata=stale):
                with self.assertRaisesRegex(ValueError, "pinned metadata"):
                    validate_fold_cache_payload(valid, stale, identifiers)

    def test_fold_cache_rejects_misordered_validation_ids(self) -> None:
        metadata, identifiers, probabilities = _fold_cache_values()
        payload = make_fold_cache_payload(metadata, identifiers, probabilities, 1.0)

        with self.assertRaisesRegex(ValueError, "misordered"):
            validate_fold_cache_payload(
                payload,
                metadata,
                identifiers[::-1],
            )

        with self.assertRaisesRegex(ValueError, "misordered"):
            validate_fold_cache_payload(
                {**payload, "validation_ids": identifiers.astype("float64")},
                metadata,
                identifiers,
            )

    def test_fold_cache_rejects_broadcastable_probability_shape(self) -> None:
        metadata, identifiers, probabilities = _fold_cache_values()
        payload = make_fold_cache_payload(metadata, identifiers, probabilities, 1.0)
        payload["probabilities"] = np.asarray([[1.0], [1.0]])

        with self.assertRaisesRegex(ValueError, "probability shape"):
            validate_fold_cache_payload(payload, metadata, identifiers)

    def test_fold_cache_rejects_invalid_probabilities_and_timing(self) -> None:
        metadata, identifiers, probabilities = _fold_cache_values()
        valid = make_fold_cache_payload(metadata, identifiers, probabilities, 1.0)
        invalid_probabilities = valid.copy()
        invalid_probabilities["probabilities"] = np.asarray(
            [[0.7, 0.2, 0.2], [0.1, 0.2, 0.7]]
        )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_fold_cache_payload(
                invalid_probabilities,
                metadata,
                identifiers,
            )

        for timing in (float("nan"), float("inf"), 0.0, True):
            malformed = {**valid, "fit_seconds": timing}
            with self.subTest(timing=timing):
                with self.assertRaisesRegex(ValueError, "timing"):
                    validate_fold_cache_payload(
                        malformed,
                        metadata,
                        identifiers,
                    )

    def test_summary_reports_all_rows_and_both_historical_groups(self) -> None:
        accuracy, recall = summarise_predictions(_predictions())

        self.assertEqual(
            accuracy["group"].tolist(),
            ["all_labelled", "historical_local", "all_other_ids"],
        )
        self.assertEqual(accuracy["rows"].tolist(), [12, 6, 6])
        self.assertEqual(len(recall), 9)
        self.assertEqual(
            recall.groupby("group")["actual_rows"].sum().tolist(),
            [12, 6, 6],
        )

    def test_stratified_bootstrap_is_deterministic_and_unpaired(self) -> None:
        first = stratified_accuracy_difference_interval(
            _predictions(), seed=7, draws=100
        )
        second = stratified_accuracy_difference_interval(
            _predictions(), seed=7, draws=100
        )

        self.assertEqual(first, second)
        self.assertIs(first["paired"], False)
        self.assertAlmostEqual(first["accuracy_difference"], 0.0)
        self.assertEqual(
            first["method"],
            "class-stratified parametric binomial-count bootstrap",
        )
        self.assertIn("observed class composition", first["estimand"])

    def test_wilson_interval_rejects_empty_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            wilson_interval(0, 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for the exact recording-batch CatBoost representation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_evaluation import DEFERRED_IDENTITY_FEATURES
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import EXPECTED_SOURCE_FEATURES
from recording_batch_catboost_evaluation import RECORDING_BATCH_FEATURE
from recording_batch_catboost_evaluation import (
    engineer_recording_batch_catboost_features,
)


class RecordingBatchCatBoostEvaluationTests(unittest.TestCase):
    def test_exact_date_is_appended_as_a_native_category(self) -> None:
        engineered, categorical = engineer_recording_batch_catboost_features(
            _source_frame(["2013-01-02"])
        )

        self.assertEqual(
            categorical,
            (
                *CATEGORICAL_FEATURES,
                *DEFERRED_IDENTITY_FEATURES,
                RECORDING_BATCH_FEATURE,
            ),
        )
        self.assertEqual(
            engineered.loc[0, RECORDING_BATCH_FEATURE],
            "2013-01-02",
        )

    def test_invalid_source_date_preserves_strict_shared_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "date_recorded must be complete"):
            engineer_recording_batch_catboost_features(_source_frame([None]))


    def test_source_frame_is_not_mutated(self) -> None:
        source = _source_frame(["2013-01-02"])
        original = source.copy(deep=True)

        engineer_recording_batch_catboost_features(source)

        pd.testing.assert_frame_equal(source, original)


def _source_frame(dates: list[object]) -> pd.DataFrame:
    rows = []
    for date in dates:
        values: dict[str, object] = {
            column: "value" for column in EXPECTED_SOURCE_FEATURES
        }
        values.update(
            {
                "amount_tsh": 10.0,
                "date_recorded": date,
                "gps_height": 100.0,
                "longitude": 31.0,
                "latitude": -6.0,
                "num_private": 0.0,
                "region_code": 11,
                "district_code": 4,
                "population": 100.0,
                "construction_year": 2000,
            }
        )
        rows.append(values)
    return pd.DataFrame(rows, columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

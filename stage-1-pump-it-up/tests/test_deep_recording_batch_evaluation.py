"""Tests for exact recording batches in deep XGBoost preprocessing."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_recording_batch_evaluation import RecordingBatchEncoder
from feature_engineering import EXPECTED_SOURCE_FEATURES


class DeepRecordingBatchEvaluationTests(unittest.TestCase):
    def test_encoder_groups_supported_dates_and_handles_unseen_dates(self) -> None:
        training = _source_frame(
            ["2013-01-01"] * 20 + ["2013-01-02"] * 20 + ["2013-01-03"]
        )
        prediction = _source_frame(["2013-01-01", "2013-02-01"])

        encoder = RecordingBatchEncoder().fit(training)
        transformed = encoder.transform(prediction)

        self.assertEqual(transformed.shape[0], 2)
        self.assertEqual(transformed.shape[1], len(encoder.get_feature_names_out()))
        self.assertTrue(np.isfinite(transformed.data).all())
        self.assertEqual(transformed.getnnz(axis=1).tolist(), [1, 1])

    def test_missing_recording_date_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "date_recorded must be complete"):
            RecordingBatchEncoder().fit(_source_frame([None]))

    def test_source_frame_is_not_mutated(self) -> None:
        source = _source_frame(["2013-01-01"] * 20)
        original = source.copy(deep=True)

        RecordingBatchEncoder().fit_transform(source)

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

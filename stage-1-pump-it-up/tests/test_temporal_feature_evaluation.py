"""Tests for explicit recording-time features."""

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

from feature_engineering import EXPECTED_SOURCE_FEATURES
from temporal_feature_evaluation import TEMPORAL_CATEGORICAL_FEATURES
from temporal_feature_evaluation import engineer_temporal_features
from temporal_feature_evaluation import make_temporal_preprocessor


class TemporalFeatureEvaluationTests(unittest.TestCase):
    def test_recording_categories_are_zero_padded_and_ordered(self) -> None:
        frame = _source_frame()

        engineered = engineer_temporal_features(frame)

        self.assertEqual(
            engineered.columns[-3:].tolist(),
            list(TEMPORAL_CATEGORICAL_FEATURES),
        )
        self.assertEqual(engineered["recorded_year"].tolist(), ["2013", "2011"])
        self.assertEqual(engineered["recorded_month"].tolist(), ["01", "11"])
        self.assertEqual(
            engineered["recorded_year_month"].tolist(),
            ["2013-01", "2011-11"],
        )

    def test_temporal_preprocessor_produces_finite_fold_fitted_values(self) -> None:
        frame = _source_frame()

        transformed = make_temporal_preprocessor().fit_transform(frame)

        self.assertEqual(transformed.shape[0], 2)
        self.assertTrue(np.isfinite(transformed.data).all())


def _source_frame() -> pd.DataFrame:
    defaults: dict[str, object] = {
        name: "value" for name in EXPECTED_SOURCE_FEATURES
    }
    defaults.update(
        {
            "amount_tsh": 10.0,
            "date_recorded": "2013-01-02",
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
    second = defaults.copy()
    second["date_recorded"] = "2011-11-30"
    return pd.DataFrame([defaults, second], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

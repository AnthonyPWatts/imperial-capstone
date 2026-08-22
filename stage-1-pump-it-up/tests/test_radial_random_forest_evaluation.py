"""Tests for radial distance in accepted Random Forest features."""

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
from radial_distance_features import RADIAL_DISTANCE_FEATURE
from radial_random_forest_evaluation import RADIAL_RANDOM_FOREST_MODEL_FEATURES
from radial_random_forest_evaluation import engineer_radial_random_forest_features


class RadialRandomForestEvaluationTests(unittest.TestCase):
    def test_radial_feature_appends_valid_distance_and_missing_sentinel(self) -> None:
        source = _source_frame(2)
        source.loc[1, ["longitude", "latitude"]] = [0.0, -2e-08]

        engineered = engineer_radial_random_forest_features(source)

        self.assertEqual(tuple(engineered.columns), RADIAL_RANDOM_FOREST_MODEL_FEATURES)
        self.assertTrue(np.isfinite(engineered.loc[0, RADIAL_DISTANCE_FEATURE]))
        self.assertTrue(pd.isna(engineered.loc[1, RADIAL_DISTANCE_FEATURE]))

    def test_source_frame_is_not_mutated(self) -> None:
        source = _source_frame(1)
        original = source.copy(deep=True)

        engineer_radial_random_forest_features(source)

        pd.testing.assert_frame_equal(source, original)


def _source_frame(rows: int) -> pd.DataFrame:
    values: dict[str, object] = {
        column: "value" for column in EXPECTED_SOURCE_FEATURES
    }
    values.update(
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
    return pd.DataFrame([values.copy() for _ in range(rows)], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

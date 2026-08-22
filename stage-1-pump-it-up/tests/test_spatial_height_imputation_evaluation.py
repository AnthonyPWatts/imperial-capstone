"""Tests for fold-fitted spatial GPS-height reconstruction."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import EXPECTED_SOURCE_FEATURES
from spatial_height_imputation_evaluation import SpatialHeightFeatureEngineer


class SpatialHeightImputationEvaluationTests(unittest.TestCase):
    def test_prediction_rows_do_not_enter_the_height_model(self) -> None:
        training = _source_frame(
            [(30.0 + i * 0.01, -6.0, 100.0 + i) for i in range(10)]
        )
        prediction = _source_frame([(30.02, -6.0, 0.0)])

        transformer = SpatialHeightFeatureEngineer().fit(training)
        transformed = transformer.transform(prediction)

        self.assertGreater(transformed.loc[0, "gps_height"], 99.0)
        self.assertLess(transformed.loc[0, "gps_height"], 110.0)
        self.assertEqual(transformed.loc[0, "gps_height_missing"], 1)

    def test_invalid_coordinates_use_training_height_median(self) -> None:
        training = _source_frame(
            [(30.0 + i * 0.01, -6.0, float(i * 10)) for i in range(1, 11)]
        )
        prediction = _source_frame([(0.0, -2e-8, 0.0)])

        transformed = SpatialHeightFeatureEngineer().fit(training).transform(
            prediction
        )

        self.assertEqual(transformed.loc[0, "gps_height"], 55.0)

    def test_source_frame_is_not_mutated(self) -> None:
        training = _source_frame(
            [(30.0 + i * 0.01, -6.0, 100.0 + i) for i in range(10)]
        )
        original = training.copy(deep=True)

        SpatialHeightFeatureEngineer().fit_transform(training)

        pd.testing.assert_frame_equal(training, original)


def _source_frame(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    result = []
    for longitude, latitude, height in rows:
        values: dict[str, object] = {
            column: "value" for column in EXPECTED_SOURCE_FEATURES
        }
        values.update(
            {
                "amount_tsh": 10.0,
                "date_recorded": "2013-01-02",
                "gps_height": height,
                "longitude": longitude,
                "latitude": latitude,
                "num_private": 0.0,
                "region_code": 11,
                "district_code": 4,
                "population": 100.0,
                "construction_year": 2000,
            }
        )
        result.append(values)
    return pd.DataFrame(result, columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

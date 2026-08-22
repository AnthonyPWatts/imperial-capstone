"""Tests for target-free fold-fitted waterpoint density."""

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
from spatial_density_features import SpatialDensityTransformer


class SpatialDensityFeatureTests(unittest.TestCase):
    def test_training_transform_excludes_each_row_itself(self) -> None:
        training = _source_frame(12)
        transformer = SpatialDensityTransformer()

        distances = transformer.fit_transform(training)
        prediction_distance = transformer.transform(training.iloc[[0]])

        self.assertEqual(distances.shape, (12, 2))
        self.assertGreater(distances[0, 0], 0)
        self.assertGreater(distances[0, 1], distances[0, 0])
        self.assertAlmostEqual(prediction_distance[0, 0], 0.0)

    def test_invalid_prediction_coordinates_use_training_medians(self) -> None:
        training = _source_frame(12)
        invalid = training.iloc[[0]].copy()
        invalid.loc[:, "longitude"] = 0.0
        invalid.loc[:, "latitude"] = -2e-08
        transformer = SpatialDensityTransformer().fit(training)

        distances = transformer.transform(invalid)

        np.testing.assert_allclose(distances[0], transformer.imputation_medians_)
        self.assertTrue(np.isfinite(distances).all())


def _source_frame(rows: int) -> pd.DataFrame:
    values: dict[str, object] = {
        column: "value" for column in EXPECTED_SOURCE_FEATURES
    }
    result = []
    for row in range(rows):
        current = values.copy()
        current.update(
            {
                "amount_tsh": 10.0,
                "date_recorded": "2013-01-01",
                "gps_height": 100.0,
                "longitude": 31.0 + row * 0.01,
                "latitude": -6.0,
                "num_private": 0.0,
                "region_code": 11,
                "district_code": 4,
                "population": 100.0,
                "construction_year": 2000,
            }
        )
        result.append(current)
    return pd.DataFrame(result, columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

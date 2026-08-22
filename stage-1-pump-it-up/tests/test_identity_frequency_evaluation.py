"""Tests for fold-fitted deferred-identity frequency features."""

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
from identity_frequency_evaluation import IDENTITY_FREQUENCY_FEATURES
from identity_frequency_evaluation import IdentityFrequencyFeatureEngineer


class IdentityFrequencyEvaluationTests(unittest.TestCase):
    def test_mappings_are_fitted_without_prediction_rows(self) -> None:
        training = _source_frame(["Alpha", "Alpha", "Beta"])
        prediction = _source_frame(["Alpha", "Gamma"])
        transformer = IdentityFrequencyFeatureEngineer().fit(training)

        transformed = transformer.transform(prediction)

        self.assertAlmostEqual(
            transformed.loc[0, "wpt_name_log_frequency"],
            np.log1p(2),
        )
        self.assertEqual(transformed.loc[1, "wpt_name_log_frequency"], 0.0)
        self.assertEqual(
            transformed.columns[-6:].tolist(),
            list(IDENTITY_FREQUENCY_FEATURES),
        )

    def test_source_frames_are_not_mutated(self) -> None:
        training = _source_frame(["Alpha", "Beta"])
        original = training.copy(deep=True)

        IdentityFrequencyFeatureEngineer().fit_transform(training)

        pd.testing.assert_frame_equal(training, original)


def _source_frame(names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
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
                "wpt_name": name,
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

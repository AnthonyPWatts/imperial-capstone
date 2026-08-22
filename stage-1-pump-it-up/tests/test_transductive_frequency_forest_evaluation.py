"""Tests for target-free transductive categorical counts."""

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
from transductive_frequency_forest_evaluation import TransductiveFrequencyFeatureEngineer


class TransductiveFrequencyForestEvaluationTests(unittest.TestCase):
    def test_reference_covariates_cross_the_fixed_rare_boundary(self) -> None:
        training = _source_frame(["Alpha"] * 4)
        reference = _source_frame(["Alpha"] * 5 + ["Beta"])
        prediction = _source_frame(["Alpha", "Beta"])

        transformed = TransductiveFrequencyFeatureEngineer(reference).fit(
            training
        ).transform(prediction)

        self.assertEqual(transformed.loc[0, "wpt_name_occurrence_count"], 5)
        self.assertEqual(transformed.loc[1, "wpt_name_occurrence_count"], -1)

    def test_fit_labels_do_not_affect_reference_counts(self) -> None:
        training = _source_frame(["Alpha"] * 4)
        reference = _source_frame(["Alpha"] * 5)

        left = TransductiveFrequencyFeatureEngineer(reference).fit_transform(
            training,
            ["functional"] * 4,
        )
        right = TransductiveFrequencyFeatureEngineer(reference).fit_transform(
            training,
            ["non functional"] * 4,
        )

        pd.testing.assert_frame_equal(left, right)


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

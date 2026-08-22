"""Tests for target-free two-stage numeric reconstruction."""

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
from learned_numeric_imputation_evaluation import LearnedNumericFeatureEngineer


class LearnedNumericImputationEvaluationTests(unittest.TestCase):
    def test_zero_amount_is_reconstructed_and_original_state_is_retained(self) -> None:
        training = _source_frame([10.0 + i for i in range(40)])
        prediction = _source_frame([0.0])

        transformed = LearnedNumericFeatureEngineer().fit(training).transform(
            prediction
        )

        self.assertGreater(transformed.loc[0, "amount_tsh"], 0.0)
        self.assertEqual(transformed.loc[0, "amount_tsh_recorded"], 0)

    def test_class_labels_do_not_affect_numeric_reconstruction(self) -> None:
        training = _source_frame([10.0 + i for i in range(40)])
        prediction = _source_frame([0.0])

        left = LearnedNumericFeatureEngineer().fit(
            training,
            ["functional"] * len(training),
        ).transform(prediction)
        right = LearnedNumericFeatureEngineer().fit(
            training,
            ["non functional"] * len(training),
        ).transform(prediction)

        pd.testing.assert_frame_equal(left, right)

    def test_source_frame_is_not_mutated(self) -> None:
        training = _source_frame([10.0 + i for i in range(40)])
        original = training.copy(deep=True)

        LearnedNumericFeatureEngineer().fit_transform(training)

        pd.testing.assert_frame_equal(training, original)


def _source_frame(amounts: list[float]) -> pd.DataFrame:
    rows = []
    for position, amount in enumerate(amounts):
        values: dict[str, object] = {
            column: "value" for column in EXPECTED_SOURCE_FEATURES
        }
        values.update(
            {
                "amount_tsh": amount,
                "date_recorded": "2013-01-02",
                "gps_height": 100.0 + position,
                "longitude": 30.0 + position * 0.01,
                "latitude": -6.0,
                "num_private": 0.0,
                "region_code": 11,
                "district_code": 4,
                "population": 100.0 + position,
                "construction_year": 2000,
            }
        )
        rows.append(values)
    return pd.DataFrame(rows, columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

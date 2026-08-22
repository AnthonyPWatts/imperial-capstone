"""Tests for the fixed pump-age cohort representation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from age_cohort_evaluation import AGE_COHORT_FEATURE
from age_cohort_evaluation import AgeCohortFeatureEngineer
from age_cohort_evaluation import engineer_age_cohort_features
from feature_engineering import EXPECTED_SOURCE_FEATURES


class AgeCohortEvaluationTests(unittest.TestCase):
    def test_boundaries_and_special_states_are_fixed(self) -> None:
        frame = _source_frame([2013, 2011, 2008, 2003, 1993, 1983, 0, 2014])

        engineered = engineer_age_cohort_features(frame)

        self.assertEqual(
            engineered[AGE_COHORT_FEATURE].tolist(),
            [
                "0-2",
                "0-2",
                "3-5",
                "6-10",
                "11-20",
                "21-30",
                "__unknown__",
                "__inconsistent__",
            ],
        )

    def test_source_frame_is_not_mutated(self) -> None:
        frame = _source_frame([2000, 0])
        original = frame.copy(deep=True)

        AgeCohortFeatureEngineer().fit_transform(frame)

        pd.testing.assert_frame_equal(frame, original)


def _source_frame(construction_years: list[int]) -> pd.DataFrame:
    rows = []
    for construction_year in construction_years:
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
                "construction_year": construction_year,
            }
        )
        rows.append(values)
    return pd.DataFrame(rows, columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

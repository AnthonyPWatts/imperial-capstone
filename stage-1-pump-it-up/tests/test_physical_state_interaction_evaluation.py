"""Tests for quantity-by-physical native categorical interactions."""

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
from physical_state_interaction_evaluation import (
    PHYSICAL_STATE_INTERACTION_FEATURES,
)
from physical_state_interaction_evaluation import (
    engineer_physical_state_interaction_catboost_features,
)


class PhysicalStateInteractionEvaluationTests(unittest.TestCase):
    def test_all_three_composites_are_normalised_and_categorical(self) -> None:
        engineered, categorical = (
            engineer_physical_state_interaction_catboost_features(_source_frame())
        )

        self.assertEqual(
            tuple(engineered.columns[-3:]),
            PHYSICAL_STATE_INTERACTION_FEATURES,
        )
        self.assertEqual(
            engineered.loc[0, "quantity_extraction_type_interaction"],
            "seasonal::gravity",
        )
        self.assertEqual(
            engineered.loc[0, "quantity_source_interaction"],
            "seasonal::spring",
        )
        self.assertEqual(
            engineered.loc[0, "quantity_waterpoint_type_interaction"],
            "seasonal::communal standpipe",
        )
        self.assertEqual(
            categorical,
            (
                *CATEGORICAL_FEATURES,
                *DEFERRED_IDENTITY_FEATURES,
                *PHYSICAL_STATE_INTERACTION_FEATURES,
            ),
        )

    def test_source_frame_is_not_mutated(self) -> None:
        source = _source_frame()
        original = source.copy(deep=True)

        engineer_physical_state_interaction_catboost_features(source)

        pd.testing.assert_frame_equal(source, original)


def _source_frame() -> pd.DataFrame:
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
            "quantity": " Seasonal ",
            "extraction_type": " Gravity ",
            "source": " Spring ",
            "waterpoint_type": "Communal  Standpipe",
        }
    )
    return pd.DataFrame([values], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

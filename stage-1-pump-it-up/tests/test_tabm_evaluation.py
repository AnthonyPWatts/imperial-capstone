"""Focused tests for the fold-safe TabM preprocessing contract."""

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
from tabm_evaluation import encode_categories
from tabm_evaluation import fit_tabm_preprocessor
from tabm_evaluation import select_category_values
from tabm_evaluation import transform_tabm_features


class TabMEvaluationTests(unittest.TestCase):
    def test_frequent_category_selection_uses_lexical_tie_breaking(self) -> None:
        values = pd.Series(["z", "z", "b", "b", "a", "a", "c"])

        selected = select_category_values(values, limit=2)

        self.assertEqual(selected, ("a", "b"))

    def test_rare_and_unseen_categories_share_zero(self) -> None:
        selected = ("common", "second")

        encoded = encode_categories(
            pd.Series(["common", "rare", "second", None]),
            selected,
        )

        np.testing.assert_array_equal(encoded, [1, 0, 2, 0])

    def test_preprocessing_is_finite_and_reserves_unknown_category(self) -> None:
        training = _source_frame(12)
        training.loc[0, "gps_height"] = 0.0
        prediction = training.iloc[[0, 1]].copy()
        prediction.loc[:, "basin"] = "unseen basin"

        state = fit_tabm_preprocessor(training)
        transformed = transform_tabm_features(prediction, state)

        self.assertEqual(transformed.numeric.shape[0], 2)
        self.assertTrue(np.isfinite(transformed.numeric).all())
        np.testing.assert_array_equal(transformed.categorical[:, 0], [0, 0])
        self.assertTrue(
            all(
                cardinality >= 2
                for cardinality in state.category_cardinalities
            )
        )


def _source_frame(rows: int) -> pd.DataFrame:
    defaults: dict[str, object] = {
        name: "value" for name in EXPECTED_SOURCE_FEATURES
    }
    defaults.update(
        {
            "amount_tsh": 10.0,
            "date_recorded": "2013-01-01",
            "funder": "Government",
            "gps_height": 100.0,
            "installer": "DWE",
            "longitude": 31.0,
            "latitude": -6.0,
            "wpt_name": "Pump A",
            "num_private": 0.0,
            "basin": "Lake Victoria",
            "subvillage": "Village",
            "region": "Kagera",
            "region_code": 11,
            "district_code": 4,
            "lga": "Misenyi",
            "ward": "Ward",
            "population": 100.0,
            "public_meeting": True,
            "scheme_management": "VWC",
            "scheme_name": "Scheme",
            "permit": True,
            "construction_year": 2000,
            "extraction_type": "gravity",
            "management": "vwc",
            "payment_type": "never pay",
            "water_quality": "soft",
            "quantity": "enough",
            "source": "spring",
            "waterpoint_type": "communal standpipe",
        }
    )
    records = []
    for position in range(rows):
        record = defaults.copy()
        record["wpt_name"] = f"Pump {position % 3}"
        record["subvillage"] = f"Village {position % 4}"
        record["gps_height"] = float(100 + position)
        records.append(record)
    return pd.DataFrame(records, columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

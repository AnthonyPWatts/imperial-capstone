"""Tests for the complete deferred-identity CatBoost representation."""

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
from catboost_identity_evaluation import CONTEXT_IDENTITY_FEATURES
from catboost_identity_evaluation import PHYSICAL_BACKOFF_FEATURES
from catboost_identity_evaluation import engineer_complete_identity_catboost_features
from catboost_identity_evaluation import engineer_context_identity_catboost_features
from catboost_identity_evaluation import (
    engineer_hierarchy_backoff_identity_catboost_features,
)
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import EXPECTED_SOURCE_FEATURES


class CatBoostIdentityEvaluationTests(unittest.TestCase):
    def test_all_deferred_identities_are_normalised_native_categories(self) -> None:
        frame = _source_frame()

        engineered, categorical = engineer_complete_identity_catboost_features(
            frame
        )

        self.assertEqual(
            categorical,
            (*CATEGORICAL_FEATURES, *DEFERRED_IDENTITY_FEATURES),
        )
        self.assertEqual(engineered.loc[0, "funder_identity"], "government")
        self.assertEqual(engineered.loc[0, "wpt_name_identity"], "pump a")
        self.assertEqual(engineered.loc[1, "funder_identity"], "__missing__")
        self.assertEqual(engineered.loc[1, "scheme_name_identity"], "__blank__")
        self.assertTrue(
            all(
                str(engineered[column].dtype) in {"object", "str"}
                for column in categorical
            )
        )

    def test_context_identities_use_only_predeclared_supported_pairs(self) -> None:
        frame = _source_frame()

        engineered, categorical = engineer_context_identity_catboost_features(
            frame
        )

        self.assertEqual(categorical[-3:], CONTEXT_IDENTITY_FEATURES)
        self.assertEqual(
            engineered.loc[0, "lga_ward_context_identity"],
            "misenyi::value",
        )
        self.assertEqual(
            engineered.loc[0, "lga_scheme_context_identity"],
            "misenyi::scheme",
        )
        self.assertEqual(
            engineered.loc[0, "funder_installer_context_identity"],
            "government::dwe",
        )

    def test_physical_backoffs_add_all_four_deterministic_parents(self) -> None:
        frame = _source_frame()

        engineered, categorical = (
            engineer_hierarchy_backoff_identity_catboost_features(frame)
        )

        self.assertEqual(categorical[-4:], PHYSICAL_BACKOFF_FEATURES)
        self.assertTrue(
            all(
                engineered.loc[0, feature] == "value"
                for feature in PHYSICAL_BACKOFF_FEATURES
            )
        )


def _source_frame() -> pd.DataFrame:
    defaults: dict[str, object] = {
        name: "value" for name in EXPECTED_SOURCE_FEATURES
    }
    defaults.update(
        {
            "amount_tsh": 10.0,
            "date_recorded": "2013-01-01",
            "funder": " Government ",
            "installer": "DWE",
            "wpt_name": "Pump   A",
            "gps_height": 100.0,
            "longitude": 31.0,
            "latitude": -6.0,
            "num_private": 0.0,
            "region_code": 11,
            "district_code": 4,
            "population": 100.0,
            "construction_year": 2000,
            "region": "Kagera",
            "lga": "Misenyi",
            "scheme_name": "Scheme",
        }
    )
    second = defaults.copy()
    second.update({"funder": None, "scheme_name": ""})
    return pd.DataFrame([defaults, second], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

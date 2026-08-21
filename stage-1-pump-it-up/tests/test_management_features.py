"""Tests for deterministic management hierarchy policies."""

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
from feature_engineering import NUMERIC_FEATURES
from feature_engineering import engineer_initial_features
from management_features import MANAGEMENT_GROUP_FEATURE
from management_features import MANAGEMENT_POLICIES
from management_features import MANAGEMENT_SCHEME_COMPOSITE
from management_features import MANAGEMENT_SCHEME_RELATIONSHIP
from management_features import ManagementFeatureEngineer
from management_features import make_management_preprocessor
from management_features import management_categorical_features
from management_features import management_model_features


class ManagementFeatureTests(unittest.TestCase):
    def test_baseline_reproduces_initial_feature_frame(self) -> None:
        frame = _source_frame()
        policy = MANAGEMENT_POLICIES["baseline"]

        actual = ManagementFeatureEngineer(policy).fit_transform(frame)

        pd.testing.assert_frame_equal(actual, engineer_initial_features(frame))

    def test_group_only_replaces_both_source_features(self) -> None:
        frame = _source_frame()
        policy = MANAGEMENT_POLICIES["management_group_only"]

        transformed = ManagementFeatureEngineer(policy).fit_transform(frame)

        self.assertNotIn("management", transformed.columns)
        self.assertNotIn("scheme_management", transformed.columns)
        self.assertEqual(transformed.loc[0, MANAGEMENT_GROUP_FEATURE], "user-group")

    def test_composite_normalises_case_whitespace_and_missingness(self) -> None:
        frame = _source_frame()
        frame.loc[0, "management"] = "  VWC "
        frame.loc[0, "scheme_management"] = "vwc"
        policy = MANAGEMENT_POLICIES["baseline_plus_composite"]

        transformed = ManagementFeatureEngineer(policy).fit_transform(frame)

        self.assertEqual(
            transformed.loc[0, MANAGEMENT_SCHEME_COMPOSITE],
            "vwc::vwc",
        )
        self.assertEqual(
            transformed.loc[2, MANAGEMENT_SCHEME_COMPOSITE],
            "vwc::__missing__",
        )

    def test_relationship_distinguishes_same_different_and_missing(self) -> None:
        frame = _source_frame()
        policy = MANAGEMENT_POLICIES["baseline_plus_relationship"]

        transformed = ManagementFeatureEngineer(policy).fit_transform(frame)

        self.assertEqual(
            transformed[MANAGEMENT_SCHEME_RELATIONSHIP].tolist(),
            ["same", "different", "scheme_missing"],
        )

    def test_policy_feature_contracts_are_disjoint_and_complete(self) -> None:
        for policy in MANAGEMENT_POLICIES.values():
            features = management_model_features(policy)
            categorical = management_categorical_features(policy)

            self.assertEqual(len(features), len(set(features)))
            self.assertTrue(set(NUMERIC_FEATURES).isdisjoint(categorical))
            self.assertEqual(set(NUMERIC_FEATURES) | set(categorical), set(features))

    def test_preprocessor_handles_all_registered_policies(self) -> None:
        frame = pd.concat([_source_frame()] * 8, ignore_index=True)
        for policy in MANAGEMENT_POLICIES.values():
            preprocessor = make_management_preprocessor(policy)
            matrix = preprocessor.fit_transform(frame)

            self.assertEqual(matrix.shape[0], len(frame))
            self.assertEqual(
                matrix.shape[1],
                len(preprocessor.get_feature_names_out()),
            )
            values = matrix.data if hasattr(matrix, "data") else np.asarray(matrix)
            self.assertTrue(np.isfinite(values).all())


def _source_frame() -> pd.DataFrame:
    defaults: dict[str, object] = {
        name: "value" for name in EXPECTED_SOURCE_FEATURES
    }
    defaults.update(
        {
            "amount_tsh": 10.0,
            "date_recorded": "2013-01-01",
            "gps_height": 100.0,
            "longitude": 31.774483,
            "latitude": -0.998916,
            "num_private": 0.0,
            "region_code": 11,
            "district_code": 4,
            "population": 100.0,
            "construction_year": 2000,
            "region": "Kagera",
            "lga": "Misenyi",
            "ward": "Kashenye",
            "management": "vwc",
            "management_group": "user-group",
            "scheme_management": "VWC",
        }
    )
    same = defaults.copy()
    different = defaults.copy()
    different.update(
        {
            "management": "private operator",
            "management_group": "commercial",
            "scheme_management": "Company",
        }
    )
    missing = defaults.copy()
    missing["scheme_management"] = None
    return pd.DataFrame([same, different, missing], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

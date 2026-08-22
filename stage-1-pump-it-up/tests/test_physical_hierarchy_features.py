"""Tests for deterministic physical-category hierarchy policies."""

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
from physical_hierarchy_features import HIERARCHY_FAMILIES
from physical_hierarchy_features import PHYSICAL_HIERARCHY_POLICIES
from physical_hierarchy_features import PhysicalHierarchyFeatureEngineer
from physical_hierarchy_features import hierarchy_categorical_features
from physical_hierarchy_features import hierarchy_model_features
from physical_hierarchy_features import make_physical_hierarchy_preprocessor
from physical_hierarchy_evaluation import COMPONENT_HYBRID_SPECS


class PhysicalHierarchyFeatureTests(unittest.TestCase):
    def test_all_baselines_reproduce_the_initial_feature_frame(self) -> None:
        frame = _source_frame()
        expected = engineer_initial_features(frame)

        for policies in PHYSICAL_HIERARCHY_POLICIES.values():
            actual = PhysicalHierarchyFeatureEngineer(
                policies["baseline"]
            ).fit_transform(frame)
            pd.testing.assert_frame_equal(actual, expected)

    def test_policy_register_has_the_bounded_family_sizes(self) -> None:
        self.assertEqual(
            {key: len(value) for key, value in PHYSICAL_HIERARCHY_POLICIES.items()},
            {"extraction": 8, "source": 8, "quality": 4, "waterpoint": 4},
        )

    def test_parent_only_policy_replaces_the_current_child(self) -> None:
        frame = _source_frame()
        policy = PHYSICAL_HIERARCHY_POLICIES["extraction"]["group_only"]

        transformed = PhysicalHierarchyFeatureEngineer(policy).fit_transform(frame)

        self.assertNotIn("extraction_type", transformed.columns)
        self.assertIn("extraction_type_group", transformed.columns)
        self.assertEqual(transformed.loc[0, "extraction_type_group"], "gravity")

    def test_none_policy_removes_only_the_selected_family(self) -> None:
        frame = _source_frame()
        policy = PHYSICAL_HIERARCHY_POLICIES["source"]["none"]

        transformed = PhysicalHierarchyFeatureEngineer(policy).fit_transform(frame)

        self.assertNotIn("source", transformed.columns)
        self.assertIn("extraction_type", transformed.columns)
        self.assertEqual(len(transformed.columns), 28)

    def test_feature_contracts_are_disjoint_and_complete(self) -> None:
        for policies in PHYSICAL_HIERARCHY_POLICIES.values():
            for policy in policies.values():
                features = hierarchy_model_features(policy)
                categorical = hierarchy_categorical_features(policy)
                self.assertEqual(len(features), len(set(features)))
                self.assertTrue(set(NUMERIC_FEATURES).isdisjoint(categorical))
                self.assertEqual(
                    set(NUMERIC_FEATURES) | set(categorical),
                    set(features),
                )

    def test_preprocessor_handles_every_registered_policy(self) -> None:
        frame = pd.concat([_source_frame()] * 8, ignore_index=True)
        for family_key in HIERARCHY_FAMILIES:
            for policy in PHYSICAL_HIERARCHY_POLICIES[family_key].values():
                preprocessor = make_physical_hierarchy_preprocessor(policy)
                matrix = preprocessor.fit_transform(frame)
                self.assertEqual(matrix.shape[0], len(frame))
                self.assertEqual(
                    matrix.shape[1],
                    len(preprocessor.get_feature_names_out()),
                )
                values = matrix.data if hasattr(matrix, "data") else np.asarray(matrix)
                self.assertTrue(np.isfinite(values).all())

    def test_component_hybrid_screen_is_bounded_and_unique(self) -> None:
        self.assertEqual(len(COMPONENT_HYBRID_SPECS), 6)
        self.assertEqual(
            len({spec.key for spec in COMPONENT_HYBRID_SPECS}),
            len(COMPONENT_HYBRID_SPECS),
        )


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
            "extraction_type": "gravity",
            "extraction_type_group": "gravity",
            "extraction_type_class": "gravity",
            "source": "spring",
            "source_type": "spring",
            "source_class": "groundwater",
            "water_quality": "soft",
            "quality_group": "good",
            "waterpoint_type": "communal standpipe",
            "waterpoint_type_group": "communal standpipe",
        }
    )
    alternate = defaults.copy()
    alternate.update(
        {
            "extraction_type": "nira/tanira",
            "extraction_type_group": "nira/tanira",
            "extraction_type_class": "handpump",
            "source": "river",
            "source_type": "river/lake",
            "source_class": "surface",
            "water_quality": "salty abandoned",
            "quality_group": "salty",
            "waterpoint_type": "communal standpipe multiple",
            "waterpoint_type_group": "communal standpipe",
        }
    )
    return pd.DataFrame([defaults, alternate], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

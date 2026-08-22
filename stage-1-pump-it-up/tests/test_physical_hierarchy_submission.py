"""Focused tests for physical-hierarchy submission recipes."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from physical_hierarchy_submission import PHYSICAL_HIERARCHY_SUBMISSION_SPECS


class PhysicalHierarchySubmissionTests(unittest.TestCase):
    def test_three_fixed_entries_use_only_four_unique_components(self) -> None:
        self.assertEqual(len(PHYSICAL_HIERARCHY_SUBMISSION_SPECS), 3)
        components = {
            ("xgboost", spec.xgboost_policy.family, spec.xgboost_policy.key)
            for spec in PHYSICAL_HIERARCHY_SUBMISSION_SPECS
        } | {
            (
                "random_forest",
                spec.random_forest_policy.family,
                spec.random_forest_policy.key,
            )
            for spec in PHYSICAL_HIERARCHY_SUBMISSION_SPECS
        }
        self.assertEqual(len(components), 4)

    def test_hybrid_crosses_waterpoint_xgboost_with_source_forest(self) -> None:
        hybrid = PHYSICAL_HIERARCHY_SUBMISSION_SPECS[-1]

        self.assertEqual(hybrid.xgboost_policy.family, "waterpoint")
        self.assertEqual(hybrid.xgboost_policy.key, "both_levels")
        self.assertEqual(hybrid.random_forest_policy.family, "source")
        self.assertEqual(hybrid.random_forest_policy.key, "source_plus_class")


if __name__ == "__main__":
    unittest.main()

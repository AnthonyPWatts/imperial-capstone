"""Tests for the fold-safe repair-rule ensemble."""

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

from repair_rule_ensemble import FULL_RULE_NAMES
from repair_rule_ensemble import HISTORY_RULE_NAMES
from repair_rule_ensemble import IDENTITY_CONFLICT_RULE
from repair_rule_ensemble import REPAIR_HISTORY_RULES
from repair_rule_ensemble import STRICT_HISTORY_RULE_NAMES
from repair_rule_ensemble import build_repair_rule_masks
from repair_rule_ensemble import cross_fit_repair_rule_masks
from repair_rule_ensemble import fit_repair_history_mapping
from repair_rule_ensemble import overlay_repair_rule_union
from repair_rule_ensemble import rule_overlap_counts
from repair_rule_ensemble import summarise_rule_contributions


class RepairRuleEnsembleTests(unittest.TestCase):
    def test_fixed_rule_contract_preserves_six_thresholds(self) -> None:
        actual = {
            rule.name: (
                rule.key,
                rule.minimum_repair_rate,
                rule.minimum_support,
                rule.minimum_repair_to_functional_ratio,
            )
            for rule in REPAIR_HISTORY_RULES
        }

        self.assertEqual(
            actual,
            {
                "scheme_name_history": ("scheme_name", 0.35, 50, 0.60),
                "grid_005_history": ("grid_005", 0.75, 3, 0.90),
                "lga_subvillage_history": (
                    "lga_subvillage",
                    1.00,
                    3,
                    0.50,
                ),
                "exact_coordinate_name_history": (
                    "exact_coordinate_name",
                    0.45,
                    1,
                    0.90,
                ),
                "subvillage_history": ("subvillage", 1.00, 3, 0.50),
            },
        )
        self.assertEqual(len(FULL_RULE_NAMES), 6)
        self.assertEqual(FULL_RULE_NAMES[-1], IDENTITY_CONFLICT_RULE)
        self.assertEqual(
            STRICT_HISTORY_RULE_NAMES,
            (
                "grid_005_history",
                "lga_subvillage_history",
                "exact_coordinate_name_history",
                "subvillage_history",
            ),
        )

    def test_history_mapping_ignores_non_functional_support(self) -> None:
        mapping = fit_repair_history_mapping(
            pd.Series(["a", "a", "a", "b"]),
            pd.Series(
                [
                    "functional needs repair",
                    "functional",
                    "non functional",
                    "non functional",
                ]
            ),
        )

        self.assertEqual(mapping.loc["a", "support"], 2)
        self.assertEqual(mapping.loc["a", "repair_rate"], 0.5)
        self.assertEqual(mapping.loc["b", "support"], 0)
        self.assertTrue(np.isnan(mapping.loc["b", "repair_rate"]))

    def test_rules_are_guarded_and_only_override_functional_rows(self) -> None:
        training = _frame(55)
        target = pd.Series(
            ["functional needs repair"] * 20
            + ["functional"] * 35
        )
        prediction = _frame(3)
        base = np.asarray(
            [
                [0.50, 0.35, 0.15],
                [0.75, 0.20, 0.05],
                [0.20, 0.10, 0.70],
            ]
        )
        identity = np.asarray(
            [
                [0.20, 0.60, 0.20],
                [0.20, 0.60, 0.20],
                [0.20, 0.60, 0.20],
            ]
        )

        masks = build_repair_rule_masks(
            training,
            target,
            prediction,
            base,
            identity,
        )

        self.assertTrue(masks.loc[0, "scheme_name_history"])
        self.assertFalse(masks.loc[1, "scheme_name_history"])
        self.assertFalse(masks.loc[2, "scheme_name_history"])
        self.assertTrue(masks.loc[0, IDENTITY_CONFLICT_RULE])
        self.assertTrue(masks.loc[1, IDENTITY_CONFLICT_RULE])
        self.assertFalse(masks.loc[2, IDENTITY_CONFLICT_RULE])

    def test_cross_fit_does_not_use_the_validation_fold_history(self) -> None:
        frame = _frame(100)
        frame.loc[:49, "scheme_name"] = "fold-one-only"
        frame.loc[50:, "scheme_name"] = "fold-two-only"
        target = pd.Series(["functional needs repair"] * 100)
        folds = pd.Series([1] * 50 + [2] * 50)
        base = np.tile([0.50, 0.35, 0.15], (100, 1))
        identity = np.tile([0.80, 0.10, 0.10], (100, 1))

        masks = cross_fit_repair_rule_masks(
            frame,
            target,
            folds,
            base,
            identity,
        )

        self.assertFalse(masks["scheme_name_history"].any())

    def test_overlay_and_attribution_use_the_declared_union(self) -> None:
        probabilities = np.asarray(
            [
                [0.60, 0.30, 0.10],
                [0.55, 0.40, 0.05],
                [0.10, 0.20, 0.70],
            ]
        )
        masks = pd.DataFrame(False, index=range(3), columns=FULL_RULE_NAMES)
        masks.loc[0, HISTORY_RULE_NAMES[0]] = True
        masks.loc[0, IDENTITY_CONFLICT_RULE] = True
        masks.loc[1, IDENTITY_CONFLICT_RULE] = True

        overlaid, selected = overlay_repair_rule_union(probabilities, masks)
        contributions = summarise_rule_contributions(
            masks,
            target=pd.Series(
                [
                    "functional needs repair",
                    "functional",
                    "non functional",
                ]
            ),
        )
        overlaps = rule_overlap_counts(masks)

        np.testing.assert_array_equal(selected, [True, True, False])
        np.testing.assert_allclose(overlaid[0], [0.30, 0.60, 0.10])
        np.testing.assert_allclose(overlaid[1], [0.40, 0.55, 0.05])
        self.assertEqual(
            contributions.loc[IDENTITY_CONFLICT_RULE, "standalone_net_correct"],
            0,
        )
        self.assertEqual(
            overlaps.loc[HISTORY_RULE_NAMES[0], IDENTITY_CONFLICT_RULE],
            1,
        )

    def test_overlay_combines_external_mask_on_functional_rows_only(self) -> None:
        probabilities = np.asarray(
            [
                [0.60, 0.30, 0.10],
                [0.55, 0.40, 0.05],
                [0.10, 0.20, 0.70],
            ]
        )
        masks = pd.DataFrame(False, index=range(3), columns=FULL_RULE_NAMES)

        overlaid, selected = overlay_repair_rule_union(
            probabilities,
            masks,
            rule_names=STRICT_HISTORY_RULE_NAMES,
            additional_mask=np.asarray([False, True, True]),
        )

        np.testing.assert_array_equal(selected, [False, True, False])
        np.testing.assert_allclose(overlaid[1], [0.40, 0.55, 0.05])
        np.testing.assert_allclose(overlaid[2], probabilities[2])

    def test_overlay_rejects_misaligned_external_mask(self) -> None:
        probabilities = np.asarray([[0.60, 0.30, 0.10]])
        masks = pd.DataFrame(False, index=range(1), columns=FULL_RULE_NAMES)

        with self.assertRaisesRegex(ValueError, "wrong shape"):
            overlay_repair_rule_union(
                probabilities,
                masks,
                additional_mask=np.asarray([True, False]),
            )


def _frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "longitude": [35.1234567] * rows,
            "latitude": [-6.1234567] * rows,
            "scheme_name": ["scheme"] * rows,
            "lga": ["lga"] * rows,
            "subvillage": ["subvillage"] * rows,
            "wpt_name": ["name"] * rows,
        }
    )


if __name__ == "__main__":
    unittest.main()

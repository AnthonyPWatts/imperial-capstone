"""Tests for conservative residual repair decisions."""

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

from repair_residual_specialist import REPAIR_META_COMPONENTS
from repair_residual_specialist import build_repair_meta_features
from repair_residual_specialist import overlay_repair_probabilities
from repair_residual_specialist import repair_history_mask


class RepairResidualSpecialistTests(unittest.TestCase):
    def test_meta_features_have_fixed_shape(self) -> None:
        probabilities = np.asarray(
            [[0.8, 0.1, 0.1], [0.2, 0.6, 0.2]],
            dtype="float64",
        )
        components = {
            name: probabilities.copy() for name in REPAIR_META_COMPONENTS
        }

        features = build_repair_meta_features(components, probabilities)

        self.assertEqual(features.shape, (2, 26))
        self.assertTrue(np.isfinite(features).all())

    def test_meta_features_require_every_component(self) -> None:
        probabilities = np.asarray([[0.8, 0.1, 0.1]])
        components = {
            name: probabilities.copy()
            for name in REPAIR_META_COMPONENTS
            if name != "identity_catboost"
        }

        with self.assertRaisesRegex(KeyError, "identity_catboost"):
            build_repair_meta_features(components, probabilities)

    def test_overlay_only_changes_selected_functional_rows(self) -> None:
        base = np.asarray(
            [
                [0.60, 0.35, 0.05],
                [0.60, 0.35, 0.05],
                [0.20, 0.10, 0.70],
            ]
        )

        overlaid, selected = overlay_repair_probabilities(
            base,
            np.asarray([0.80, 0.60, 0.90]),
            threshold=0.70,
        )

        np.testing.assert_array_equal(selected, [True, False, False])
        np.testing.assert_array_equal(overlaid.argmax(axis=1), [1, 0, 2])
        np.testing.assert_allclose(overlaid.sum(axis=1), 1.0)

    def test_history_rule_uses_only_supported_operational_counts(self) -> None:
        X_training = pd.DataFrame(
            {
                "subvillage": ["A"] * 5 + ["B"] * 4 + ["C"] * 3,
            }
        )
        y_training = pd.Series(
            [
                "functional needs repair",
                "functional needs repair",
                "functional needs repair",
                "functional",
                "non functional",
                "functional needs repair",
                "functional needs repair",
                "functional",
                "functional",
                "functional needs repair",
                "non functional",
                "non functional",
            ]
        )
        X_prediction = pd.DataFrame({"subvillage": [" a ", "B", "C", "D"]})

        selected = repair_history_mask(
            X_training,
            y_training,
            X_prediction,
            minimum_support=3,
            minimum_rate=0.60,
        )

        np.testing.assert_array_equal(selected, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()

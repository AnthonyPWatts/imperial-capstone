"""Tests for source-plus-class and complete-identity recipe blending."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from source_identity_confirmation import blend_source_identity_recipes


class SourceIdentityConfirmationTests(unittest.TestCase):
    def test_fixed_source_and_identity_recipes_preserve_declared_weights(self) -> None:
        xgboost = _probabilities(0.70, 0.20, 0.10)
        forest = _probabilities(0.50, 0.30, 0.20)
        identity = _probabilities(0.20, 0.30, 0.50)

        recipes = blend_source_identity_recipes(
            {
                "source_xgboost": xgboost,
                "source_random_forest": forest,
                "identity_catboost": identity,
            }
        )

        np.testing.assert_allclose(
            recipes["source_plus_class_55_45"],
            0.55 * xgboost + 0.45 * forest,
        )
        np.testing.assert_allclose(
            recipes["source_plus_class_identity_44_36_20"],
            0.44 * xgboost + 0.36 * forest + 0.20 * identity,
        )

    def test_missing_component_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "source_random_forest"):
            blend_source_identity_recipes(
                {
                    "source_xgboost": _probabilities(0.70, 0.20, 0.10),
                    "identity_catboost": _probabilities(0.20, 0.30, 0.50),
                }
            )


def _probabilities(functional: float, repair: float, non_functional: float):
    return np.asarray([[functional, repair, non_functional]])


if __name__ == "__main__":
    unittest.main()

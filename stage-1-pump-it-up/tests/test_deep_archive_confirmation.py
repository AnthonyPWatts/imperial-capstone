"""Tests for the fixed deep-XGBoost archive-substitution recipe."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from deep_archive_confirmation import blend_deep_archive


class DeepArchiveConfirmationTests(unittest.TestCase):
    def test_six_component_recipe_preserves_frozen_weights(self) -> None:
        components = {
            key: _probabilities(position / 10.0)
            for position, key in enumerate(DEEP_ARCHIVE_WEIGHTS, start=1)
        }

        actual = blend_deep_archive(components)
        expected = sum(
            DEEP_ARCHIVE_WEIGHTS[key] * components[key]
            for key in DEEP_ARCHIVE_WEIGHTS
        )

        np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(sum(DEEP_ARCHIVE_WEIGHTS.values()), 1.0)
        self.assertEqual(DEEP_ARCHIVE_WEIGHTS["deep_xgboost"], 0.33)

    def test_missing_component_is_rejected(self) -> None:
        components = {
            key: _probabilities(position / 10.0)
            for position, key in enumerate(DEEP_ARCHIVE_WEIGHTS, start=1)
            if key != "deep_xgboost"
        }

        with self.assertRaisesRegex(KeyError, "deep_xgboost"):
            blend_deep_archive(components)


def _probabilities(offset: float) -> np.ndarray:
    values = np.asarray([[0.50 + offset / 10, 0.30, 0.20 - offset / 10]])
    return values / values.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    unittest.main()

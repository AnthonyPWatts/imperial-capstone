"""Tests for the fixed deep residual-voter recipe."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_residual_confirmation import DEEP_RESIDUAL_WEIGHTS
from deep_residual_confirmation import blend_deep_residual_candidate


class DeepResidualConfirmationTests(unittest.TestCase):
    def test_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(DEEP_RESIDUAL_WEIGHTS.values()), 1.0)

    def test_blend_uses_fixed_weights(self) -> None:
        deep = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.2, 0.7]])
        location = np.asarray([[0.6, 0.2, 0.2], [0.2, 0.2, 0.6]])
        grid = np.asarray([[0.4, 0.3, 0.3], [0.3, 0.2, 0.5]])

        actual = blend_deep_residual_candidate(
            {
                "deep_archive": deep,
                "location_identity_blend": location,
                "grid_10km_blend": grid,
            }
        )

        expected = 0.925 * deep + 0.025 * location + 0.05 * grid
        np.testing.assert_allclose(actual, expected)

    def test_missing_component_is_rejected(self) -> None:
        probabilities = np.asarray([[0.8, 0.1, 0.1]])

        with self.assertRaisesRegex(KeyError, "grid_10km_blend"):
            blend_deep_residual_candidate(
                {
                    "deep_archive": probabilities,
                    "location_identity_blend": probabilities,
                }
            )


if __name__ == "__main__":
    unittest.main()

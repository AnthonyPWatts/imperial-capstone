"""Tests for cumulative XGBoost probability reconstruction."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ordinal_xgboost_evaluation import reconstruct_cumulative_memberships


class OrdinalXGBoostEvaluationTests(unittest.TestCase):
    def test_non_crossing_cumulative_probabilities_reconstruct_classes(self) -> None:
        memberships = reconstruct_cumulative_memberships(
            np.array([0.7]),
            np.array([0.3]),
        )

        np.testing.assert_allclose(memberships, [[0.3, 0.4, 0.3]])

    def test_crossing_probabilities_are_projected_to_their_midpoint(self) -> None:
        memberships = reconstruct_cumulative_memberships(
            np.array([0.2]),
            np.array([0.6]),
        )

        np.testing.assert_allclose(memberships, [[0.6, 0.0, 0.4]])

    def test_invalid_probability_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero to one"):
            reconstruct_cumulative_memberships(
                np.array([1.1]),
                np.array([0.2]),
            )


if __name__ == "__main__":
    unittest.main()

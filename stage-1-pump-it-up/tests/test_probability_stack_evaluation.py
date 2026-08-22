"""Tests for centred log-membership stack features."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from probability_stack_evaluation import centred_log_membership_features


class ProbabilityStackEvaluationTests(unittest.TestCase):
    def test_centred_log_features_are_finite_and_zero_sum_by_component(self) -> None:
        first = np.array([[0.7, 0.2, 0.1], [0.0, 0.4, 0.6]])
        second = np.array([[0.6, 0.1, 0.3], [0.2, 0.2, 0.6]])

        features = centred_log_membership_features([first, second])

        self.assertEqual(features.shape, (2, 6))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(
            features.reshape(2, 2, 3).sum(axis=2),
            0.0,
            atol=1e-12,
        )

    def test_misaligned_component_shape_is_rejected(self) -> None:
        first = np.array([[0.7, 0.2, 0.1]])
        second = np.array([[0.6, 0.4]])

        with self.assertRaisesRegex(ValueError, "shapes differ"):
            centred_log_membership_features([first, second])

    def test_non_probability_rows_are_rejected(self) -> None:
        invalid = np.array([[0.7, 0.2, 0.2]])

        with self.assertRaisesRegex(ValueError, "sum to one"):
            centred_log_membership_features([invalid, invalid])


if __name__ == "__main__":
    unittest.main()

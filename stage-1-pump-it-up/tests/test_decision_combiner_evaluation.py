"""Tests for hard-majority and soft-tie decision combination."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_combiner_evaluation import consensus_majority_probabilities


class DecisionCombinerEvaluationTests(unittest.TestCase):
    def test_majority_overrides_fallback_and_all_disagree_uses_it(self) -> None:
        components = (
            np.array([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]),
            np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]),
            np.array([[0.1, 0.7, 0.2], [0.1, 0.1, 0.8]]),
        )
        fallback = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2]])

        probabilities, all_disagree = consensus_majority_probabilities(
            components,
            fallback,
        )

        np.testing.assert_array_equal(probabilities.argmax(axis=1), [1, 1])
        np.testing.assert_array_equal(all_disagree, [False, True])
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_malformed_probabilities_are_rejected(self) -> None:
        valid = np.array([[0.8, 0.1, 0.1]])
        invalid = np.array([[0.8, 0.1, 0.2]])

        with self.assertRaisesRegex(ValueError, "sum to one"):
            consensus_majority_probabilities(
                (valid, valid, invalid),
                valid,
            )


if __name__ == "__main__":
    unittest.main()

"""Tests for independent one-vs-rest probability memberships."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from one_vs_rest_evaluation import normalise_ovr_memberships


class OneVsRestEvaluationTests(unittest.TestCase):
    def test_independent_probabilities_are_normalised_by_row(self) -> None:
        raw = np.array([[0.8, 0.1, 0.3], [0.2, 0.2, 0.2]])

        memberships = normalise_ovr_memberships(raw)

        np.testing.assert_allclose(memberships.sum(axis=1), 1.0)
        np.testing.assert_allclose(memberships[1], [1 / 3, 1 / 3, 1 / 3])
        self.assertEqual(memberships[0].argmax(), 0)

    def test_invalid_memberships_are_rejected(self) -> None:
        invalid = (
            np.zeros((1, 3)),
            np.array([[0.2, -0.1, 0.9]]),
            np.array([[0.2, np.nan, 0.8]]),
            np.ones((1, 2)),
        )

        for values in invalid:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    normalise_ovr_memberships(values)


if __name__ == "__main__":
    unittest.main()

"""Tests for fixed high-confidence transductive pseudo-labelling."""

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

from transductive_pseudo_labelling import append_pseudo_labelled_rows
from transductive_pseudo_labelling import select_high_confidence_pseudo_labels


class TransductivePseudoLabellingTests(unittest.TestCase):
    def test_selection_uses_only_fixed_high_confidence_rows(self) -> None:
        probabilities = np.asarray(
            [
                [0.990, 0.005, 0.005],
                [0.010, 0.980, 0.010],
                [0.020, 0.010, 0.970],
            ]
        )

        positions, labels, confidence = select_high_confidence_pseudo_labels(
            probabilities
        )

        np.testing.assert_array_equal(positions, [0, 1])
        np.testing.assert_array_equal(
            labels,
            ["functional", "functional needs repair"],
        )
        np.testing.assert_allclose(confidence, [0.99, 0.98])

    def test_append_retains_training_rows_and_selected_unlabelled_order(self) -> None:
        X_training = pd.DataFrame({"value": [1, 2]}, index=[10, 11])
        y_training = pd.Series(["functional", "non functional"], index=[10, 11])
        X_unlabelled = pd.DataFrame({"value": [3, 4, 5]}, index=[20, 21, 22])

        X_augmented, y_augmented = append_pseudo_labelled_rows(
            X_training,
            y_training,
            X_unlabelled,
            np.asarray([2, 0]),
            np.asarray(["functional", "non functional"]),
        )

        self.assertEqual(X_augmented["value"].tolist(), [1, 2, 5, 3])
        self.assertEqual(
            y_augmented.tolist(),
            ["functional", "non functional", "functional", "non functional"],
        )
        self.assertEqual(X_training.index.tolist(), [10, 11])
        self.assertEqual(X_unlabelled.index.tolist(), [20, 21, 22])


if __name__ == "__main__":
    unittest.main()

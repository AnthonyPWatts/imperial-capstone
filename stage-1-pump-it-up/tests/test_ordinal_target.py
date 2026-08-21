"""Focused tests for the cumulative-threshold ordinal helpers."""

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

from ordinal_target import CLASS_LABELS
from ordinal_target import cumulative_class_probabilities
from ordinal_target import cumulative_threshold_predictions
from ordinal_target import ordinal_error_metrics
from ordinal_target import project_cumulative_probabilities
from ordinal_target import tune_cumulative_cutoffs


class OrdinalTargetTests(unittest.TestCase):
    def test_crossing_cumulative_probabilities_are_projected_to_their_mean(self):
        projected = project_cumulative_probabilities(
            np.array([0.8, 0.3]),
            np.array([0.2, 0.7]),
        )

        np.testing.assert_allclose(projected, [[0.8, 0.2], [0.5, 0.5]])

    def test_class_probabilities_are_non_negative_and_sum_to_one(self):
        probabilities = cumulative_class_probabilities(
            np.array([0.8, 0.3]),
            np.array([0.2, 0.7]),
        )

        self.assertTrue((probabilities >= 0).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_two_cutoffs_can_select_each_ordered_class(self):
        predictions = cumulative_threshold_predictions(
            np.array([0.2, 0.7, 0.9]),
            np.array([0.1, 0.3, 0.8]),
        )

        self.assertEqual(predictions.tolist(), list(CLASS_LABELS))

    def test_cutoff_tuning_uses_accuracy_then_balanced_accuracy(self):
        target = pd.Series(list(CLASS_LABELS) * 2)
        repair_or_worse = np.array([0.1, 0.7, 0.9] * 2)
        non_functional = np.array([0.1, 0.2, 0.9] * 2)

        result = tune_cumulative_cutoffs(
            target,
            repair_or_worse,
            non_functional,
            candidate_cutoffs=[0.4, 0.5, 0.6],
        )

        self.assertEqual(result["calibration_accuracy"], 1.0)

    def test_ordinal_error_distinguishes_one_and_two_step_errors(self):
        result = ordinal_error_metrics(
            np.array(CLASS_LABELS, dtype=object),
            np.array([CLASS_LABELS[2], CLASS_LABELS[0], CLASS_LABELS[2]]),
        )

        self.assertAlmostEqual(result["mean_absolute_ordinal_error"], 1.0)
        self.assertAlmostEqual(result["two_step_error_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()

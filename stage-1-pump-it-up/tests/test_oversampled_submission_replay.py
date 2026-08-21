"""Focused tests for the frozen submission-replay helpers."""

from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from final_model import CLASS_LABELS
from oversampled_submission_replay import ComponentFit
from oversampled_submission_replay import blend_recipe_probabilities
from oversampled_submission_replay import score_recipe_probabilities
from oversampled_submission_replay import summarise_submission_csv
from oversampled_submission_replay import target_count_for_replica_fraction
from oversampled_submission_replay import target_count_for_multiplier


class OversampledSubmissionReplayTests(unittest.TestCase):
    def test_blend_uses_the_declared_weights(self) -> None:
        first = np.array([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
        second = np.array([[0.4, 0.4, 0.2], [0.6, 0.2, 0.2]])
        fits = {
            "first": ComponentFit(first, 1.0),
            "second": ComponentFit(second, 2.0),
        }

        actual = blend_recipe_probabilities(
            fits,
            (("first", 0.25), ("second", 0.75)),
        )

        np.testing.assert_allclose(actual, 0.25 * first + 0.75 * second)

    def test_blend_rejects_weights_that_do_not_sum_to_one(self) -> None:
        fits = {"only": ComponentFit(np.array([[0.7, 0.1, 0.2]]), 1.0)}

        with self.assertRaisesRegex(ValueError, "sum to one"):
            blend_recipe_probabilities(fits, (("only", 0.8),))

    def test_scoring_reports_class_recall_and_prediction_share(self) -> None:
        target = pd.Series(CLASS_LABELS, name="status_group")
        probabilities = np.array(
            [
                [0.9, 0.05, 0.05],
                [0.2, 0.3, 0.5],
                [0.1, 0.1, 0.8],
            ]
        )

        result = score_recipe_probabilities(target, probabilities)

        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["class_recall"][CLASS_LABELS[1]], 0.0)
        self.assertAlmostEqual(result["prediction_share"][CLASS_LABELS[2]], 2 / 3)

    def test_submission_summary_preserves_all_class_slots(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            pd.DataFrame(
                {
                    "id": [1, 2],
                    "status_group": [CLASS_LABELS[0], CLASS_LABELS[2]],
                }
            ).to_csv(path, index=False)

            result = summarise_submission_csv(path)

        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["prediction_share"][CLASS_LABELS[1]], 0.0)

    def test_replica_fraction_scales_only_the_added_rows(self) -> None:
        counts = pd.Series(
            {
                "functional": 32_259,
                "functional needs repair": 4_317,
                "non functional": 22_824,
            }
        )

        result = target_count_for_replica_fraction(
            counts,
            replica_fraction=0.8,
        )

        self.assertEqual(result, 19_123)

    def test_multiplier_scales_the_final_minority_count(self) -> None:
        counts = pd.Series(
            {
                "functional": 32_259,
                "functional needs repair": 4_317,
                "non functional": 22_824,
            }
        )

        result = target_count_for_multiplier(
            counts,
            target_multiplier=2.5,
        )

        self.assertEqual(result, 10_793)


if __name__ == "__main__":
    unittest.main()

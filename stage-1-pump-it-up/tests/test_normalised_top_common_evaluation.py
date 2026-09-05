"""Tests for the locked normalised top-common comparison gate."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from normalised_top_common_evaluation import summarise_normalised_comparison


class NormalisedTopCommonEvaluationTests(unittest.TestCase):
    def test_all_four_gate_requirements_pass_at_the_boundaries(self) -> None:
        raw = _evaluation([0.8] * 5, [0.30] * 5)
        normalised = _evaluation(
            [0.8025, 0.8025, 0.8025, 0.7975, 0.8],
            [0.28] * 5,
        )

        summary, folds = summarise_normalised_comparison(raw, normalised)

        candidate = summary.loc["normalised_top_50_identities"]
        self.assertTrue(candidate["passes_gate"])
        self.assertAlmostEqual(candidate["mean_accuracy_delta"], 0.001)
        self.assertEqual(candidate["fold_wins"], 3)
        self.assertAlmostEqual(candidate["worst_fold_delta"], -0.0025)
        self.assertAlmostEqual(candidate["repair_recall_delta"], -0.02)
        self.assertEqual(folds["normalised_wins"].tolist(), [True, True, True, False, False])

    def test_each_failed_gate_requirement_rejects_the_candidate(self) -> None:
        raw = _evaluation([0.8] * 5, [0.30] * 5)
        cases = {
            "mean": ([0.802, 0.802, 0.802, 0.7975, 0.8], [0.30] * 5),
            "wins": ([0.803, 0.803, 0.8, 0.8, 0.8], [0.30] * 5),
            "worst": ([0.804, 0.804, 0.804, 0.797, 0.799], [0.30] * 5),
            "repair": ([0.8025, 0.8025, 0.8025, 0.7975, 0.8], [0.279] * 5),
        }
        for name, (accuracy, repair) in cases.items():
            with self.subTest(name=name):
                summary, _ = summarise_normalised_comparison(
                    raw, _evaluation(accuracy, repair)
                )
                self.assertFalse(
                    summary.loc["normalised_top_50_identities", "passes_gate"]
                )

    def test_different_fold_design_is_rejected(self) -> None:
        raw = _evaluation([0.8] * 5, [0.3] * 5)
        normalised = _evaluation([0.8] * 5, [0.3] * 5, fingerprint="other")
        with self.assertRaisesRegex(ValueError, "different cross-validation"):
            summarise_normalised_comparison(raw, normalised)


def _evaluation(accuracy, repair, *, fingerprint="locked-folds"):
    return SimpleNamespace(
        cross_validation_fingerprint=fingerprint,
        fold_metrics=pd.DataFrame(
            {
                "accuracy": accuracy,
                "recall: functional needs repair": repair,
            },
            index=pd.Index(range(1, 6), name="validation_fold"),
        ),
    )


if __name__ == "__main__":
    unittest.main()

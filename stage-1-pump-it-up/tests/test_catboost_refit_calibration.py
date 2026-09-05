"""Tests for the locked CatBoost refit-iteration calibration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_refit_calibration import FOLD_WIN_GATE
from catboost_refit_calibration import MEAN_ACCURACY_GAIN_GATE
from catboost_refit_calibration import REPAIR_RECALL_DELTA_GATE
from catboost_refit_calibration import WORST_FOLD_DELTA_GATE
from catboost_refit_calibration import corrected_refit_iterations
from catboost_refit_calibration import make_refit_recipe
from catboost_refit_calibration import summarise_refit_calibration
from catboost_identity_evaluation import make_complete_identity_catboost_spec


class CatBoostRefitCalibrationTests(unittest.TestCase):
    def test_exact_iteration_correction_matches_locked_fold_plan(self) -> None:
        selected = [2580, 2807, 2311, 2663, 2905]

        corrected = [corrected_refit_iterations(value) for value in selected]

        self.assertEqual(corrected, [2867, 3119, 2568, 2959, 3228])

    def test_iteration_correction_rejects_invalid_counts(self) -> None:
        for value in (0, -1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    corrected_refit_iterations(value)
        for value in (True, 1.0, "10"):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    corrected_refit_iterations(value)

    def test_recipe_freezes_each_fixed_refit_without_early_stopping(self) -> None:
        selected = dict(enumerate([2580, 2807, 2311, 2663, 2905], start=1))

        recipe = make_refit_recipe(
            make_complete_identity_catboost_spec(),
            selected,
        )

        self.assertEqual(
            recipe["corrected_refit_iterations_by_fold"],
            [2867, 3119, 2568, 2959, 3228],
        )
        self.assertFalse(recipe["new_early_stopping"])
        for fold, entry in enumerate(
            recipe["model_parameters_by_fold"],
            start=1,
        ):
            parameters = entry["parameters"]
            self.assertEqual(entry["validation_fold"], fold)
            self.assertEqual(parameters["depth"], 8)
            self.assertEqual(parameters["random_seed"], 20260821)
            self.assertEqual(
                parameters["iterations"],
                recipe["corrected_refit_iterations_by_fold"][fold - 1],
            )
            self.assertNotIn("od_type", parameters)
            self.assertNotIn("od_wait", parameters)

    def test_all_four_gate_requirements_pass_at_boundaries(self) -> None:
        baseline = _evaluation([0.8] * 5, [0.30] * 5)
        corrected = _evaluation(
            [0.8025, 0.8025, 0.8025, 0.7975, 0.8],
            [0.28] * 5,
        )

        summary, folds = summarise_refit_calibration(baseline, corrected)

        candidate = summary.loc["corrected_complete_identity_catboost"]
        self.assertAlmostEqual(
            candidate["mean_accuracy_delta"],
            MEAN_ACCURACY_GAIN_GATE,
        )
        self.assertEqual(candidate["fold_wins"], FOLD_WIN_GATE)
        self.assertAlmostEqual(
            candidate["worst_fold_delta"],
            WORST_FOLD_DELTA_GATE,
        )
        self.assertAlmostEqual(
            candidate["repair_recall_delta"],
            REPAIR_RECALL_DELTA_GATE,
        )
        self.assertTrue(candidate["passes_gate"])
        self.assertEqual(
            folds["corrected_wins"].tolist(),
            [True, True, True, False, False],
        )

    def test_each_failed_gate_requirement_rejects_candidate(self) -> None:
        baseline = _evaluation([0.8] * 5, [0.30] * 5)
        failures = {
            "mean gain": ([0.802, 0.802, 0.802, 0.7975, 0.8], [0.30] * 5),
            "fold wins": ([0.803, 0.803, 0.8, 0.8, 0.8], [0.30] * 5),
            "worst fold": ([0.804, 0.804, 0.804, 0.797, 0.799], [0.30] * 5),
            "repair recall": (
                [0.8025, 0.8025, 0.8025, 0.7975, 0.8],
                [0.279] * 5,
            ),
        }

        for name, (accuracies, repair_recalls) in failures.items():
            with self.subTest(name=name):
                summary, _ = summarise_refit_calibration(
                    baseline,
                    _evaluation(accuracies, repair_recalls),
                )
                self.assertFalse(
                    summary.loc[
                        "corrected_complete_identity_catboost",
                        "passes_gate",
                    ]
                )

    def test_different_fold_design_is_rejected(self) -> None:
        baseline = _evaluation([0.8] * 5, [0.30] * 5)
        corrected = _evaluation(
            [0.8] * 5,
            [0.30] * 5,
            fingerprint="different",
        )

        with self.assertRaisesRegex(ValueError, "different cross-validation"):
            summarise_refit_calibration(baseline, corrected)


def _evaluation(
    accuracies: list[float],
    repair_recalls: list[float],
    *,
    fingerprint: str = "fresh-full-data-folds",
):
    fold_metrics = pd.DataFrame(
        {
            "accuracy": accuracies,
            "recall: functional needs repair": repair_recalls,
        },
        index=pd.Index(range(1, 6), name="validation_fold"),
    )
    return SimpleNamespace(
        cross_validation_fingerprint=fingerprint,
        fold_metrics=fold_metrics,
    )


if __name__ == "__main__":
    unittest.main()

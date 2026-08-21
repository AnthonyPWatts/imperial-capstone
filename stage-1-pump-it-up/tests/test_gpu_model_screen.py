"""Focused tests for the bounded model-family screen plumbing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gpu_model_evaluation import FROZEN_FEATURE_POLICY
from gpu_model_evaluation import make_catboost_spec
from gpu_model_evaluation import make_lightgbm_spec
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import median_selected_iterations
from gpu_model_evaluation import with_seed
from model_screen_submission import select_submission_candidate_names


class GpuModelScreenTests(unittest.TestCase):
    def test_all_families_retain_the_frozen_feature_policy(self) -> None:
        specs = [
            make_catboost_spec(variant="d8"),
            make_xgboost_spec(variant="depth 8"),
            make_lightgbm_spec(variant="leaves 63"),
        ]

        self.assertTrue(
            all(spec.feature_policy == FROZEN_FEATURE_POLICY for spec in specs)
        )

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown XGBoost variant"):
            make_xgboost_spec(variant="invented")

    def test_seed_override_is_explicit_in_the_candidate_name(self) -> None:
        original = make_xgboost_spec(variant="depth 8")

        changed = with_seed(original, seed=17)

        self.assertEqual(changed.seed, 17)
        self.assertIn("seed 17", changed.name)
        self.assertEqual(changed.variant, original.variant)

    def test_full_refit_uses_the_median_fold_tree_count(self) -> None:
        evaluation = SimpleNamespace(
            diagnostics=pd.DataFrame(
                {"selected_iterations": [100, 500, 300, 200, 400]}
            )
        )

        self.assertEqual(median_selected_iterations(evaluation), 300)

    def test_second_submission_must_disagree_with_the_first(self) -> None:
        first = self._evaluation([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        near_duplicate = self._evaluation([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        different = self._evaluation([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        screen = {
            "summary": pd.DataFrame(
                {"passes_gate": [True, True, True]},
                index=["first", "near duplicate", "different"],
            ),
            "evaluations": {
                "first": first,
                "near duplicate": near_duplicate,
                "different": different,
            },
        }

        selected = select_submission_candidate_names(
            screen,
            minimum_second_disagreement=0.10,
        )

        self.assertEqual(selected, ["first", "different"])

    def test_no_submission_is_selected_when_nothing_passes(self) -> None:
        screen = {
            "summary": pd.DataFrame(
                {"passes_gate": [False]},
                index=["failed"],
            ),
            "evaluations": {"failed": self._evaluation([0, 0])},
        }

        self.assertEqual(select_submission_candidate_names(screen), [])

    @staticmethod
    def _evaluation(predictions: list[int]) -> SimpleNamespace:
        probabilities = np.full((len(predictions), 3), 0.05)
        probabilities[np.arange(len(predictions)), predictions] = 0.90
        return SimpleNamespace(
            out_of_fold_probabilities=pd.DataFrame(probabilities)
        )


if __name__ == "__main__":
    unittest.main()

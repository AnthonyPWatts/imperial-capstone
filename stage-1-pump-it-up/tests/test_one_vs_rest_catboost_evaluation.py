"""Tests for complete-identity one-vs-rest CatBoost."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from one_vs_rest_catboost_evaluation import make_ovr_identity_catboost_model


class OneVsRestCatBoostEvaluationTests(unittest.TestCase):
    def test_boundary_uses_fixed_binary_depth8_policy(self) -> None:
        model = make_ovr_identity_catboost_model(
            iterations=321,
            early_stopping=True,
        )
        parameters = model.get_params()

        self.assertEqual(parameters["loss_function"], "Logloss")
        self.assertEqual(parameters["iterations"], 321)
        self.assertEqual(parameters["depth"], 8)
        self.assertIn("od_wait", parameters)

    def test_non_positive_iterations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            make_ovr_identity_catboost_model(
                iterations=0,
                early_stopping=False,
            )


if __name__ == "__main__":
    unittest.main()

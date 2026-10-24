"""Checks for Function 1's shared target transform."""

from pathlib import Path
import sys
import unittest
from copy import deepcopy

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" /
                       "Function_1" / "FunctionSpecificShared"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "Shared"))
from function_1_model_helpers import (
    signed_log, run_target_reviews, summarise_target_reviews,
    validate_saved_target_review, compare_cohort_targets,
)
from model_helpers import make_models


class Function1ModelTests(unittest.TestCase):
    def fixture(self):
        inputs = np.random.default_rng(3).random((8, 5))
        raw = np.array([-1e-6, -1e-20, -1e-100, -1e-170, 1e-13, 1e-50, 1e-90, 1e-190])
        models = make_models()
        settings = dict(input_columns=[f"control_{i}" for i in range(5)], height_scale=1e-200,
                        gp_names=["GP: local (l=0.15)", "GP: broad (l=0.35)"],
                        candidate_settings=dict(power=3, seed=42, precision=6), selection_metric="LOO MAE")
        return inputs, raw, models, settings

    def test_both_targets_are_refitted_in_five_dimensions_without_changing_the_plan_or_data(self):
        inputs, raw, models, settings = self.fixture()
        before = raw.copy()
        planned = np.full(5, .99)
        study = run_target_reviews(models, inputs, raw, planned, **settings)
        np.testing.assert_array_equal(raw, before)
        np.testing.assert_array_equal(study["targets"]["magnitude"], np.abs(study["targets"]["signed"]))
        gp = settings["gp_names"][1]
        self.assertFalse(np.allclose(study["runs"]["magnitude"]["predictions"][gp],
                                     np.abs(study["runs"]["signed"]["predictions"][gp])))
        for name, run in study["runs"].items():
            errors = run["table"].set_index("Model").loc[settings["gp_names"], "LOO MAE"]
            self.assertEqual(run["selected_name"], errors.idxmin())
            np.testing.assert_array_equal(study["planned_predictions"][name]["point"], planned)
            self.assertEqual(len(study["gp_sensitivity"][name]), 8)
        self.assertFalse(hasattr(models[gp], "X_train_"))

    def test_multi_target_comparison_scores_the_same_targets_and_common_candidates(self):
        inputs, raw, models, settings = self.fixture()
        planned = np.full(5, .99)
        own = run_target_reviews(models, inputs, raw, planned, **settings)
        pooled_inputs = np.vstack([inputs, np.full(5, .44)])
        pooled = run_target_reviews(models, pooled_inputs, np.r_[raw, -1e-8], planned, **settings)
        comparisons = compare_cohort_targets(own, pooled, pooled_inputs, own_positions=np.arange(8), peer_count=1,
            planned_point=planned, input_columns=settings["input_columns"])
        for target_name, comparison in comparisons.items():
            self.assertEqual(len(comparison["paired_targets"]), 5)
            self.assertEqual(len(comparison["proposal_predictions"]), 6)
            self.assertEqual(comparison["common_candidates"].Candidates.nunique(), 1)
            self.assertEqual(comparison["proposal_predictions"].iloc[0].Point, "Saved coverage query")
            for label in settings["input_columns"]:
                self.assertIn(label, comparison["common_candidates"])

    def test_saved_evidence_rejects_changed_targets_or_an_actual_plan_change(self):
        inputs, raw, models, settings = self.fixture()
        study = run_target_reviews(models, inputs, raw, np.full(5, .99), **settings)
        summaries = summarise_target_reviews(study, target_labels={"signed": "h", "magnitude": "g"})
        saved = dict(observations=8, planned_portal_input="saved", target_reviews=deepcopy(summaries))
        validate_saved_target_review(saved, summaries, observation_count=8, planned_portal_input="saved")
        with self.assertRaisesRegex(ValueError, "coverage plan"):
            validate_saved_target_review(saved, summaries, observation_count=8, planned_portal_input="changed")
        saved["target_reviews"]["signed"]["selected_mae"] += 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_saved_target_review(saved, summaries, observation_count=8, planned_portal_input="saved")

    def test_fixed_model_settings_are_configurable_and_defaults_are_unchanged(self):
        defaults = make_models()
        self.assertEqual(defaults["GP: local (l=0.15)"].kernel.length_scale, .15)
        self.assertIsNone(defaults["GP: broad (l=0.35)"].optimizer)
        configured = make_models(ridge_alpha=3, noise_fraction=.2, gp_length_scales={"GP declared": .4})
        self.assertEqual(configured["Linear ridge"].steps[-1][1].alpha, 3)
        self.assertAlmostEqual(configured["GP declared"].alpha, .04)
        self.assertEqual(configured["GP declared"].kernel.length_scale, .4)

    def test_signed_log_keeps_zero_sign_order_and_does_not_change_values(self):
        values = np.array([-0.01, -1e-100, -1e-200, 0, 1e-200, 1e-100, 1e-15])
        original = values.copy()
        heights = signed_log(values)
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(np.sign(heights), np.sign(values))
        self.assertTrue(np.all(np.diff(heights) > 0))
        self.assertEqual(heights[3], 0)
        self.assertAlmostEqual(heights[4], np.log10(2))
        for scale in (0, -1, np.inf):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                signed_log(values, scale)



if __name__ == "__main__":
    unittest.main()

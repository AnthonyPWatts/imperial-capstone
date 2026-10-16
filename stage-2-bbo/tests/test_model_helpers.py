"""Checks for shared validation, slice and hypothetical-query semantics."""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "Shared"))
from model_helpers import (compare_models, gp_slice, make_gp,
                           make_query_candidates, score_query_candidates)
from two_input_model_helpers import prediction_grid


class SharedModelTests(unittest.TestCase):
    def test_held_out_mean_cannot_see_the_result_it_predicts(self):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        target = np.array([0., 0., 0., 100.])
        estimator = DummyRegressor(strategy="mean")
        table, predictions, fitted = compare_models({"Mean baseline": estimator}, inputs, target)
        np.testing.assert_allclose(predictions["Mean baseline"], (target.sum() - target) / 3)
        self.assertEqual(predictions["Mean baseline"][-1], 0)
        self.assertEqual(fitted["Mean baseline"].predict([[0.5, 0.5]])[0], 25)
        self.assertEqual(table.loc["Mean baseline", "MAE / baseline"], 1)
        self.assertFalse(hasattr(estimator, "constant_"))


    def test_modified_target_floor_is_used_on_held_out_predictions(self):
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        target = np.array([0., 2., 0., 1.])
        models = {"Mean baseline": DummyRegressor(),
                  "Negative predictor": DummyRegressor(strategy="constant", constant=-5)}
        _, original, _ = compare_models(models, inputs, target)
        _, floored, _ = compare_models(models, inputs, target, floor_zero=True)
        np.testing.assert_array_equal(original["Negative predictor"], -np.ones(4) * 5)
        np.testing.assert_array_equal(floored["Negative predictor"], np.zeros(4))
        np.testing.assert_array_equal(target, [0, 2, 0, 1])


    def test_slice_holds_one_coordinate_and_excludes_off_slice_observations(self):
        class CoordinateResponse:
            def predict(self, points, return_std=False):
                mean = points[:, 0] + 10 * points[:, 1]
                return (mean, np.ones(len(points))) if return_std else mean

        inputs = np.array([[0.2, 0.3], [0.2, 0.8], [0.7, 0.3]])
        target = np.array([3.2, 8.2, 3.7])
        horizontal = gp_slice(CoordinateResponse(), inputs, target, inputs[0], 0, side=11)
        np.testing.assert_allclose(horizontal["points"][:, 1], 0.3)
        np.testing.assert_allclose(horizontal["mean"], horizontal["coordinates"] + 3)
        np.testing.assert_array_equal(horizontal["observed_y"], [3.2, 3.7])
        vertical = gp_slice(CoordinateResponse(), inputs, target, inputs[0], 1, side=11)
        np.testing.assert_allclose(vertical["points"][:, 0], 0.2)
        np.testing.assert_array_equal(vertical["observed_y"], [3.2, 8.2])


    def test_gp_band_contracts_at_an_observation_under_fixed_assumptions(self):
        model = make_gp(0.15, 0.01).set_params(normalize_y=False)
        model.fit(np.array([[0.5]]), [0.4])
        _, std = model.predict([[0.5], [0]], return_std=True)
        self.assertLess(std[0], 0.02)
        self.assertGreater(std[1], 0.9)


    def test_grid_distinguishes_hull_support_and_keeps_uncertainty_separate(self):
        inputs = np.array([[0, 0], [0.8, 0], [0, 0.8]])
        model = make_gp().fit(inputs, [-2, -1, -3])
        grid = prediction_grid(model, inputs, side=11, uncertainty=True)
        self.assertTrue(grid["inside"][0, 0])
        self.assertFalse(grid["inside"][-1, -1])
        self.assertTrue(np.isfinite(grid["mean"]).all())
        self.assertTrue(np.all(grid["std"] >= 0))
        floored = prediction_grid(model, inputs, side=11, floor_zero=True)
        self.assertTrue(np.all(floored["mean"] >= 0))
        self.assertTrue(np.any(grid["mean"] < 0))


    def test_query_pool_is_repeatable_and_excludes_observations_at_submission_precision(self):
        pool = make_query_candidates(np.empty((0, 2)), power=3, seed=7)
        observed = np.vstack([pool[1] + 1e-8, [0.999999, 0.999999]])
        original = observed.copy()
        filtered = make_query_candidates(observed, power=3, seed=7)
        np.testing.assert_array_equal(filtered, make_query_candidates(observed, power=3, seed=7))
        np.testing.assert_array_equal(observed, original)
        self.assertFalse(np.any(np.all(filtered == pool[1], axis=1)))
        self.assertFalse(np.any(np.all(filtered == [0.999999, 0.999999], axis=1)))
        np.testing.assert_array_equal(filtered, filtered.round(6))
        self.assertTrue(((filtered >= 0) & (filtered <= 0.999999)).all())
        self.assertEqual(len(filtered), len(np.unique(filtered, axis=0)))


    def test_ucb_weight_can_choose_uncertainty_over_the_highest_mean(self):
        class MeanUncertaintyTradeoff:
            def predict(self, points, return_std=False):
                return points[:, 0], 1 - points[:, 0]

        points = np.array([[0.1, 0.4], [0.9, 0.4]])
        exploit = score_query_candidates(MeanUncertaintyTradeoff(), points, weight=0)
        explore = score_query_candidates(MeanUncertaintyTradeoff(), points, weight=2)
        self.assertEqual(exploit.iloc[0]["x1"], 0.9)
        self.assertEqual(explore.iloc[0]["x1"], 0.1)
        self.assertAlmostEqual(explore.iloc[0]["ucb"], 1.9)
        np.testing.assert_array_equal(points, [[0.1, 0.4], [0.9, 0.4]])


    def test_query_scores_use_the_returned_coordinates_and_have_deterministic_ties(self):
        class CoordinateResponse:
            def predict(self, points, return_std=False):
                return points[:, 0] + points[:, 1], np.zeros(len(points))

        points = np.array([[0.7, 0.3], [0.3, 0.7]])
        scored = score_query_candidates(CoordinateResponse(), points)
        self.assertEqual(scored.iloc[0]["x1"], 0.3)
        np.testing.assert_allclose(scored["posterior_mean"], scored["x1"] + scored["x2"])
        for invalid in (np.empty((0, 2)), np.array([[1, 0]]), np.array([[np.nan, 0]])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                score_query_candidates(CoordinateResponse(), invalid)
        with self.assertRaises(ValueError):
            score_query_candidates(CoordinateResponse(), points, weight=-1)



if __name__ == "__main__":
    unittest.main()

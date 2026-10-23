"""Behavioural checks for configurable reviews, candidate pools and proposals."""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "Shared"))
from model_helpers import make_gp, make_query_candidates, score_query_candidates
from model_review_helpers import (
    make_regression_models, fit_model_review, review_latest_predictions,
    build_review_candidates, score_review_candidates, resolve_unmodelled_coordinates,
    query_prediction, noise_sensitivity, weight_sensitivity,
)
from model_review_plots import plot_held_out_comparison, plot_latest_predictions


class CoordinateResponse:
    def predict(self, points, return_std=False):
        mean = np.asarray(points).sum(axis=1)
        return (mean, np.full(len(points), .2)) if return_std else mean


class RegressionReviewTests(unittest.TestCase):
    def test_optional_quadratic_pipeline_uses_all_five_inputs_and_scales_features(self):
        models = make_regression_models(5, polynomial_degree=2, polynomial_ridge_alpha=1e-10)
        inputs = np.random.default_rng(17).random((50, 5))
        target = inputs[:, 0] ** 2 + 3 * inputs[:, 1] * inputs[:, 4] - inputs[:, 3]
        model = models["Quadratic ridge"].fit(inputs, target)
        self.assertEqual(model.steps[0][1].n_output_features_, 20)
        self.assertEqual(model.steps[1][1].n_features_in_, 20)
        unseen = np.random.default_rng(18).random((5, 5))
        expected = unseen[:, 0] ** 2 + 3 * unseen[:, 1] * unseen[:, 4] - unseen[:, 3]
        np.testing.assert_allclose(model.predict(unseen), expected, atol=1e-8)
        self.assertNotIn("Quadratic ridge", make_regression_models(5))

    def test_projected_model_ignores_only_its_declared_axes_in_five_inputs(self):
        models = make_regression_models(5, feature_sets={"GP subset": (1, 3)}, optimiser_restarts=0)
        self.assertEqual(len(models["GP: separate length scales"].kernel.k2.length_scale), 5)
        inputs = np.random.default_rng(2).random((8, 5))
        target = inputs[:, 1] + inputs[:, 3]
        projected = models["GP subset"]
        projected.steps[-1][1].set_params(optimizer=None)
        projected.fit(inputs, target)
        points = np.array([[.1, .4, .2, .7, .3], [.9, .4, .8, .7, .6]])
        mean, std = projected.predict(points, return_std=True)
        self.assertAlmostEqual(mean[0], mean[1], places=13)
        self.assertAlmostEqual(std[0], std[1], places=13)
        self.assertEqual(projected.steps[-1][1].n_features_in_, 2)

    def test_gp_selection_stays_separate_from_the_best_regressor(self):
        inputs = np.random.default_rng(3).random((8, 2))
        target = 3 * inputs[:, 0] + inputs[:, 1]
        models = {"Mean baseline": DummyRegressor(), "Linear": LinearRegression(),
                  "GP broad": make_gp(.5), "GP local": make_gp(.03)}
        result = fit_model_review(models, inputs, target, gp_names=["GP broad", "GP local"])
        self.assertEqual(result["table"].iloc[0]["Model"], "Linear")
        gp_errors = result["table"].set_index("Model").loc[["GP broad", "GP local"], "LOO RMSE"]
        self.assertEqual(result["selected_name"], gp_errors.idxmin())
        self.assertFalse(hasattr(models["GP broad"], "kernel_"))

    def test_zero_latest_output_and_perfect_predictions_have_undefined_percentages(self):
        table = review_latest_predictions({"Perfect": np.array([.2, .1, 0.])},
                                          np.array([.2, .1, 0.]), np.array([False, False, True]))
        self.assertEqual(table.loc[0, "Latest error"], 0)
        self.assertTrue(np.isnan(table.loc[0, "Latest absolute error (%)"]))
        self.assertTrue(np.isnan(table.loc[0, "Latest share of squared error (%)"]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            review_latest_predictions({"A": [.2, .1]}, [.2, .1], np.array([True, True]))


class CandidateReviewTests(unittest.TestCase):
    def test_one_and_five_input_pools_remain_repeatable_and_score_custom_columns(self):
        for dimensions in (1, 5):
            with self.subTest(dimensions=dimensions):
                observed = np.full((1, dimensions), .5)
                before = observed.copy()
                pool = make_query_candidates(observed, power=3, seed=8, include_corners=False)
                np.testing.assert_array_equal(pool, make_query_candidates(observed, power=3, seed=8, include_corners=False))
                self.assertEqual(pool.shape, (8, dimensions))
                names = [f"control_{i}" for i in range(dimensions)]
                scores = score_query_candidates(CoordinateResponse(), pool, input_columns=names)
                np.testing.assert_allclose(scores.posterior_mean, scores[names].sum(axis=1))
                np.testing.assert_array_equal(observed, before)

    def test_combined_pool_excludes_rounded_observations_and_keeps_source_labels(self):
        inputs = np.array([[.1] * 5, [.8] * 5])
        columns = [f"control_{i}" for i in range(5)]
        coverage = np.array([[.1 + 1e-8] * 5, [.3333333] * 5])
        pool = build_review_candidates(inputs, inputs[1], coverage, input_columns=columns,
                                       broad_power=3, local_power=2)
        points = pool[columns].to_numpy()
        self.assertEqual(len(pool), len(np.unique(points, axis=0)))
        self.assertFalse(any(tuple(p) in set(map(tuple, inputs)) for p in points))
        np.testing.assert_array_equal(points, points.round(6))
        scored = score_review_candidates(CoordinateResponse(), pool, input_columns=columns)
        self.assertEqual(set(scored.candidate_source), set(pool.candidate_source))
        self.assertIn("Single coverage query", set(scored.candidate_source))

    def test_single_free_coordinate_matches_the_original_geometric_search(self):
        observed = np.array([[.7, .1], [.4, .5], [.7, .9]])
        point = np.array([.7, .4])
        actual = resolve_unmodelled_coordinates(point, observed, modelled_axes=(0,), grid_size=11)
        grid = np.column_stack([np.full(11, .7), np.linspace(0, .999999, 11).round(6)])
        grid = grid[[tuple(p) not in set(map(tuple, observed)) for p in grid]]
        expected = grid[cdist(grid, observed).min(axis=1).argmax()]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(point, [.7, .4])

    def test_multiple_unmodelled_axes_are_repeatable_and_leave_modelled_inputs_fixed(self):
        observed = np.random.default_rng(4).random((8, 5))
        point = np.full(5, .5)
        settings = dict(modelled_axes=(1, 3), grid_size=32, seed=8)
        first = resolve_unmodelled_coordinates(point, observed, **settings)
        np.testing.assert_array_equal(first, resolve_unmodelled_coordinates(point, observed, **settings))
        np.testing.assert_array_equal(first[[1, 3]], point[[1, 3]])
        np.testing.assert_array_equal(first, first.round(6))
        self.assertFalse(any(np.array_equal(first, row.round(6)) for row in observed))

    def test_proposal_reports_uncertainty_separately_and_checks_the_full_five_input_hull(self):
        inputs = np.vstack([np.zeros(5), np.eye(5) * .8])
        model = CoordinateResponse()
        inside = query_prediction(model, [.1] * 5, inputs, np.arange(6), weight=2)
        outside = query_prediction(model, [.5] * 5, inputs, np.arange(6), weight=2)
        self.assertTrue(inside["inside_sampled_hull"])
        self.assertFalse(outside["inside_sampled_hull"])
        self.assertAlmostEqual(inside["ucb"], inside["predicted_output"] + 2 * inside["posterior_std"])
        with self.assertRaisesRegex(ValueError, "new finite input"):
            query_prediction(model, inputs[0], inputs, np.arange(6))


class SensitivityAndPlotTests(unittest.TestCase):
    def test_noise_refits_keep_the_original_fitted_model_unchanged(self):
        inputs = np.random.default_rng(5).random((8, 5))
        target = inputs[:, 1] + inputs[:, 3]
        model = make_regression_models(5, feature_sets={"GP subset": (1, 3)})["GP subset"]
        model.steps[-1][1].set_params(optimizer=None)
        model.fit(inputs, target)
        before = model.predict(inputs)
        table = noise_sensitivity(model, inputs, target, np.full(5, .5),
                                  fractions=(.1, .3), fitted_noise_fraction=.1)
        np.testing.assert_array_equal(model.predict(inputs), before)
        self.assertAlmostEqual(model.steps[-1][1].alpha, .01)
        self.assertGreater(table.loc[1, "Posterior SD"], table.loc[0, "Posterior SD"])

    def test_weight_sensitivity_uses_the_same_five_input_pool(self):
        columns = [f"control_{i}" for i in range(5)]
        pool = pd.DataFrame([[.2] * 5, [.8] * 5], columns=columns)
        pool["posterior_mean"], pool["posterior_sd"] = [.9, .1], [0., .8]
        table = weight_sensitivity(pool, np.full((1, 5), .5), input_columns=columns,
                                   modelled_axes=range(5), weights=(0, 2))
        np.testing.assert_array_equal(table.loc[0, columns].to_numpy(), [.2] * 5)
        np.testing.assert_array_equal(table.loc[1, columns].to_numpy(), [.8] * 5)

    def test_validation_plots_support_one_model_and_constant_outputs(self):
        target = np.array([0., 0., 0.])
        mask = np.array([False, False, True])
        figure = plot_held_out_comparison({"A": target}, target, model_names=["A"],
                                          latest_mask=mask, title="Constant outputs")
        self.assertLess(figure.axes[0].get_xlim()[0], figure.axes[0].get_xlim()[1])
        table = review_latest_predictions({"A": target}, target, mask)
        figure = plot_latest_predictions(table, 0., n_training_rows=2, title="Latest")
        self.assertTrue(np.isfinite(figure.axes[0].get_xlim()).all())


if __name__ == "__main__":
    unittest.main()

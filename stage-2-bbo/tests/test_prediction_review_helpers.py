"""Guard target alignment, candidate constraints and exported scientific predictions."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/Shared"))
from model_helpers import make_gp
from neural_explorer import export_network, export_gp, network_prediction, gp_prediction
from prediction_review_helpers import paired_peer_review, choose_training_scope, make_candidate_review, neural_architecture_review, neural_gradient_review


class PredictionReviewTests(unittest.TestCase):
    def test_peer_review_holds_out_each_official_target_and_keeps_all_peers(self):
        x = np.array([[.1, .2], [.2, .8], [.8, .3], [.9, .9]])
        y = np.array([1., 4., 2., 8.]);peer_x = np.array([[.4, .5]]);peer_y = np.array([3.])
        model = LinearRegression()
        own_predictions = np.array([clone(model).fit(np.delete(x, i, 0), np.delete(y, i)).predict(x[i:i+1])[0]
                                    for i in range(len(y))])
        run = paired_peer_review({"Linear": model}, x, peer_x, y, peer_y,
                                 {"predictions": {"Linear": own_predictions}}, verbose=False)
        expected = np.array([clone(model).fit(np.vstack([np.delete(x, i, 0), peer_x]),
                         np.r_[np.delete(y, i), peer_y]).predict(x[i:i+1])[0] for i in range(len(y))])
        np.testing.assert_allclose(run["predictions"]["Linear"], expected)
        self.assertFalse(hasattr(model, "coef_"))

    def test_scope_selection_uses_the_declared_metric_and_retains_official_ties(self):
        official_model, pooled_model = object(), object()
        own = dict(table=pd.DataFrame([{"Model": "GP", "LOO RMSE": 1., "LOO MAE": .5}]), fitted={"GP": official_model})
        peers = dict(table=pd.DataFrame([{"Model": "GP", "Peer-assisted LOO RMSE": .9, "Peer-assisted LOO MAE": .7}]), fitted={"GP": pooled_model})
        self.assertIs(choose_training_scope(own, ["GP"], peer_run=peers)["model"], pooled_model)
        self.assertIs(choose_training_scope(own, ["GP"], peer_run=peers, metric="LOO MAE")["model"], official_model)
        peers["table"]["Peer-assisted LOO RMSE"] = 1.
        self.assertEqual(choose_training_scope(own, ["GP"], peer_run=peers)["scope"], "official")

    def test_candidates_exclude_measured_and_peer_inputs_at_portal_precision_and_obey_spacing(self):
        inputs = np.array([[0., 0.], [.5, .5], [.999999, .999999]])
        pool, coverage = make_candidate_review(inputs[:2], inputs[1], input_columns=["x1", "x2"],
            excluded_inputs=inputs, minimum_distance=.1,
            search_settings=dict(broad_power=4, local_power=3, precision=6))
        self.assertTrue((pool.nearest_observed_distance >= .1).all())
        self.assertEqual(tuple(coverage), tuple(pool.loc[pool.nearest_observed_distance.idxmax(), ["x1", "x2"]]))
        self.assertTrue(all(tuple(row) not in set(map(tuple, inputs)) for row in pool[["x1", "x2"]].to_numpy()))

    def test_neural_architecture_selection_averages_all_seeds_instead_of_taking_the_best_seed(self):
        rows = [{"Model": name, "LOO RMSE": error} for name, error in [("a1", .1), ("a2", 1.9), ("b1", .8), ("b2", .8)]]
        metadata = {name: {"architecture": name[0], "seed": int(name[1])} for name in ("a1", "a2", "b1", "b2")}
        run = dict(table=pd.DataFrame(rows), predictions={name: np.array([1., 2.]) for name in metadata},
                   fitted={name: object() for name in metadata})
        result = neural_architecture_review(run, metadata)
        self.assertEqual(result["architecture"], "b")
        self.assertEqual(result["names"], ["b1", "b2"])

    def test_exported_neural_forward_pass_includes_input_and_target_scaling(self):
        rng = np.random.default_rng(12);inputs = rng.random((12, 3))
        target = 100 + 70*inputs[:, 0] - 11*inputs[:, 2]
        model = TransformedTargetRegressor(regressor=Pipeline([
            ("scale", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(4,), activation="tanh",
             solver="lbfgs", max_iter=400, random_state=42))]), transformer=StandardScaler()).fit(inputs, target)
        points = np.array([[0., .8, .999999], [.3, .2, .7]])
        np.testing.assert_allclose(network_prediction(export_network(model), points), model.predict(points), rtol=1e-12)

    def test_exported_gp_mean_and_sd_match_fixed_and_projected_fitted_models(self):
        inputs = np.random.default_rng(1).random((8, 3));target = 30 + 5*inputs[:, 0] - inputs[:, 1]
        projected = make_pipeline(ColumnTransformer([("inputs", "passthrough", [0])]), make_gp())
        for estimator in (make_gp(), projected):
            with self.subTest(projected=hasattr(estimator, "steps")):
                model = clone(estimator).fit(inputs, target);points = np.array([[0., .7, .9], [.5, .5, .5]])
                expected = model.predict(points, return_std=True)
                actual = gp_prediction(export_gp(model, 3), points)
                np.testing.assert_allclose(actual[0], expected[0], rtol=1e-12)
                np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12)

    def test_relu_kink_is_labelled_without_claiming_its_anchor_gradient_was_validated(self):
        x = np.array([[0.], [.5], [1.]])
        model = TransformedTargetRegressor(regressor=Pipeline([
            ("scale", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(1,), activation="relu",
             solver="lbfgs", max_iter=400, random_state=42))]), transformer=StandardScaler()).fit(x, x.ravel())
        network = model.regressor_.named_steps["model"]
        network.coefs_[0][:] = 1.;network.intercepts_[0][:] = 0.;network.coefs_[1][:] = 1.
        result = neural_gradient_review({"names": ["ReLU"], "models": [model]}, np.array([.5])).iloc[0]
        self.assertFalse(result.anchor_check_passes)
        self.assertTrue(result.passes)
        self.assertIn("ReLU kink", result.check_location)
        self.assertNotEqual(result.check_point, [.5])


if __name__ == "__main__":
    unittest.main()

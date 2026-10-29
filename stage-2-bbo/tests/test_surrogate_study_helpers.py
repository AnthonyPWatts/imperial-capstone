"""Check leakage boundaries, backpropagation and bounded surrogate experiments."""

from pathlib import Path
import sys
import unittest
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/Shared"))
from surrogate_study_helpers import (bounded_gradient_path, common_study_candidates,
    evaluate_classifiers, finite_difference_gradient, network_input_gradient,
    summarise_network_seeds, svm_support_review, training_threshold)


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    fits = []

    def fit(self, X, y):
        type(self).fits.append((np.asarray(X).copy(), np.asarray(y).copy()))
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([.75, .25], (len(X), 1))


class GradientTests(unittest.TestCase):
    def setUp(self):
        self.inputs = np.random.default_rng(2).uniform(.1, .9, (24, 5))

    def test_regression_backprop_includes_both_scalers_and_two_hidden_layers(self):
        target = 3000 * (self.inputs[:, 0] - self.inputs[:, 2]**2) + 500
        model = TransformedTargetRegressor(
            regressor=Pipeline([("scale", StandardScaler()), ("model", MLPRegressor(
                hidden_layer_sizes=(5, 3), activation="tanh", solver="lbfgs", alpha=.1,
                max_iter=300, random_state=3))]), transformer=StandardScaler())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(self.inputs, target)
        points = self.inputs[:3]
        actual = network_input_gradient(model, points)
        expected = finite_difference_gradient(model.predict, points, step=1e-6)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-4)
        self.assertEqual(actual.shape, (3, 5))

    def test_classification_backprop_matches_probability_derivative_for_relu(self):
        labels = (self.inputs[:, 0] + self.inputs[:, 2] > 1).astype(int)
        model = Pipeline([("scale", StandardScaler()), ("model", MLPClassifier(
            hidden_layer_sizes=(5, 3), activation="relu", solver="lbfgs", alpha=.1,
            max_iter=200, random_state=4))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(self.inputs, labels)
        expected = finite_difference_gradient(lambda points: model.predict_proba(points)[:, 1], self.inputs[:3], step=1e-6)
        np.testing.assert_allclose(network_input_gradient(model, self.inputs[:3]), expected, rtol=2e-5, atol=1e-6)

    def test_finite_differences_stay_in_domain_at_both_boundaries(self):
        def predict(points):
            self.assertTrue(((points >= 0) & (points < 1)).all())
            return 2*points[:, 0]-3*points[:, 1]
        actual = finite_difference_gradient(predict, [[0, .999999]], step=1e-5)
        np.testing.assert_allclose(actual, [[2, -3]], atol=1e-9)

    def test_ascent_and_descent_are_monotone_and_bounded(self):
        model = TransformedTargetRegressor(regressor=Pipeline([
            ("scale", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(4,), activation="tanh",
                solver="lbfgs", max_iter=200, random_state=7))]), transformer=StandardScaler())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(self.inputs, self.inputs[:, 0]-self.inputs[:, 1])
        for direction in (-1, 1):
            path = bounded_gradient_path(model, np.full(5, .5), direction=direction, steps=5)
            self.assertTrue((direction*np.diff(path.prediction) > 0).all())
            self.assertTrue(((path.filter(regex=r"^x\d").to_numpy() >= 0) & (path.filter(regex=r"^x\d").to_numpy() < 1)).all())


class ComparisonTests(unittest.TestCase):
    def test_each_classification_fold_uses_only_its_training_threshold(self):
        inputs = np.arange(8)[:, None]/10
        target = np.array([0., 1., 2., 3., 4., 5., 6., 1000.])
        RecordingClassifier.fits = []
        run = evaluate_classifiers({"Recorded": RecordingClassifier()}, inputs, target, verbose=False)
        for held, (training, labels) in enumerate(RecordingClassifier.fits[:-1]):
            kept = np.arange(len(target)) != held
            threshold = np.quantile(target[kept], .75)
            np.testing.assert_array_equal(training, inputs[kept])
            np.testing.assert_array_equal(labels, target[kept] >= threshold)
            self.assertEqual(run["fold_thresholds"][held], threshold)
        self.assertLess(run["fold_thresholds"][-1], run["threshold"])

    def test_architecture_selection_uses_seed_average_and_fixed_representative(self):
        metadata = {f"{arch}-{seed}": {"architecture": arch, "seed": seed, "layers": [4], "activation": "tanh"}
                    for arch in ("stable", "lucky") for seed in (11, 42, 73)}
        scores = [2., 2., 2., .1, 4., 4.]
        table = pd.DataFrame({"Model": list(metadata), "LOO RMSE": scores})
        selected = summarise_network_seeds({"table": table}, metadata, task="regression")
        self.assertEqual(selected["architecture"], "stable")
        self.assertEqual(selected["representative"], "stable-42")

    def test_candidate_pool_is_reproducible_and_excludes_measured_inputs(self):
        inputs = np.random.default_rng(4).uniform(0, 1, (8, 5))
        first = common_study_candidates(inputs, inputs[2], broad_power=3, local_power=2)
        second = common_study_candidates(inputs, inputs[2], broad_power=3, local_power=2)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first.columns), 6)
        self.assertNotIn(tuple(inputs[2].round(6)), set(map(tuple, first.iloc[:, :5].to_numpy())))

    def test_actual_support_indices_keep_all_original_coordinates_and_ids(self):
        from sklearn.svm import SVC
        inputs = np.random.default_rng(9).uniform(0, 1, (12, 5))
        target = inputs[:, 0]+inputs[:, 2]
        labels = (target >= np.quantile(target, .75)).astype(int)
        model = Pipeline([("scale", StandardScaler()), ("model", SVC())]).fit(inputs, labels)
        columns = [f"x{i+1}" for i in range(5)]
        data = pd.DataFrame(inputs, columns=columns).assign(observation_id=[f"row-{i}" for i in range(12)], y=target)
        support, count = svm_support_review(model, data, input_columns=columns, labels=labels, threshold=.5)
        self.assertEqual(count, len(model.named_steps["model"].support_))
        self.assertEqual(support[columns].shape[1], 5)
        self.assertTrue(set(support.observation_id) <= set(data.iloc[model.named_steps["model"].support_].observation_id))

    def test_degenerate_threshold_is_rejected_instead_of_inventing_classes(self):
        with self.assertRaisesRegex(ValueError, "both training classes"):
            training_threshold(np.ones(8))


if __name__ == "__main__":
    unittest.main()

"""Check units, noise and the meaning of the two working explanations."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from function_2_selection import (
    fit_explanation, mix_scores, predict_explanation, standardised_disagreement,
)


class Function2SelectionTests(unittest.TestCase):
    def test_disagreement_is_invariant_to_output_units_and_zero_for_equal_means(self):
        means = np.array([[1., 2.], [3., 2.]])
        variances = np.array([[.5, .25], [.5, .75]])
        expected = np.array([4., 0.])
        np.testing.assert_allclose(standardised_disagreement(means, variances), expected)
        np.testing.assert_allclose(
            standardised_disagreement(10 * means + 17, 100 * variances), expected)
        np.testing.assert_allclose(standardised_disagreement(means, 2 * variances), expected / 2)

    def test_mixing_respects_endpoints_and_separate_score_units(self):
        coverage, disagreement = np.array([.1, .2]), np.array([4., 1.])
        np.testing.assert_allclose(mix_scores(coverage, disagreement, .2, 1), [.5, 1.])
        np.testing.assert_allclose(mix_scores(coverage, disagreement, .2, 0), [.8, .5])
        expected = mix_scores(coverage, disagreement, .2, .5)
        np.testing.assert_allclose(expected, [.65, .75])
        np.testing.assert_allclose(mix_scores(100 * coverage, disagreement, 20, .5), expected)
        np.testing.assert_allclose(mix_scores(coverage, np.zeros(2), .2, .5), [.25, .5])

    def test_x1_explanation_ignores_x2_and_predictions_preserve_output_units(self):
        inputs = np.array([[.05, .2], [.2, .8], [.4, .1], [.6, .9], [.8, .3], [.95, .7]])
        outputs = np.array([-.1, .02, .2, .55, .4, .1])
        points = np.array([[.7, .1], [.7, .9]])
        noise = .05
        fit = fit_explanation(inputs, outputs, 1, noise)
        mean, variance = predict_explanation(fit, points)
        self.assertAlmostEqual(mean[0], mean[1], places=13)
        self.assertAlmostEqual(variance[0], variance[1], places=13)
        self.assertTrue(np.all(variance >= noise ** 2))
        scaled = fit_explanation(inputs, 3 * outputs + 2, 1, 3 * noise)
        shifted_mean, shifted_variance = predict_explanation(scaled, points)
        np.testing.assert_allclose(shifted_mean, 3 * mean + 2, rtol=1e-6)
        np.testing.assert_allclose(shifted_variance, 9 * variance, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

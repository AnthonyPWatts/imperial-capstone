"""Independent checks of the finite-set decision-risk calculation."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from function_2_risk import knowledge_gradient, condition_on_observation


class Function2RiskTests(unittest.TestCase):
    def test_symmetric_lines_have_known_normal_expectation(self):
        self.assertAlmostEqual(knowledge_gradient([0, 0], [-2, 2]), 2 * np.sqrt(2 / np.pi), places=13)
        self.assertEqual(knowledge_gradient([0, 1, 2], [3, 3, 3]), 0)

    def test_upper_envelope_matches_independent_quadrature(self):
        means = np.array([.1, .7, -.2, .3])
        slopes = np.array([-.5, .1, .8, .1])
        crossings = [(means[j] - means[i]) / (slopes[i] - slopes[j])
                     for i in range(4) for j in range(i) if slopes[i] != slopes[j]]
        expected = quad(lambda z: np.max(means + slopes * z) * np.exp(-z*z/2) / np.sqrt(2*np.pi),
                        -10, 10, points=[z for z in crossings if -10 < z < 10], epsabs=1e-11)[0]
        self.assertAlmostEqual(knowledge_gradient(means, slopes), expected - means.max(), places=10)

    def test_gain_preserves_units_order_and_duplicate_alternatives(self):
        means, slopes = np.array([.2, .1, -.4]), np.array([.1, .5, -.2])
        gain = knowledge_gradient(means, slopes)
        self.assertAlmostEqual(knowledge_gradient(3*means+17, 3*slopes), 3*gain, places=12)
        self.assertAlmostEqual(knowledge_gradient(means[::-1], slopes[::-1]), gain, places=12)
        self.assertAlmostEqual(knowledge_gradient(np.r_[means, means], np.r_[slopes, slopes]), gain, places=12)
        self.assertLessEqual(knowledge_gradient(means, slopes / 2), gain)

    def test_noisy_gaussian_update_matches_hand_calculation(self):
        mean = np.array([0., 1.])
        covariance = np.array([[1., .5], [.5, 2.]])
        updated_mean, updated_covariance = condition_on_observation(mean, covariance, 0, 2., 1.)
        np.testing.assert_allclose(updated_mean, [1., 1.5])
        np.testing.assert_allclose(updated_covariance, [[.5, .25], [.25, 1.875]])
        np.testing.assert_allclose(mean, [0., 1.])
        np.testing.assert_allclose(covariance, [[1., .5], [.5, 2.]])


if __name__ == "__main__":
    unittest.main()

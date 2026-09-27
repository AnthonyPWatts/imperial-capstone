"""Independent checks of the geometric mean-distance objective."""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coverage_optimisation import MeanCoverage2D, _distance_integral, optimise_average_coverage, optimise_coverage_batch


class CoverageOptimisationTests(unittest.TestCase):
    def test_square_distance_integrals_match_known_values(self):
        square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        corner_mean = (np.sqrt(2) + np.arcsinh(1)) / 3
        self.assertAlmostEqual(_distance_integral(square, np.array([0., 0.])), corner_mean, places=13)
        self.assertAlmostEqual(_distance_integral(square, np.array([.5, .5])), corner_mean / 2, places=13)

    def test_site_outside_polygon_matches_independent_quadrature(self):
        square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        site = np.array([2., -.3])
        expected = quad(lambda y: quad(lambda x: np.hypot(x - site[0], y - site[1]),
                                      0, 1, epsabs=1e-11)[0], 0, 1, epsabs=1e-11)[0]
        self.assertAlmostEqual(_distance_integral(square, site), expected, places=12)

    def test_adding_points_matches_independent_grid_and_never_increases_distance(self):
        inputs = np.array([[.1, .2], [.7, .25], [.55, .85]])
        objective = MeanCoverage2D(inputs)
        axis = (np.arange(501) + .5) / 501
        first, second = np.meshgrid(axis, axis)
        grid = np.column_stack([first.ravel(), second.ravel()])
        for point in ([0., .999999], [.45, .45], inputs[0]):
            with self.subTest(point=point):
                expected = cKDTree(np.vstack([inputs, point])).query(grid)[0].mean()
                self.assertAlmostEqual(objective(point), expected, delta=3e-6)
                self.assertLessEqual(objective(point), objective.mean_before + 1e-12)
        self.assertAlmostEqual(objective(inputs[0]), objective.mean_before, places=13)
        self.assertAlmostEqual(MeanCoverage2D(np.vstack([inputs, inputs])).mean_before,
                               objective.mean_before, places=13)

    def test_symmetric_geometry_selects_centre(self):
        data = pd.DataFrame({"x1": [0., 0., 1., 1.], "x2": [0., 1., 0., 1.]})
        result = optimise_average_coverage(data)
        np.testing.assert_allclose(result["point"], [.5, .5], atol=2e-6)
        self.assertGreater(result["reduction"], 0)
        self.assertLess(result["rounding_penalty"], 1e-10)

    def test_three_dimensional_input_is_not_silently_projected(self):
        data = pd.DataFrame({"x1": [.1, .9], "x2": [.2, .8], "x3": [.3, .7]})
        with self.assertRaisesRegex(ValueError, "not a projection"):
            optimise_average_coverage(data)

    def test_batch_overlap_is_counted_once_and_matches_independent_grid(self):
        inputs = np.array([[.1, .2], [.7, .25], [.55, .85]])
        objective = MeanCoverage2D(inputs)
        points = np.array([[.4, .45], [.47, .48], [.8, .75]])
        axis = (np.arange(501) + .5) / 501
        first, second = np.meshgrid(axis, axis)
        grid = np.column_stack([first.ravel(), second.ravel()])
        expected = cKDTree(np.vstack([inputs, points])).query(grid)[0].mean()
        actual = objective.mean_after(points)
        self.assertAlmostEqual(actual, expected, delta=3e-6)
        self.assertAlmostEqual(actual, objective.mean_after(points[::-1]), places=13)
        self.assertAlmostEqual(actual, objective.mean_after(np.vstack([points, points])), places=13)
        self.assertLessEqual(actual, objective.mean_after(points[:2]))
        summed_gain = sum(objective.mean_before - objective(point) for point in points)
        self.assertLess(objective.mean_before - actual, summed_gain)
        self.assertAlmostEqual(actual, MeanCoverage2D(np.vstack([inputs, points])).mean_before, places=12)

    def test_empty_batch_does_not_change_coverage(self):
        objective = MeanCoverage2D([[.1, .2], [.7, .8]])
        self.assertEqual(objective.mean_after(np.empty((0, 2))), objective.mean_before)

    def test_batch_search_rejects_unsupported_size_and_input_dimension(self):
        data = pd.DataFrame({"x1": [.1, .9], "x2": [.2, .8]})
        for count in (0, 4):
            with self.assertRaisesRegex(ValueError, "one to three"):
                optimise_coverage_batch(data, count)
        with self.assertRaisesRegex(ValueError, "not a projection"):
            optimise_coverage_batch(data.assign(x3=[.3, .7]), 2)


if __name__ == "__main__":
    unittest.main()

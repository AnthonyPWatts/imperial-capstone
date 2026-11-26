import sys
import unittest
from pathlib import Path

import numpy as np

SHARED = Path(__file__).resolve().parents[1] / "notebooks/Shared"
sys.path.insert(0, str(SHARED))

from weekly_peak_review import _short_horizon, separable_quadratic


class WeeklyPeakReviewTests(unittest.TestCase):
    def test_quadratic_fits_starters_and_leaves_return_for_validation(self):
        starter = np.array([
            [.05, .1], [.15, .2], [.1, .3], [.25, .4],
            [.35, .25], [.4, .6], [.5, .1], [.65, .4],
        ])
        x = np.vstack([starter, [.8, .35], [.25, .7]])
        y = 10 - 2 * (x[:, 0] - .4)**2 - 3 * (x[:, 1] - .35)**2
        y[-1] -= .25
        result = separable_quadratic(x, y, np.arange(len(x)) >= len(starter))
        self.assertEqual(result["portal_input"], "0.400000-0.350000")
        self.assertAlmostEqual(result["predicted_maximum"], 10)
        self.assertAlmostEqual(result["later_max_abs_error"], .25)

    def test_horizon_keeps_spacing_after_first_query(self):
        points = np.array([[.1], [.15], [.2]])
        rows = _short_horizon(points, np.array([.6, .7, .8]), np.eye(3) * .02, .01, .75,
                              np.array([True, True, True]), np.array([True, True, False]), .08)
        by_input = rows.set_index("Input")
        midpoint = by_input.loc["0.150000"]
        endpoint = by_input.loc["0.100000"]
        self.assertEqual(midpoint["One-query EI"], midpoint["Two-query gain"])
        self.assertGreater(endpoint["Two-query gain"], endpoint["One-query EI"])

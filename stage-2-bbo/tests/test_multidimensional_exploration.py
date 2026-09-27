"""Checks that cube coverage does not silently collapse into a projection."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multidimensional_exploration import nearest_distances, slice_distances


class FullSpaceCoverageTests(unittest.TestCase):
    def test_hidden_coordinate_changes_nearest_observation(self):
        observations = np.array([[0.5, 0.5, 0.1], [0.6, 0.5, 0.9]])
        point = np.array([[0.5, 0.5, 0.9]])
        self.assertAlmostEqual(nearest_distances(observations, point)[0], 0.1)
        self.assertEqual(nearest_distances(observations[:, :2], point[:, :2])[0], 0)

    def test_slice_matches_hand_calculated_cube_distance(self):
        _, planes = slice_distances([[0, 0, 0]], levels=(0, 0.5, 1), resolution=1)
        np.testing.assert_allclose([plane.item() for plane in planes],
                                   np.sqrt([0.5, 0.75, 1.5]))


if __name__ == "__main__":
    unittest.main()

"""Checks for conditional sections and hypothetical queries in three inputs."""

from pathlib import Path
import sys
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "Shared"))
from model_helpers import gp_slice, make_gp, make_query_candidates, score_query_candidates
from three_input_model_helpers import (build_plane_slices, input_cloud,
                                        interactive_terrain_slices, plane_grid)


class CoordinateResponse:
    def predict(self, points, return_std=False):
        mean = points @ np.array([1, 10, 100])
        return (mean, np.full(len(points), 0.1)) if return_std else mean


class ThreeInputModelTests(unittest.TestCase):
    def setUp(self):
        self.inputs = np.array([[0., 0., 0.], [.8, 0., 0.], [0., .8, 0.], [0., 0., .8]])
        self.outputs = np.array([-.4, -.1, -.2, -.3])

    def test_curve_requires_both_other_coordinates_to_match(self):
        inputs = np.array([[.2, .3, .4], [.7, .3, .4], [.2, .3, .9], [.2, .8, .4]])
        outputs = np.array([1, 2, 3, 4])
        section = gp_slice(CoordinateResponse(), inputs, outputs, inputs[0], 0)
        np.testing.assert_allclose(section["points"][:, 1:], [[.3, .4]] * 250)
        np.testing.assert_array_equal(section["observed_y"], [1, 2])
        section = gp_slice(CoordinateResponse(), inputs, outputs, inputs[0], 2)
        np.testing.assert_allclose(section["points"][:, :2], [[.2, .3]] * 250)
        np.testing.assert_array_equal(section["observed_y"], [1, 3])

    def test_plane_retains_fixed_coordinate_and_joint_model_response(self):
        for axes, fixed in [((0, 1), 2), ((0, 2), 1), ((1, 2), 0)]:
            with self.subTest(axes=axes):
                grid = plane_grid(CoordinateResponse(), self.inputs, axes, .3, side=11, uncertainty=True)
                np.testing.assert_allclose(grid["points"][:, fixed], .3)
                np.testing.assert_allclose(grid["mean"].ravel(), grid["points"] @ [1, 10, 100])
                np.testing.assert_allclose(grid["std"], .1)
                self.assertEqual(grid["inside"].shape, (11, 11))

    def test_full_space_hull_rejects_a_point_that_looks_covered_in_projection(self):
        grid = plane_grid(CoordinateResponse(), self.inputs, (0, 1), .3, side=11)
        projected = Delaunay(self.inputs[:3, :2]).find_simplex([[.3, .3]]) >= 0
        self.assertTrue(projected[0])
        self.assertFalse(grid["inside"][3, 3])
        self.assertTrue(grid["inside"][1, 1])
        self.assertTrue(np.isfinite(grid["mean"]).all())

    def test_three_input_candidates_keep_all_corners_and_exclude_rounded_observations(self):
        initial = make_query_candidates(np.empty((0, 3)), power=3, seed=7)
        self.assertEqual(len(initial), 16)
        observed = initial[[0, 3, 8]] + 1e-8
        pool = make_query_candidates(observed, power=3, seed=7)
        self.assertEqual(pool.shape, (13, 3))
        self.assertFalse(set(map(tuple, pool)) & set(map(tuple, observed.round(6))))
        np.testing.assert_array_equal(pool, pool.round(6))
        np.testing.assert_array_equal(pool, make_query_candidates(observed, power=3, seed=7))
        scored = score_query_candidates(CoordinateResponse(), pool)
        self.assertIn("x3", scored.columns)
        np.testing.assert_allclose(scored["posterior_mean"], scored[["x1", "x2", "x3"]] @ [1, 10, 100])
        np.testing.assert_allclose(scored["ucb"], scored["posterior_mean"] + 1.96 * .1)

    def test_three_input_gp_returns_finite_plane_uncertainty(self):
        model = make_gp().fit(self.inputs, self.outputs)
        grid = plane_grid(model, self.inputs, (0, 2), .3, side=11, uncertainty=True)
        self.assertTrue(np.isfinite(grid["mean"]).all())
        self.assertTrue(np.isfinite(grid["std"]).all())
        self.assertTrue((grid["std"] >= 0).all())
        with self.assertRaises(ValueError):
            plane_grid(model, self.inputs, (0, 0), .3)

    def test_interactive_frames_use_small_meshes_and_correct_masked_sections(self):
        values = np.array([.1, .5, .9])
        slices = build_plane_slices(CoordinateResponse(), self.inputs, values)
        figure = interactive_terrain_slices(slices, values, self.outputs, title="Test sections")
        self.assertEqual(len(figure.frames), 3)
        self.assertEqual(len(figure.layout.sliders[0].steps), 3)
        self.assertEqual(np.asarray(figure.data[0].z).shape, (31, 31))
        for level, frame in enumerate(figure.frames):
            for pair, trace in enumerate(frame.data):
                grid = slices[level][pair]
                expected = np.where(grid["inside"], grid["mean"], np.nan)[::2, ::2]
                np.testing.assert_array_equal(trace.z, expected)
                self.assertIsNone(trace.x)
                self.assertIn(f"={values[level]:.2f}", trace.hovertemplate)
        self.assertEqual(figure.layout.scene.aspectratio.z, .16)

    def test_cloud_uses_three_input_axes_and_does_not_invent_outputs_for_queries(self):
        figure = input_cloud(self.inputs, self.outputs, title="Test cloud",
                             planned=[.3, .2, .1], suggestion=[.8, .7, .6])
        np.testing.assert_array_equal(figure.data[0].z, self.inputs[:, 2])
        np.testing.assert_array_equal(figure.data[0].marker.color, self.outputs)
        self.assertEqual(figure.data[1].z[0], .1)
        self.assertEqual(figure.data[2].z[0], .6)
        self.assertIn("Output not measured", figure.data[2].hovertemplate)


if __name__ == "__main__":
    unittest.main()

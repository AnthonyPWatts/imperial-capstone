"""Behavioural checks for exploration summaries, highlights and terrain controls."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

notebooks = Path(__file__).resolve().parents[1] / "notebooks"
sys.path.insert(0, str(notebooks / "Shared"))
sys.path.insert(0, str(notebooks / "Function_1" / "FunctionSpecificShared"))
from function_1_plots import plot_output_number_line
from output_helpers import summarise_output_observations
from plot_helpers import plot_fixed_exploration_progression, plot_observed_terrain, plot_sampling_gaps
from plot_helpers import plot_output_number_line as plot_linear_output_number_line
from plot_helpers import plot_pairwise_sampling_gaps, plot_sampling_gap_projections


class Function1NumberLineTests(unittest.TestCase):
    def review(self, values, rounds):
        count = len(values)
        data = pd.DataFrame({
            "observation_id": [f"row-{index}" for index in range(count)],
            "source_round": rounds,
            "x1": np.linspace(0.1, 0.9, count),
            "x2": np.linspace(0.9, 0.1, count),
            "y": values,
        })
        return summarise_output_observations(data, input_columns=["x1", "x2"])

    def test_tiny_positive_stays_positive_and_both_signs_share_one_style(self):
        review = self.review([-1e-3, -1e-80, 2e-200, 4e-20], [0, 0, 2, 1])
        before = review["data"].copy(deep=True)
        figure = plot_output_number_line(review, title="Function 1 - another week")
        negative, centre, positive = figure.axes
        self.assertTrue(negative.xaxis_inverted())
        self.assertFalse(positive.xaxis_inverted())
        np.testing.assert_allclose(negative.get_xlim(), positive.get_xlim()[::-1])
        np.testing.assert_array_equal(negative.collections[0].get_offsets()[:, 0], [1e-3, 1e-80])
        self.assertEqual(positive.collections[-1].get_offsets()[0, 0], 2e-200)
        np.testing.assert_array_equal(negative.collections[0].get_facecolors(),
                                      positive.collections[0].get_facecolors())
        np.testing.assert_array_equal(negative.collections[0].get_paths()[0].vertices,
                                      positive.collections[0].get_paths()[0].vertices)
        for axis in figure.axes:
            for collection in axis.collections:
                np.testing.assert_array_equal(collection.get_offsets()[:, 1], 0)
        self.assertEqual(len(figure.legends), 1)
        self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()],
                         ["Previous observations", "Latest submission (round 2)"])
        self.assertEqual(figure._suptitle.get_text(), "Function 1 - another week")
        self.assertEqual(len(centre.collections), 0)
        pd.testing.assert_frame_equal(review["data"], before)

    def test_actual_zeros_have_their_own_centre_marker_and_legend_entry(self):
        for values in ([-1e-3, 0, 2e-200], [0, 0, 0]):
            with self.subTest(values=values):
                review = self.review(values, [0, 0, 1])
                figure = plot_output_number_line(review)
                self.assertEqual(len(figure.legends), 1)
                labels = [text.get_text() for text in figure.legends[0].get_texts()]
                self.assertIn("Exact zero", labels)
                self.assertIn("Latest submission (round 1)", labels)
                for collection in figure.axes[1].collections:
                    np.testing.assert_array_equal(collection.get_offsets()[:, 0], 0.5)
                    np.testing.assert_array_equal(collection.get_offsets()[:, 1], 0)


class LinearNumberLineTests(unittest.TestCase):
    def test_zoom_preserves_observations_and_original_ranks(self):
        review = self.review([-.4, -.1, -.05], [0, 0, 4])
        before = review["data"].copy(deep=True)
        figure = plot_linear_output_number_line(review, output_limits=(-.15, 0))
        np.testing.assert_array_equal(figure.axes[0].get_xlim(), [-.15, 0])
        np.testing.assert_array_equal(figure.axes[0].collections[0].get_offsets()[:, 0], [-.4, -.1])
        self.assertIn("rank 1/3", "\n".join(text.get_text() for text in figure.texts))
        pd.testing.assert_frame_equal(review["data"], before)

    def review(self, values, rounds):
        data = pd.DataFrame({"observation_id": [f"row-{i}" for i in range(len(values))],
                             "source_round": rounds, "x1": .3, "x2": .7, "y": values},
                            index=np.arange(len(values)) * 10 + 5)
        return summarise_output_observations(data, input_columns=["x1", "x2"])

    def test_linear_positions_preserve_both_signs_and_highlight_latest_round(self):
        for rounds in ([0, 3, 1, 0, 2], [0, 3, 3, 0, 2]):
            with self.subTest(rounds=rounds):
                values = np.array([-.06, .15, .61, .24, .031])
                review = self.review(values, rounds)
                before = review["data"].copy(deep=True)
                mask_before = review["latest_mask"].copy()
                figure = plot_linear_output_number_line(review, title="Function 2 - Week 4")
                axis, = figure.axes
                self.assertEqual(axis.get_xscale(), "linear")
                self.assertFalse(axis.yaxis.get_visible())
                np.testing.assert_array_equal(axis.collections[0].get_offsets()[:, 0], values[~mask_before])
                np.testing.assert_array_equal(axis.collections[1].get_offsets()[:, 0], values[mask_before])
                for points in axis.collections:
                    np.testing.assert_array_equal(points.get_offsets()[:, 1], 0)
                self.assertLess(axis.get_xlim()[0], values.min())
                self.assertGreater(axis.get_xlim()[1], values.max())
                self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()],
                                 ["Previous observations", "Latest submission (round 3)"])
                notes = "\n".join(text.get_text() for text in figure.texts)
                self.assertIn("row-1: y = 0.15; rank 3/5", notes)
                if mask_before[2]:
                    self.assertIn("row-2: y = 0.61; rank 1/5", notes)
                pd.testing.assert_frame_equal(review["data"], before)
                np.testing.assert_array_equal(review["latest_mask"], mask_before)

    def test_exact_zeros_and_constant_ranges_remain_visible(self):
        for values in ([-.1, 0., .6], [0., 0., 0.], [2., 2., 2.]):
            with self.subTest(values=values):
                figure = plot_linear_output_number_line(self.review(values, [0, 0, 1]))
                axis, = figure.axes
                lower, upper = axis.get_xlim()
                self.assertTrue(np.isfinite([lower, upper]).all())
                self.assertLess(lower, min(values))
                self.assertGreater(upper, max(values))
                labels = [text.get_text() for text in figure.legends[0].get_texts()]
                self.assertEqual("Exact zero" in labels, 0 in values)
                if 0 in values:
                    self.assertIn(0, np.concatenate([points.get_offsets()[:, 0] for points in axis.collections]))

    def test_initial_observations_need_no_id_or_latest_submission(self):
        review = self.review([.2, .2, .2], [0, 0, 0])
        review["data"] = review["data"].drop(columns="observation_id")
        figure = plot_linear_output_number_line(review)
        self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()],
                         ["Previous observations"])
        self.assertEqual(len(figure.axes[0].collections[0].get_offsets()), 3)
        self.assertGreater(figure.axes[0].get_xlim()[0], 0)


class SamplingGapHighlightTests(unittest.TestCase):
    def setUp(self):
        # A non-default index makes the positional mask convention explicit.
        self.data = pd.DataFrame({"x1": [.1, .7, .8], "x2": [.1, .7, 0.],
                                  "rank": [2, 1, 3]}, index=[10, 20, 30])

    def test_configured_figure_size_keeps_the_observed_coordinates(self):
        figure = plot_sampling_gaps(self.data, figsize=(5.7, 4.575))
        np.testing.assert_allclose(figure.get_size_inches(), [5.7, 4.575])
        np.testing.assert_array_equal(figure.axes[0].collections[1].get_offsets(),
                                      self.data[["x1", "x2"]])

    def test_one_or_multiple_highlights_keep_ranks_and_coverage_unchanged(self):
        before = self.data.copy(deep=True)
        original = plot_sampling_gaps(self.data).axes[0]
        for selection in ([False, False, True], [True, False, True]):
            with self.subTest(selection=selection):
                axis = plot_sampling_gaps(self.data, highlight=selection,
                                          highlight_label="Latest submission").axes[0]
                np.testing.assert_array_equal(axis.collections[0].get_array(),
                                              original.collections[0].get_array())
                np.testing.assert_array_equal(axis.collections[1].get_offsets(),
                                              self.data.loc[np.logical_not(selection), ["x1", "x2"]])
                np.testing.assert_array_equal(axis.collections[2].get_offsets(),
                                              self.data.loc[selection, ["x1", "x2"]])
                self.assertFalse(axis.collections[2].get_clip_on())
                self.assertEqual([text.get_text() for text in axis.texts], ["2", "1", "3"])
                self.assertEqual(axis.get_legend().get_texts()[-1].get_text(), "Latest submission")
        pd.testing.assert_frame_equal(self.data, before)

    def test_highlights_respect_hidden_markers_and_labels(self):
        selected = [False, False, True]
        hidden = plot_sampling_gaps(self.data, highlight=selected, show_points=False).axes[0]
        self.assertEqual(len(hidden.collections), 1)
        self.assertEqual(len(hidden.texts), 0)
        self.assertIsNone(hidden.get_legend())
        unlabelled = plot_sampling_gaps(self.data.drop(columns="rank"), highlight=selected,
                                       show_labels=False).axes[0]
        self.assertEqual(len(unlabelled.collections), 3)
        self.assertEqual(len(unlabelled.texts), 0)
        self.assertEqual(unlabelled.get_legend().get_texts()[0].get_text(), "Highlighted observations")

    def test_missing_highlights_keep_original_circles_and_invalid_masks_are_rejected(self):
        for selection in (None, [False, False, False]):
            axis = plot_sampling_gaps(self.data, highlight=selection).axes[0]
            self.assertEqual(len(axis.collections), 2)
            np.testing.assert_array_equal(axis.collections[1].get_offsets(), self.data[["x1", "x2"]])
            self.assertEqual(len(axis.get_legend().get_texts()), 1)
        for selection in ([True], [0, 1, 0], [[True, False, False]]):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "boolean mask"):
                plot_sampling_gaps(self.data, highlight=selection)


class SamplingGapProjectionTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({"x1": [.1, .7, .8, .3], "x2": [.1, .7, 0., .9],
                                  "x3": [.8, .2, .4, .7], "rank": [2, 1, 4, 3]},
                                 index=[10, 20, 30, 40])

    def test_three_projections_share_one_scale_and_key_with_correct_distances(self):
        before = self.data.copy(deep=True)
        selected = np.array([False, False, False, True])
        figure = plot_sampling_gap_projections(self.data, input_columns=["x1", "x2", "x3"],
                                               highlight=selected, highlight_label="Latest submission")
        self.assertEqual(len(figure.axes), 4)  # Three maps and one colour bar.
        self.assertEqual(len(figure.legends), 1)
        self.assertEqual([text.get_text() for text in figure.legends[0].get_texts()],
                         ["Numbers: observed output rank (1 = highest output)", "Latest submission"])
        centres = (np.arange(201) + .5) / 201
        first, second = np.meshgrid(centres, centres)
        grid = np.column_stack([first.ravel(), second.ravel()])
        expected_max = 0.
        for axis, pair in zip(figure.axes[:3], (("x1", "x2"), ("x1", "x3"), ("x2", "x3"))):
            expected = cdist(grid, self.data[list(pair)]).min(axis=1).reshape(201, 201)
            expected_max = max(expected_max, expected.max())
            np.testing.assert_allclose(axis.collections[0].get_array(), expected, atol=1e-12)
            np.testing.assert_array_equal(axis.collections[2].get_offsets(), self.data.loc[selected, list(pair)])
            self.assertTrue(all(not points.get_clip_on() for points in axis.collections[1:]))
            self.assertEqual([text.get_text() for text in axis.texts], ["2", "1", "4", "3"])
            self.assertIsNone(axis.get_legend())
            self.assertEqual((axis.get_xlabel(), axis.get_ylabel()), pair)
        for axis in figure.axes[:3]:
            self.assertEqual(axis.collections[0].get_clim(), (0, expected_max))
        figure.canvas.draw()
        self.assertLess(np.ptp([axis.get_position().y0 for axis in figure.axes[:3]]), 1e-8)
        pd.testing.assert_frame_equal(self.data, before)

    def test_selected_single_pair_and_explicit_scale_are_supported(self):
        figure = plot_sampling_gap_projections(self.data, input_columns=["x1", "x2", "x3"],
                                               pairs=[("x3", "x1")], vmax=.75)
        self.assertEqual(len(figure.axes), 2)
        self.assertEqual((figure.axes[0].get_xlabel(), figure.axes[0].get_ylabel()), ("x3", "x1"))
        self.assertEqual(figure.axes[0].collections[0].get_clim(), (0, .75))
        self.assertEqual(len(figure.legends[0].get_texts()), 1)

    def test_renderer_can_omit_keys_and_existing_pairwise_api_keeps_separate_figures(self):
        single = plot_sampling_gaps(self.data, show_legend=False, show_colorbar=False)
        self.assertEqual(len(single.axes), 1)
        self.assertEqual(len(single.axes[0].texts), len(self.data))
        self.assertIsNone(single.axes[0].get_legend())
        separate = plot_pairwise_sampling_gaps(self.data, input_columns=["x1", "x2", "x3"])
        self.assertEqual(list(separate), [("x1", "x2"), ("x1", "x3"), ("x2", "x3")])
        self.assertTrue(all(len(figure.axes) == 2 for figure in separate.values()))


class ObservedTerrainViewTests(unittest.TestCase):
    def test_compact_export_and_reset_preserve_observations_and_initial_camera(self):
        inputs = np.array([[.1, .1], [.7, .2], [.3, .9], [.8, .8]])
        outputs = np.array([-2., 0., .4, 1.])
        initial_camera = dict(eye=dict(x=-1.05, y=-1.35, z=.9),
                              center=dict(x=0, y=.1, z=0), up=dict(x=0, y=0, z=1))
        before = json.dumps(initial_camera)
        default = plot_observed_terrain(inputs, outputs, title="Terrain")
        self.assertIsNone(default.layout.width)
        self.assertEqual(default.layout.height, 700)
        self.assertEqual(len(default.layout.updatemenus), 0)
        self.assertEqual(default.layout.scene.camera.eye.to_plotly_json(), dict(x=-1.4, y=-1.8, z=1.2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terrain.html"
            compact = plot_observed_terrain(inputs, outputs, title="Terrain", width=850, height=560,
                                            camera=initial_camera, show_reset=True, save_to=path)
            self.assertEqual(json.loads(compact.to_json())["data"], json.loads(default.to_json())["data"])
            self.assertEqual((compact.layout.width, compact.layout.height), (850, 560))
            self.assertFalse(compact.layout.autosize)
            button = compact.layout.updatemenus[0].buttons[0]
            self.assertEqual(button.label, "Reset view")
            self.assertEqual(button.method, "relayout")
            self.assertEqual(button.args[0]["scene.camera"], compact.layout.scene.camera.to_plotly_json())
            self.assertEqual(button.args[0]["scene.camera"]["center"], initial_camera["center"])
            html = path.read_text(encoding="utf-8")
            self.assertIn('"responsive": false', html)
            self.assertIn('width:850px', html)
            self.assertIn('"label":"Reset view"', html)
        self.assertEqual(json.dumps(initial_camera), before)


class FixedPlanProgressionTests(unittest.TestCase):
    def setUp(self):
        self.observations = pd.DataFrame({
            "x1": [0., .5, 0., .5, .2],
            "x2": [0., 0., .5, .5, .8],
            "source_round": [0, 0, 0, 1, 2],
        })
        # The recorded Week 2 input follows submission precision, rather than
        # exactly matching the extra digits in the saved planning coordinates.
        self.plan = pd.DataFrame({"x1": [.2000002, .8, .8], "x2": [.7999998, .2, .8]})

    def test_recorded_history_and_future_plan_share_comparable_distance_maps(self):
        observed_before = self.observations.copy(deep=True)
        plan_before = self.plan.copy(deep=True)
        figure = plot_fixed_exploration_progression(self.observations, self.plan, current_week=3)
        axes = figure.axes[:-1]
        self.assertEqual([len(axis.collections[1].get_offsets()) for axis in axes], [4, 5, 6, 7])
        self.assertEqual([i for i, axis in enumerate(axes) if "Current Position" in axis.get_title()], [1])
        np.testing.assert_array_equal(axes[1].collections[1].get_offsets(), self.observations[["x1", "x2"]])
        np.testing.assert_array_equal(axes[-1].collections[1].get_offsets()[-2:], self.plan.iloc[1:])
        self.assertTrue(all(len(axis.texts) == 0 for axis in axes))
        meshes = [axis.collections[0] for axis in axes]
        self.assertTrue(all(mesh.norm is meshes[0].norm for mesh in meshes))
        for before, after in zip(meshes, meshes[1:]):
            self.assertTrue((after.get_array() <= before.get_array() + 1e-12).all())
        # Check selected grid cells against an independent pairwise calculation.
        rows = np.array([0, 60, 100, 200])
        columns = np.array([200, 40, 100, 0])
        locations = np.column_stack([(columns + .5) / 201, (rows + .5) / 201])
        for axis in axes:
            expected = cdist(locations, axis.collections[1].get_offsets()).min(axis=1)
            np.testing.assert_allclose(axis.collections[0].get_array()[rows, columns], expected, atol=1e-12)
        pd.testing.assert_frame_equal(self.observations, observed_before)
        pd.testing.assert_frame_equal(self.plan, plan_before)

    def test_inconsistent_recorded_history_is_rejected(self):
        different = self.observations.copy(deep=True)
        different.loc[4, "x1"] = .4
        missing = self.observations.iloc[:-1].copy()
        future = self.observations.copy(deep=True)
        future.loc[4, "source_round"] = 3
        for name, observations, message in (
            ("different", different, "does not match"),
            ("missing", missing, "does not match"),
            ("future", future, "must precede"),
        ):
            with self.subTest(history=name), self.assertRaisesRegex(ValueError, message):
                plot_fixed_exploration_progression(observations, self.plan, current_week=3)

    def test_three_input_plan_is_rejected_instead_of_projected(self):
        with self.assertRaisesRegex(ValueError, "two-input"):
            plot_fixed_exploration_progression(self.observations.assign(x3=.3),
                                               self.plan.assign(x3=.7), current_week=3)


if __name__ == "__main__":
    unittest.main()

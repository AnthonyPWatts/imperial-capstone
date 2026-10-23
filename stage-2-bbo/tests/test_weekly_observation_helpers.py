"""Checks for measured review preparation, historical forecasts and arbitrary input counts."""

from pathlib import Path
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import nbformat
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "Shared"))
from output_helpers import summarise_output_observations
from observation_review_helpers import prepare_weekly_observation_review, read_saved_forecast, plot_output_review
from model_review_plots import plot_coordinate_model_review, plot_paired_cohort_profiles


class CoordinateResponse:
    def predict(self, inputs, return_std=False):
        mean = np.asarray(inputs).sum(axis=1)
        return (mean, np.full(len(mean), .2)) if return_std else mean


class WeeklyObservationTests(unittest.TestCase):
    def fixture(self):
        columns = [f"x{i+1}" for i in range(5)]
        own = pd.DataFrame([
            ["initial-001", 0, *([.1]*5), -5.],
            ["initial-002", 0, *([.9]*5), -4.],
            ["round-01", 1, *([.4]*5), -2.],
            ["round-02", 2, *([.45]*5), -3.],
        ], columns=["observation_id", "source_round", *columns, "y"])
        peer = pd.DataFrame([["peer-A-round-01", 1, *([.449]*5), 1.]], columns=own.columns)
        pooled = pd.concat([own, peer], ignore_index=True)
        review = summarise_output_observations(pooled, input_columns=columns)
        review["peer_mask"] = pooled.observation_id.str.startswith("peer-").to_numpy()
        return columns, own, review

    def test_pooled_ranks_and_incumbent_do_not_change_official_progress_or_previous_neighbour(self):
        columns, own, review = self.fixture()
        before = review["data"].copy(deep=True)
        prepared = prepare_weekly_observation_review(review, history=own, strongest_count=3)
        self.assertEqual(prepared["incumbent"].observation_id, "peer-A-round-01")
        self.assertEqual(prepared["strongest"].iloc[0]["rank"], 1)
        np.testing.assert_array_equal(prepared["progress"].best_y, [-4, -2, -2])
        self.assertEqual(prepared["neighbour_summary"]["nearest_previous_id"], "round-01")
        self.assertAlmostEqual(prepared["neighbour_summary"]["distance_to_previous_best"], np.sqrt(5)*.05)
        self.assertEqual(prepared["neighbour_summary"]["latest_minus_previous_best"], -1)
        self.assertEqual(list(prepared["coordinate_changes"].index), columns)
        pd.testing.assert_frame_equal(review["data"], before)

    def test_multiple_official_returns_in_one_round_are_rejected(self):
        _, own, review = self.fixture()
        duplicate = pd.concat([own, own.iloc[[2]].assign(observation_id="extra-return")], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "one return per round"):
            prepare_weekly_observation_review(review, history=duplicate)

    def test_peer_output_can_exceed_zero_and_has_an_unambiguous_label(self):
        _, _, review = self.fixture()
        figure = plot_output_review(review)
        self.assertIn("Retained peer returns", [text.get_text() for text in figure.legends[0].get_texts()])
        self.assertIn("peer-A", "\n".join(text.get_text() for text in figure.axes[1].texts))
        self.assertGreater(figure.axes[0].get_xlim()[1], 1)

    def test_historical_forecast_requires_the_actual_submitted_coordinates(self):
        text = "Coordinates: 0.123456, 0.234567, 0.345678, 0.456789\nPredicted output at the proposed query: -1.242050\n"
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            "", outputs=[nbformat.v4.new_output("stream", name="stdout", text=text)])])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"historical.ipynb"
            nbformat.write(notebook, path)
            forecast = read_saved_forecast(path, point=[.123456, .234567, .345678, .456789])
            self.assertEqual(forecast["predicted_output"], -1.242050)
            with self.assertRaisesRegex(ValueError, "actual submitted input"):
                read_saved_forecast(path, point=[.123456, .234567, .345678, .456790])

    def test_generic_conditional_curves_and_proposal_profiles_keep_all_inputs(self):
        for dimensions in (1, 5):
            with self.subTest(dimensions=dimensions):
                columns = [f"control_{i}" for i in range(dimensions)]
                inputs = np.array([[.2]*dimensions, [.7]*dimensions])
                anchor = inputs[0]
                target = inputs.sum(axis=1)
                curves = plot_coordinate_model_review({"GP": CoordinateResponse()}, inputs, target, anchor,
                    input_columns=columns, title="Conditional", side=11)
                self.assertEqual(len(curves.axes), dimensions)
                for axis, ax in enumerate(curves.axes):
                    line = ax.lines[0]
                    np.testing.assert_allclose(line.get_ydata(), line.get_xdata()+(dimensions-1)*.2)
                paired = pd.DataFrame({"Model": ["GP"], "Official RMSE": [.2], "Pooled RMSE": [.1]})
                profiles = plot_paired_cohort_profiles(paired,
                    {"Official": [.3]*dimensions, "Cohort": [.6]*dimensions},
                    input_columns=columns, target_count=2, title="Comparison")
                np.testing.assert_array_equal(profiles.axes[1].images[0].get_array(), [[.3]*dimensions, [.6]*dimensions])


if __name__ == "__main__":
    unittest.main()

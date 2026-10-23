"""Behavioural checks for pooled identities, paired targets and noise evidence."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "Shared"))
from cohort_review_helpers import (
    load_cohort_review, paired_target_comparison, common_candidate_comparison,
    compare_proposal_predictions, read_saved_proposal,
)
from model_review_helpers import make_regression_models
from model_review_plots import plot_nearby_differences
from noise_review_helpers import nearby_output_differences, fit_noise_diagnostics
from two_input_model_helpers import plot_cohort_comparison


class CoordinateResponse:
    def predict(self, inputs, return_std=False):
        mean = np.asarray(inputs).sum(axis=1)
        return (mean, np.full(len(mean), .2)) if return_std else mean


class CohortAlignmentTests(unittest.TestCase):
    def test_reordered_five_input_pool_aligns_by_id_and_ignores_peer_round_timeline(self):
        columns = [f"control_{i}" for i in range(5)]
        own = pd.DataFrame([["initial-001", 0, *([.2]*5), .3],
                            ["round-01", 1, *([.6]*5), .8]],
                           columns=["observation_id", "source_round", *columns, "y"])
        peer = pd.DataFrame([["peer-A", 8, *([.9]*5), .9]], columns=own.columns)
        pooled = pd.concat([peer, own.iloc[[1]], own.iloc[[0]]], ignore_index=True)
        with tempfile.TemporaryDirectory() as temporary:
            own_path, pooled_path = Path(temporary)/"own.csv", Path(temporary)/"pooled.csv"
            own.to_csv(own_path, index=False)
            pooled.to_csv(pooled_path, index=False)
            result = load_cohort_review(pooled_path, own_path, input_columns=columns,
                                         expected_round_counts={0: 1, 1: 1}, expected_peer_count=1)
            np.testing.assert_array_equal(result["own_positions"], [2, 1])
            np.testing.assert_array_equal(result["latest_mask"], [False, True, False])
            self.assertEqual(result["latest_submissions"].iloc[0].output_rank, 2)
            self.assertEqual(result["data"].iloc[0].source_round, 8)
            pooled.loc[1, "y"] += .001
            pooled.to_csv(pooled_path, index=False)
            with self.assertRaisesRegex(ValueError, "changed official"):
                load_cohort_review(pooled_path, own_path, input_columns=columns)

    def test_paired_errors_use_aligned_targets_not_pooled_order(self):
        own_run = {"predictions": {"A": np.array([1., 2., 3.])}}
        pooled_run = {"predictions": {"A": np.array([99., 3.5, 1.5, 2.5])}}
        table = paired_target_comparison(own_run, pooled_run, [1., 2., 3.],
                                          own_positions=np.array([2, 3, 1]), peer_count=1)
        self.assertEqual(table.iloc[0, 1], 0)
        self.assertEqual(table.iloc[0, 2], .5)

    def test_common_pool_and_both_proposals_use_all_five_coordinates(self):
        columns = [f"control_{i}" for i in range(5)]
        fits = {"A": {"selected_model": CoordinateResponse(), "selected_name": "all inputs"}}
        points = np.array([[.2]*5, [.2]*4+[.3], [.2]*4+[.3+1e-8]])
        table = common_candidate_comparison(fits, [points], [[.2]*5], input_columns=columns)
        self.assertEqual(table.iloc[0].Candidates, 1)
        self.assertAlmostEqual(table.iloc[0].posterior_mean, 1.1)
        predictions = compare_proposal_predictions(fits, {"P": points[1]}, input_columns=columns)
        self.assertAlmostEqual(predictions.iloc[0].Mean, 1.1)
        self.assertEqual(predictions.iloc[0]["control_4"], .3)

    def test_stale_saved_proposal_is_rejected_against_the_refitted_snapshot(self):
        summary = dict(function=2, week=5, portal_input="0.2-0.3", point=[.2, .3],
                       observations=14, query_model="GP", gp_rmse=.15)
        run = dict(selected_name="GP", selected_rmse=.15)
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            "", outputs=[nbformat.v4.new_output("stream", name="stdout", text=json.dumps(summary))])])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"official.ipynb"
            nbformat.write(notebook, path)
            self.assertEqual(read_saved_proposal(path, function=2, week=5, model_run=run,
                                                 observation_count=14), summary)
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_saved_proposal(path, function=2, week=5, model_run=run, observation_count=15)


class NoiseEvidenceTests(unittest.TestCase):
    def frame(self, inputs, target):
        frame = pd.DataFrame(inputs, columns=[f"x{i}" for i in range(np.asarray(inputs).shape[1])])
        frame.insert(0, "observation_id", [f"id-{i}" for i in range(len(frame))])
        frame["y"] = target
        return frame

    def review(self, frame, **settings):
        return nearby_output_differences(frame, input_columns=[col for col in frame if col.startswith("x")], **settings)

    def test_exact_repeats_use_within_group_degrees_of_freedom(self):
        result = self.review(self.frame([[.2]*5]*3+[[.8]*5]*2, [1., 2., 3., 5., 7.]))
        summary = result["summary"]
        self.assertEqual(summary["exact_input_pairs"], 4)
        self.assertEqual(summary["repeat_degrees_of_freedom"], 3)
        self.assertAlmostEqual(summary["repeat_candidate_sd"], np.sqrt(4/3))

    def test_rounded_collision_is_not_an_exact_repeat(self):
        result = self.review(self.frame([[.2], [.2+1e-8]], [1., 1.2]))
        self.assertEqual(result["summary"]["exact_input_pairs"], 0)
        self.assertEqual(result["summary"]["submission_precision_pairs"], 1)
        self.assertIsNone(result["summary"]["repeat_candidate_sd"])

    def test_cutoffs_include_full_distance_and_scale_in_output_units(self):
        frame = self.frame([[0., 0., 0.], [.01, 0., 0.], [.01, 0., .2]], [0., .02, .4])
        result = self.review(frame, cutoffs=(.005, .015, .3))
        self.assertTrue(np.isnan(result["cutoffs"].iloc[0]["Noise SD proxy"]))
        self.assertEqual(result["cutoffs"].iloc[1].Pairs, 1)
        self.assertEqual(result["cutoffs"].iloc[1]["Distinct observations"], 2)
        self.assertAlmostEqual(result["cutoffs"].iloc[1]["Noise SD proxy"], .02/np.sqrt(2))
        scaled = frame.assign(y=frame.y*3+7)
        other = self.review(scaled, cutoffs=(.005, .015, .3))
        np.testing.assert_allclose(other["cutoffs"]["Noise SD proxy"], result["cutoffs"]["Noise SD proxy"]*3)

    def test_incomplete_or_nonfinite_observations_and_invalid_cutoffs_fail(self):
        frame = self.frame([[.1], [.2]], [.3, .4])
        for changed, settings in [(frame.assign(y=[np.inf, .4]), {}),
                                  (frame.assign(observation_id=["A", "A"]), {}),
                                  (frame, {"cutoffs": (0,)}), (frame, {"cutoffs": ()})]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.review(changed, **settings)

    def test_noise_fit_converts_variance_to_original_sd_without_mutating_models(self):
        inputs = np.random.default_rng(7).random((10, 5))
        target = np.sin(inputs[:, 1]*5)+inputs[:, 3]
        models = make_regression_models(5, feature_sets={"Projected": (1, 3)}, optimiser_restarts=0)
        for estimator in models.values():
            gp = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
            if hasattr(gp, "normalize_y"):
                gp.set_params(optimizer=None)
        result = fit_noise_diagnostics(models, inputs, target, model_names=["Projected", "Gaussian process"],
                                        initial_noise_variance=.04)
        np.testing.assert_allclose(result["Noise SD"], .2*target.std(ddof=0))
        scaled = fit_noise_diagnostics(models, inputs, target*4+2, model_names=["Projected"], initial_noise_variance=.04)
        self.assertAlmostEqual(scaled.iloc[0]["Noise SD"], result.iloc[0]["Noise SD"]*4)
        self.assertFalse(hasattr(models["Projected"].steps[-1][1], "kernel_"))
        self.assertAlmostEqual(models["Gaussian process"].alpha, .01)

    def test_noise_and_two_input_comparison_plots_export_readable_artifacts(self):
        result = self.review(self.frame([[.1, .1], [.11, .1], [.15, .15]], [.1, .12, .3]))
        paired = pd.DataFrame({"Model": ["A"], "Own RMSE": [.2], "Pooled RMSE": [.1]})
        with tempfile.TemporaryDirectory() as temporary:
            noise_path, map_path = Path(temporary)/"noise.png", Path(temporary)/"comparison.png"
            noise = plot_nearby_differences(result, assumed_noise_sd=.02, title="Noise", save_to=noise_path)
            plot_cohort_comparison(paired, [[.1, .1], [.11, .1]], [[.15, .15]], [.2, .3], [.4, .5],
                                    title="Comparison", save_to=map_path)
            self.assertEqual(len(noise.axes), 2)
            self.assertGreater(noise_path.stat().st_size, 1000)
            self.assertGreater(map_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

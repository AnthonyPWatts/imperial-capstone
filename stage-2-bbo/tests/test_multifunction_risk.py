"""Behavioural checks for the dimension-aware risk study."""

from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.gaussian_process.kernels import ConstantKernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multifunction_risk import (AdditiveMatern, candidate_pool, coverage_scores,
                                ensure_risk_results, fit_model, posterior_on_pool,
                                study_fingerprint)
from function_2_risk import score_queries


class MultidimensionalRiskTests(unittest.TestCase):
    def test_cache_reuses_matching_data_and_both_numerical_sources(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            data = pd.DataFrame({"x1": [.1, .7], "y": [-.2, -.1]})
            saved = {"fingerprint": study_fingerprint(data),
                     "elapsed_rollout_seconds": 1, "noise_checks": [],
                     "result": "previous calculation"}
            (output / "risk-study.json").write_text(json.dumps(saved), encoding="utf-8")
            with patch("multifunction_risk.analyse_risk") as calculate:
                self.assertEqual(ensure_risk_results(data, 1, output), saved)
                calculate.assert_not_called()

    def test_changed_numerical_source_invalidates_cached_study(self):
        for filename in ("multifunction_risk.py", "function_2_risk.py"):
            with self.subTest(source=filename), TemporaryDirectory() as directory:
                output = Path(directory)
                for source in ("multifunction_risk.py", "function_2_risk.py"):
                    (output / source).write_text("original calculation", encoding="utf-8")
                data = pd.DataFrame({"x1": [.1, .7], "y": [-.2, -.1]})
                with patch("multifunction_risk.__file__", str(output / "multifunction_risk.py")):
                    saved = {"fingerprint": study_fingerprint(data),
                             "elapsed_rollout_seconds": 1, "noise_checks": []}
                    (output / "risk-study.json").write_text(json.dumps(saved), encoding="utf-8")
                    (output / filename).write_text("changed calculation", encoding="utf-8")
                    with patch("multifunction_risk.analyse_risk",
                               side_effect=RuntimeError("recalculation requested")) as calculate:
                        with self.assertRaisesRegex(RuntimeError, "recalculation requested"):
                            ensure_risk_results(data, 1, output)
                        calculate.assert_called_once()

    def test_changed_observations_invalidate_cached_study(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            data = pd.DataFrame({"x1": [.1, .7], "y": [-.2, -.1]})
            saved = {"fingerprint": study_fingerprint(data),
                     "elapsed_rollout_seconds": 1, "noise_checks": []}
            (output / "risk-study.json").write_text(json.dumps(saved), encoding="utf-8")
            data.loc[0, "y"] = .4
            with patch("multifunction_risk.analyse_risk",
                       side_effect=RuntimeError("recalculation requested")):
                with self.assertRaisesRegex(RuntimeError, "recalculation requested"):
                    ensure_risk_results(data, 1, output)

    def test_additive_process_has_no_two_input_interaction(self):
        points = np.array([[.1, .2], [.1, .8], [.7, .2], [.7, .8]])
        kernel = AdditiveMatern((.3, .4))
        contrast = np.array([1, -1, -1, 1])
        self.assertAlmostEqual(contrast @ kernel(points) @ contrast, 0, places=12)
        np.testing.assert_allclose(kernel.diag(points), np.diag(kernel(points)))
        self.assertGreaterEqual(np.linalg.eigvalsh(kernel(points)).min(), -1e-12)

    def test_additive_gradient_matches_finite_differences_and_cloning(self):
        kernel = ConstantKernel(1.4) * AdditiveMatern((.2, .4, .7))
        points = np.array([[.2, .3, .1], [.7, .6, .8], [.4, .8, .9]])
        actual, derivative = kernel(points, eval_gradient=True)
        np.testing.assert_allclose(clone(kernel)(points), actual)
        for index in range(len(kernel.theta)):
            step = np.zeros(len(kernel.theta))
            step[index] = 1e-5
            numerical = (kernel.clone_with_theta(kernel.theta + step)(points)
                         - kernel.clone_with_theta(kernel.theta - step)(points)) / 2e-5
            np.testing.assert_allclose(derivative[:, :, index], numerical, atol=1e-8)

    def test_coverage_counts_full_space_and_existing_point_adds_nothing(self):
        values, before = coverage_scores(np.array([[.5]]), np.array([[.5], [.25], [.75]]), power=12)
        self.assertEqual(values[0], 0)
        self.assertAlmostEqual(before, .25, places=5)
        np.testing.assert_allclose(values[1:], [.3125, .3125], atol=1e-5)

    def test_pool_keeps_dimension_bounds_and_observation_neighbourhoods(self):
        inputs = np.random.default_rng(4).uniform(.1, .9, (15, 8))
        pool = candidate_pool(inputs, np.arange(15))
        self.assertEqual(pool.shape[1], 8)
        self.assertTrue(((pool >= 0) & (pool < 1)).all())
        self.assertEqual(len(pool), len(np.unique(pool, axis=0)))
        for point in inputs:
            self.assertTrue(np.any(np.all(pool == np.round(point, 6), axis=1)))

    def test_noise_and_query_values_keep_original_output_units(self):
        inputs = np.array([[.1, .2, .3], [.6, .7, .2], [.8, .3, .9], [.4, .5, .6]])
        outputs = np.array([-.3, -.1, -.2, -.6])
        pool = np.vstack([inputs, [.5, .5, .5]])
        first = fit_model(inputs, outputs, "Additive")
        second = fit_model(inputs, 3 * outputs + 2, "Additive")
        self.assertAlmostEqual(second["noise_sd"], 3 * first["noise_sd"])
        gain1 = score_queries(posterior_on_pool(first, pool), len(pool))
        gain2 = score_queries(posterior_on_pool(second, pool), len(pool))
        np.testing.assert_allclose(gain2, 3 * gain1, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

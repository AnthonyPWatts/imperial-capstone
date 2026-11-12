"""Observable recovery, log-domain behaviour and held-out isolation of source fits."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/Function_1/FunctionSpecificShared"))
from function_1_source_helpers import (
    fit_gaussian_source, predict_source_log10, source_peak,
    source_target_predictions, source_leave_one_out,
)


class SourceShapeTests(unittest.TestCase):
    def fixture(self, anisotropic=False, centre=(.57, .46)):
        inputs = np.array([[.1,.1],[.3,.2],[.4,.8],[.7,.3],[.8,.9],
                           [.2,.7],[.6,.5],[.9,.6],[.1,.9]])
        widths = np.array([.055, .095] if anisotropic else [.055, .055])
        logs = np.log10(1.7) - .5*np.sum(((inputs-centre)/widths)**2, axis=1)/np.log(10)
        return inputs, 10.**logs, logs

    def test_recovers_circular_source_from_tiny_positive_readings(self):
        inputs, outputs, logs = self.fixture()
        before = outputs.copy()
        fitted = fit_gaussian_source(inputs, outputs, positive_count=None)
        np.testing.assert_allclose(fitted["stationary_point"], [.57,.46], atol=1e-10)
        np.testing.assert_allclose(predict_source_log10(fitted, inputs), logs, atol=1e-10)
        self.assertAlmostEqual(source_peak(fitted)["predicted_height"], 1.7, places=9)
        np.testing.assert_array_equal(outputs, before)

    def test_recovers_anisotropic_widths_and_permitted_boundary_peak(self):
        inputs, outputs, _ = self.fixture(True, centre=(1.1,.46))
        fitted = fit_gaussian_source(inputs, outputs, positive_count=None, anisotropic=True)
        np.testing.assert_allclose(fitted["sigma"], [.055,.095], atol=1e-10)
        peak = source_peak(fitted)
        self.assertEqual(peak["portal_input"], "0.999999-0.460000")
        self.assertFalse(peak["stationary_point_inside_domain"])

    def test_negative_readings_are_excluded_only_from_positive_source_training(self):
        inputs, outputs, _ = self.fixture()
        inputs = np.vstack([inputs, [.5,.5]])
        outputs = np.r_[outputs, -5.]
        fitted = fit_gaussian_source(inputs, outputs, positive_count=5)
        self.assertNotIn(9, fitted["selected_indices"])
        np.testing.assert_allclose(fitted["stationary_point"], [.57,.46], atol=1e-9)

    def test_held_out_value_cannot_change_its_own_training_selection_or_prediction(self):
        inputs, outputs, _ = self.fixture()
        first = source_leave_one_out(inputs, outputs, positive_count=5)
        changed = outputs.copy();changed[0] = -100.
        second = source_leave_one_out(inputs, changed, positive_count=5)
        self.assertEqual(first["log10_predictions"][0], second["log10_predictions"][0])
        self.assertEqual(first["folds"][0]["selected_training_indices"], second["folds"][0]["selected_training_indices"])
        for row in first["folds"]:
            self.assertNotIn(row["held_out_index"], row["selected_training_indices"])

    def test_log_domain_retains_extreme_predictions_without_changing_targets(self):
        prediction = source_target_predictions(np.array([-500., -200., -10.]))
        self.assertEqual(prediction["raw_signed"][0], 0.)
        self.assertGreater(prediction["signed"][0], 0.)
        self.assertAlmostEqual(prediction["signed"][1], np.log10(2))
        np.testing.assert_array_equal(prediction["signed"], prediction["magnitude"])

    def test_invalid_and_underidentified_source_data_are_rejected(self):
        inputs, outputs, _ = self.fixture()
        for bad in [np.zeros(len(outputs)), np.full(len(outputs), np.nan)]:
            with self.assertRaisesRegex(ValueError, "non-zero"):
                fit_gaussian_source(inputs, bad)
        with self.assertRaisesRegex(ValueError, "independent"):
            fit_gaussian_source(np.full((6,2), .5), np.arange(1.,7.))
        with self.assertRaisesRegex(ValueError, "positive_count"):
            fit_gaussian_source(inputs, outputs, positive_only=False)


if __name__ == "__main__":
    unittest.main()

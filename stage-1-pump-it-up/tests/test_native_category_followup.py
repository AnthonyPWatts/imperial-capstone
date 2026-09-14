"""Regression checks for locked recipes, partition isolation and cache replay."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from native_category_followup import (
    BLENDS, CHALLENGERS, blend_candidates, cache_digest,
    compare_probabilities, make_model, native_frames, validate_cache,
)


class NativeCategoryFollowupTests(unittest.TestCase):
    def test_validation_only_categories_are_missing(self):
        training = pd.DataFrame({"category": ["b", "a"], "number": [1.0, np.nan]})
        validation = pd.DataFrame({"category": ["a", "unseen"], "number": [2.0, 3.0]})
        with patch("native_category_followup.engineer_spatial_grid_catboost_features",
                   side_effect=lambda frame: (frame.copy(), ("category",))):
            fit, predict, _ = native_frames(training, validation)
        self.assertEqual(list(fit.category.cat.categories), ["a", "b"])
        self.assertEqual(list(predict.category.cat.categories), ["a", "b"])
        self.assertTrue(pd.isna(predict.category.iloc[1]))
        self.assertEqual(str(training.category.dtype), "str")

    def test_recipes_are_locked_without_mutating_legacy_variants(self):
        from gpu_model_evaluation import CATBOOST_VARIANTS
        self.assertEqual(make_model(CHALLENGERS[0], 12).get_params()["depth"], 10)
        self.assertEqual(CATBOOST_VARIANTS["d8"]["depth"], 8)
        native = make_model(CHALLENGERS[1], 12).get_params()
        self.assertEqual(native["cat_smooth"], 20.0)
        self.assertEqual(native["n_estimators"], 12)

    def test_blends_use_exact_fixed_weights(self):
        incumbent = np.array([[0.5, 0.2, 0.3]])
        grid = np.array([[0.4, 0.3, 0.3]])
        depth = np.array([[0.6, 0.1, 0.3]])
        native = np.array([[0.2, 0.4, 0.4]])
        result = blend_candidates(incumbent, grid, dict(zip(CHALLENGERS, [depth, native])))
        np.testing.assert_allclose(result[BLENDS[0]], [[0.51, 0.19, 0.3]])
        np.testing.assert_allclose(result[BLENDS[1]], [[0.47, 0.22, 0.31]])
        np.testing.assert_allclose(result["combined_locked_changes"],
                                   0.9 * result[BLENDS[0]] + 0.1 * native)

    def test_cache_rejects_changed_probabilities_and_misordered_ids(self):
        metadata, ids = {"fold": 1}, np.array([3, 7])
        probabilities = np.array([[0.5, 0.2, 0.3], [0.2, 0.3, 0.5]])
        diagnostics = {"selected_iterations": 20}
        payload = dict(metadata=metadata, ids=ids, probabilities=probabilities.copy(),
                       diagnostics=diagnostics,
                       digest=cache_digest(metadata, ids, probabilities, diagnostics))
        validate_cache(payload, metadata, ids)
        with self.assertRaises(ValueError):
            validate_cache(payload, metadata, ids[::-1])
        payload["probabilities"][0] = [0.4, 0.3, 0.3]
        with self.assertRaises(ValueError):
            validate_cache(payload, metadata, ids)

    def test_equal_predictions_do_not_pass_positive_gain_guard(self):
        target = pd.Series(["functional", "functional needs repair", "non functional"] * 5)
        probabilities = np.tile(np.eye(3), (5, 1))
        comparison = compare_probabilities(target, np.repeat(np.arange(1, 6), 3),
                                           probabilities, probabilities)
        self.assertEqual(comparison["net_correct_rows"], 0)
        self.assertFalse(comparison["passes_guard"])


if __name__ == "__main__":
    unittest.main()

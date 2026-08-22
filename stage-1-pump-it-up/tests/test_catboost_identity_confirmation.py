"""Tests for the frozen complete-identity CatBoost confirmation recipe."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_confirmation import blend_frozen_identity_recipes
from catboost_identity_confirmation import (
    build_frozen_identity_competition_predictions,
)
from catboost_identity_confirmation import score_identity_recipes
from catboost_identity_confirmation import summarise_paired_predictions
from modelling_data import ModellingData


class CatBoostIdentityConfirmationTests(unittest.TestCase):
    def test_frozen_recipes_use_preselected_weights(self) -> None:
        xgboost = _probabilities(0.70, 0.20, 0.10)
        forest = _probabilities(0.50, 0.30, 0.20)
        catboost = _probabilities(0.20, 0.30, 0.50)

        recipes = blend_frozen_identity_recipes(
            {
                "xgboost": xgboost,
                "random_forest": forest,
                "identity_catboost": catboost,
            }
        )

        np.testing.assert_allclose(
            recipes["accepted_55_45"],
            0.55 * xgboost + 0.45 * forest,
        )
        np.testing.assert_allclose(
            recipes["identity_candidate_44_36_20"],
            0.44 * xgboost + 0.36 * forest + 0.20 * catboost,
        )

    def test_scoring_compares_unchanged_three_class_labels(self) -> None:
        target = pd.Series(
            ["functional", "functional needs repair", "non functional"]
        )
        accepted = np.asarray(
            [[0.8, 0.1, 0.1], [0.5, 0.4, 0.1], [0.1, 0.2, 0.7]]
        )
        candidate = np.asarray(
            [[0.7, 0.2, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7]]
        )

        metrics, confusions = score_identity_recipes(
            target,
            {
                "accepted_55_45": accepted,
                "identity_candidate_44_36_20": candidate,
            },
        )

        self.assertAlmostEqual(metrics.loc["accuracy", "accepted_55_45"], 2 / 3)
        self.assertEqual(
            metrics.loc["accuracy", "identity_candidate_44_36_20"],
            1.0,
        )
        self.assertEqual(
            int(confusions["identity_candidate_44_36_20"].to_numpy().sum()),
            3,
        )
        paired = summarise_paired_predictions(
            target,
            {
                "accepted_55_45": accepted,
                "identity_candidate_44_36_20": candidate,
            },
            bootstrap_samples=100,
        )
        self.assertEqual(paired["candidate_only_correct"], 1)
        self.assertEqual(paired["accepted_only_correct"], 0)
        self.assertEqual(paired["net_additional_correct"], 1)

    def test_competition_builder_preserves_template_ids_and_labels(self) -> None:
        modelling_data = ModellingData(
            original_ids=pd.Series([1, 2]),
            X_original=pd.DataFrame({"feature": [1, 2]}),
            y_original=pd.Series(["functional", "non functional"]),
            competition_ids=pd.Series([3]),
            X_competition=pd.DataFrame({"feature": [3]}),
        )
        components = {
            "xgboost": _probabilities(0.70, 0.20, 0.10),
            "random_forest": _probabilities(0.50, 0.30, 0.20),
            "identity_catboost": _probabilities(0.20, 0.30, 0.50),
        }

        predictions = build_frozen_identity_competition_predictions(
            modelling_data,
            pd.DataFrame({"id": [3], "status_group": ["functional"]}),
            components,
            pd.Series(
                {
                    "XGBoost": 1.0,
                    "Random Forest": 2.0,
                    "complete-identity CatBoost": 3.0,
                }
            ),
        )

        candidate = predictions["identity_candidate_44_36_20"]
        self.assertEqual(
            candidate.submission.to_dict(orient="records"),
            [{"id": 3, "status_group": "functional"}],
        )
        self.assertEqual(
            predictions["accepted_55_45"].component_seconds.index.tolist(),
            ["XGBoost", "Random Forest"],
        )


def _probabilities(functional: float, repair: float, non_functional: float):
    return np.asarray([[functional, repair, non_functional]])


if __name__ == "__main__":
    unittest.main()

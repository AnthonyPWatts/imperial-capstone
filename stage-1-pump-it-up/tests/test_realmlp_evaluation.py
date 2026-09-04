"""Focused tests for the fold-safe RealMLP-TD experiment utilities."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import EXPECTED_SOURCE_FEATURES
from realmlp_evaluation import CLASS_LABELS
from realmlp_evaluation import REALMLP_BLEND_WEIGHTS
from realmlp_evaluation import REALMLP_CATEGORICAL_FEATURES
from realmlp_evaluation import blend_probabilities
from realmlp_evaluation import fit_realmlp_feature_state
from realmlp_evaluation import ordered_probabilities
from realmlp_evaluation import transform_realmlp_features


class RealMLPEvaluationTests(unittest.TestCase):
    def test_training_fold_medians_impute_prediction_rows(self) -> None:
        training = _source_frame()
        training.loc[0, "gps_height"] = 100.0
        training.loc[1, "gps_height"] = 200.0
        prediction = training.iloc[[0]].copy()
        prediction.loc[:, "gps_height"] = 0.0

        state = fit_realmlp_feature_state(training)
        transformed = transform_realmlp_features(prediction, state)

        self.assertEqual(transformed.loc[0, "gps_height"], 150.0)
        self.assertEqual(transformed.loc[0, "gps_height_missing"], 1.0)
        self.assertTrue(
            np.isfinite(
                transformed.loc[:, state.numeric_features].to_numpy(
                    dtype="float64"
                )
            ).all()
        )

    def test_all_normalised_identity_fields_are_native_categories(self) -> None:
        state = fit_realmlp_feature_state(_source_frame())
        transformed = transform_realmlp_features(_source_frame(), state)

        self.assertEqual(
            transformed.loc[0, "funder_identity"],
            "government",
        )
        self.assertEqual(transformed.loc[0, "wpt_name_identity"], "pump a")
        self.assertEqual(
            transformed.loc[1, "scheme_name_identity"],
            "__blank__",
        )
        self.assertTrue(
            all(column in transformed for column in REALMLP_CATEGORICAL_FEATURES)
        )

    def test_fixed_blends_remain_normalised(self) -> None:
        incumbent = np.asarray([[0.6, 0.1, 0.3], [0.2, 0.3, 0.5]])
        realmlp = np.asarray([[0.3, 0.4, 0.3], [0.1, 0.7, 0.2]])

        for weight in REALMLP_BLEND_WEIGHTS:
            blended = blend_probabilities(incumbent, realmlp, weight)
            np.testing.assert_allclose(blended.sum(axis=1), 1.0)

        with self.assertRaisesRegex(ValueError, "strictly between"):
            blend_probabilities(incumbent, realmlp, 0.0)

    def test_probability_columns_are_reordered_to_competition_contract(self) -> None:
        classifier = SimpleNamespace(
            classes_=np.asarray(
                ["non functional", "functional", "functional needs repair"]
            ),
            predict_proba=lambda X: np.asarray([[0.2, 0.7, 0.1]]),
        )

        probabilities = ordered_probabilities(classifier, pd.DataFrame({"x": [1]}))

        self.assertEqual(CLASS_LABELS[probabilities.argmax(axis=1)[0]], "functional")
        np.testing.assert_allclose(probabilities, [[0.7, 0.1, 0.2]])


def _source_frame() -> pd.DataFrame:
    defaults: dict[str, object] = {
        name: "value" for name in EXPECTED_SOURCE_FEATURES
    }
    defaults.update(
        {
            "amount_tsh": 10.0,
            "date_recorded": "2013-01-01",
            "funder": " Government ",
            "installer": "DWE",
            "wpt_name": "Pump   A",
            "gps_height": 100.0,
            "longitude": 31.0,
            "latitude": -6.0,
            "num_private": 0.0,
            "region_code": 11,
            "district_code": 4,
            "population": 100.0,
            "construction_year": 2000,
            "region": "Kagera",
            "lga": "Misenyi",
            "scheme_name": "Scheme",
        }
    )
    second = defaults.copy()
    second.update({"funder": None, "scheme_name": ""})
    return pd.DataFrame([defaults, second], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

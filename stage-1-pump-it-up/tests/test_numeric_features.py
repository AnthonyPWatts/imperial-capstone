"""Tests for fold-fitted numeric state and imputation policies."""

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

from feature_engineering import EXPECTED_SOURCE_FEATURES
from feature_engineering import engineer_initial_features
from numeric_features import MEASUREMENT_BLOCK_MISSING_FEATURE
from numeric_features import NUMERIC_POLICIES
from numeric_features import NUM_PRIVATE_NONZERO_FEATURE
from numeric_features import POPULATION_ONE_FEATURE
from numeric_features import NumericFeatureEngineer
from numeric_features import make_numeric_preprocessor
from numeric_features import numeric_model_features
from numeric_features import numeric_numeric_features


class NumericFeatureTests(unittest.TestCase):
    def test_baseline_reproduces_initial_feature_frame(self) -> None:
        frame = _source_frame()
        policy = NUMERIC_POLICIES["baseline"]

        actual = NumericFeatureEngineer(policy).fit_transform(frame)

        pd.testing.assert_frame_equal(actual, engineer_initial_features(frame))

    def test_zero_retention_keeps_value_and_missing_state(self) -> None:
        frame = _source_frame()
        gps_policy = NUMERIC_POLICIES["gps_zero_as_value"]
        population_policy = NUMERIC_POLICIES["population_zero_as_value"]

        gps = NumericFeatureEngineer(gps_policy).fit_transform(frame)
        population = NumericFeatureEngineer(population_policy).fit_transform(frame)

        self.assertEqual(gps.loc[1, "gps_height"], 0)
        self.assertEqual(gps.loc[1, "gps_height_missing"], 1)
        self.assertEqual(population.loc[1, "population"], 0)
        self.assertEqual(population.loc[1, "population_missing"], 1)

    def test_geographic_medians_are_fitted_without_prediction_rows(self) -> None:
        training = pd.concat([_source_frame().iloc[[0]]] * 20, ignore_index=True)
        training.loc[:, "gps_height"] = np.arange(100, 120, dtype="float64")
        training.loc[:, "population"] = np.arange(10, 30, dtype="float64")
        prediction = _source_frame().iloc[[1]].copy()
        prediction.loc[:, "lga"] = "unseen lga"
        policy = NUMERIC_POLICIES["both_geographic_medians"]
        transformer = NumericFeatureEngineer(policy).fit(training)

        transformed = transformer.transform(prediction)

        self.assertAlmostEqual(transformed.iloc[0]["gps_height"], 109.5)
        self.assertAlmostEqual(transformed.iloc[0]["population"], 19.5)
        self.assertEqual(transformed.iloc[0]["gps_height_missing"], 1)
        self.assertEqual(transformed.iloc[0]["population_missing"], 1)

    def test_combined_states_replace_sparse_magnitude(self) -> None:
        frame = _source_frame()
        frame.loc[0, "population"] = 1
        frame.loc[0, "num_private"] = 3
        policy = NUMERIC_POLICIES["combined_state_indicators"]
        transformer = NumericFeatureEngineer(policy).fit(frame)

        transformed = transformer.transform(frame)

        self.assertNotIn("num_private", transformed.columns)
        self.assertEqual(transformed.loc[0, POPULATION_ONE_FEATURE], 1)
        self.assertEqual(transformed.loc[0, NUM_PRIVATE_NONZERO_FEATURE], 1)
        self.assertEqual(
            transformed.loc[1, MEASUREMENT_BLOCK_MISSING_FEATURE],
            1,
        )

    def test_policy_feature_contracts_are_unique_and_numeric(self) -> None:
        for policy in NUMERIC_POLICIES.values():
            features = numeric_model_features(policy)
            numeric = numeric_numeric_features(policy)

            self.assertEqual(len(features), len(set(features)))
            self.assertTrue(set(numeric).issubset(features))

    def test_preprocessor_handles_all_registered_policies(self) -> None:
        frame = pd.concat([_source_frame()] * 12, ignore_index=True)
        for policy in NUMERIC_POLICIES.values():
            preprocessor = make_numeric_preprocessor(policy)
            matrix = preprocessor.fit_transform(frame)

            self.assertEqual(matrix.shape[0], len(frame))
            self.assertEqual(
                matrix.shape[1],
                len(preprocessor.get_feature_names_out()),
            )
            values = matrix.data if hasattr(matrix, "data") else np.asarray(matrix)
            self.assertTrue(np.isfinite(values).all())


def _source_frame() -> pd.DataFrame:
    defaults: dict[str, object] = {
        name: "value" for name in EXPECTED_SOURCE_FEATURES
    }
    defaults.update(
        {
            "amount_tsh": 10.0,
            "date_recorded": "2013-01-01",
            "gps_height": 100.0,
            "longitude": 31.774483,
            "latitude": -0.998916,
            "num_private": 0.0,
            "region_code": 11,
            "district_code": 4,
            "population": 100.0,
            "construction_year": 2000,
            "region": "Kagera",
            "lga": "Misenyi",
            "ward": "Kashenye",
        }
    )
    valid = defaults.copy()
    sentinel = defaults.copy()
    sentinel.update(
        {
            "gps_height": 0.0,
            "population": 0.0,
            "construction_year": 0,
        }
    )
    return pd.DataFrame([valid, sentinel], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

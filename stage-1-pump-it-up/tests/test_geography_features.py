"""Tests for deterministic geography feature policies and coordinate safety."""

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
from feature_engineering import valid_tanzania_coordinates
from geography_features import GEOGRAPHY_POLICIES
from geography_features import engineer_geography_features
from geography_features import geography_categorical_features
from geography_features import geography_model_features
from geography_features import geography_numeric_features
from geography_features import make_geography_preprocessor
from geography_features import GeographyFeatureEngineer


class GeographyFeatureTests(unittest.TestCase):
    def test_coordinate_envelope_rejects_origin_and_nearby_values(self) -> None:
        longitude = pd.Series([0.0, 0.0001, 27.999, 31.774483, 42.001])
        latitude = pd.Series([0.0, -0.00000002, -6.0, -0.998916, -6.0])

        actual = valid_tanzania_coordinates(longitude, latitude)

        self.assertEqual(actual.tolist(), [False, False, False, True, False])

    def test_policies_have_disjoint_numeric_and_categorical_contracts(self) -> None:
        for policy in GEOGRAPHY_POLICIES.values():
            model_features = geography_model_features(policy)
            numeric = geography_numeric_features(policy)
            categorical = geography_categorical_features(policy)

            self.assertEqual(len(model_features), len(set(model_features)))
            self.assertTrue(set(numeric).isdisjoint(categorical))
            self.assertEqual(set(numeric) | set(categorical), set(model_features))

    def test_composite_and_grid_policy_uses_missing_coordinate_token(self) -> None:
        frame = _source_frame()
        policy = GEOGRAPHY_POLICIES["combined_context"]

        transformed = engineer_geography_features(frame, policy)

        self.assertEqual(transformed.loc[0, "region_district"], "11::4")
        self.assertEqual(transformed.loc[0, "lga_ward"], "Misenyi::Kashenye")
        self.assertNotEqual(
            transformed.loc[0, "coordinate_grid_0_2_degrees"],
            "__missing__",
        )
        self.assertEqual(
            transformed.loc[1, "coordinate_grid_0_2_degrees"],
            "__missing__",
        )
        self.assertTrue(np.isnan(transformed.loc[1, "longitude"]))
        self.assertTrue(np.isnan(transformed.loc[1, "latitude"]))
        self.assertEqual(transformed.loc[1, "coordinates_missing"], 1)

    def test_preprocessor_handles_all_registered_policies(self) -> None:
        frame = _source_frame()
        for policy in GEOGRAPHY_POLICIES.values():
            preprocessor = make_geography_preprocessor(policy)
            matrix = preprocessor.fit_transform(frame)

            self.assertEqual(matrix.shape[0], len(frame))
            self.assertEqual(
                matrix.shape[1],
                len(preprocessor.get_feature_names_out()),
            )
            values = matrix.data if hasattr(matrix, "data") else np.asarray(matrix)
            self.assertTrue(np.isfinite(values).all())

    def test_coordinate_centroids_are_fitted_without_transform_rows(self) -> None:
        training = pd.concat([_source_frame().iloc[[0]]] * 10, ignore_index=True)
        training.loc[:, "longitude"] = np.arange(10, dtype="float64") / 100 + 31.0
        training.loc[:, "latitude"] = np.arange(10, dtype="float64") / 100 - 2.0
        prediction = _source_frame().iloc[[1]].copy()
        policy = GEOGRAPHY_POLICIES["coordinate_centroids_with_level"]
        transformer = GeographyFeatureEngineer(policy).fit(training)

        transformed = transformer.transform(prediction)

        self.assertAlmostEqual(transformed.iloc[0]["longitude"], 31.045)
        self.assertAlmostEqual(transformed.iloc[0]["latitude"], -1.955)
        self.assertEqual(transformed.iloc[0]["coordinates_missing"], 1)
        self.assertEqual(
            transformed.iloc[0]["coordinate_imputation_level"],
            "ward",
        )


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
    sentinel.update({"longitude": 0.0, "latitude": -0.00000002})
    return pd.DataFrame([valid, sentinel], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

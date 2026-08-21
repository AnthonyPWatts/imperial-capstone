"""Tests for fold-fitted funder and installer feature handling."""

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
from high_cardinality_features import BLANK_CATEGORY
from high_cardinality_features import HIGH_CARDINALITY_POLICIES
from high_cardinality_features import MISSING_CATEGORY
from high_cardinality_features import HighCardinalityFeatureEngineer
from high_cardinality_features import high_cardinality_model_features
from high_cardinality_features import make_high_cardinality_preprocessor
from high_cardinality_features import normalise_organisation


class HighCardinalityFeatureTests(unittest.TestCase):
    def test_normalisation_is_conservative_and_preserves_special_states(self) -> None:
        values = pd.Series(
            ["  DWE  ", "dwe", "District   Council", "", None, "0", "unknown"]
        )

        actual = normalise_organisation(values)

        self.assertEqual(
            actual.tolist(),
            [
                "dwe",
                "dwe",
                "district council",
                BLANK_CATEGORY,
                MISSING_CATEGORY,
                "0",
                "unknown",
            ],
        )

    def test_frequency_mapping_is_fitted_without_prediction_rows(self) -> None:
        training = pd.concat([_source_frame().iloc[[0]]] * 3, ignore_index=True)
        training.loc[:, "funder"] = ["A", "A", "B"]
        prediction = _source_frame().iloc[[0, 1]].copy()
        prediction.loc[:, "funder"] = ["A", "unseen"]
        policy = HIGH_CARDINALITY_POLICIES["funder_frequency"]
        transformer = HighCardinalityFeatureEngineer(policy).fit(training)

        transformed = transformer.transform(prediction)

        self.assertAlmostEqual(transformed.iloc[0]["funder_frequency"], 2 / 3)
        self.assertEqual(transformed.iloc[1]["funder_frequency"], 0)

    def test_rare_grouping_is_fitted_without_prediction_rows(self) -> None:
        training = pd.concat([_source_frame().iloc[[0]]] * 25, ignore_index=True)
        training.loc[:, "funder"] = [*(["A"] * 24), "B"]
        policy = HIGH_CARDINALITY_POLICIES["funder_rare20"]
        preprocessor = make_high_cardinality_preprocessor(policy).fit(training)
        encoder = preprocessor.named_steps[
            "column_preprocessing"
        ].named_transformers_["organisation_categorical"]

        encoded = encoder.transform(
            pd.DataFrame({"funder_category": ["b", "unseen"]})
        ).toarray()

        self.assertNotIn("unseen", encoder.categories_[0])
        np.testing.assert_array_equal(encoded[0], encoded[1])

    def test_policy_feature_contracts_are_unique(self) -> None:
        for policy in HIGH_CARDINALITY_POLICIES.values():
            features = high_cardinality_model_features(policy)
            self.assertEqual(len(features), len(set(features)))

    def test_preprocessor_handles_all_registered_policies(self) -> None:
        frame = pd.concat([_source_frame()] * 12, ignore_index=True)
        for policy in HIGH_CARDINALITY_POLICIES.values():
            preprocessor = make_high_cardinality_preprocessor(policy)
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
            "funder": "Government",
            "installer": "DWE",
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
    second = defaults.copy()
    second.update({"funder": "", "installer": "unknown"})
    return pd.DataFrame([defaults, second], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

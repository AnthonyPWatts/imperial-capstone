"""Tests for leakage-safe multiclass target-encoding features."""

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
from target_encoding_features import BLANK_IDENTITY
from target_encoding_features import MISSING_IDENTITY
from target_encoding_features import TARGET_ENCODING_POLICIES
from target_encoding_features import identity_feature_frame
from target_encoding_features import make_target_encoding_preprocessor
from target_encoding_features import normalise_identity
from target_encoding_features import target_encoding_model_features


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


class TargetEncodingFeatureTests(unittest.TestCase):
    def test_identity_normalisation_preserves_special_states(self) -> None:
        values = pd.Series(
            ["  Pump A  ", "pump   a", "", None, "0", "unknown"]
        )

        actual = normalise_identity(values)

        self.assertEqual(
            actual.tolist(),
            [
                "pump a",
                "pump a",
                BLANK_IDENTITY,
                MISSING_IDENTITY,
                "0",
                "unknown",
            ],
        )

    def test_location_names_are_composed_with_lga_context(self) -> None:
        frame = _source_frame(2)
        frame.loc[:, "lga"] = ["Misenyi", "Other LGA"]
        frame.loc[:, "ward"] = ["Central", "Central"]
        frame.loc[:, "subvillage"] = ["Kijiji", "Kijiji"]

        identities = identity_feature_frame(frame)

        self.assertEqual(
            identities["lga_ward_identity"].tolist(),
            ["misenyi::central", "other lga::central"],
        )
        self.assertEqual(
            identities["lga_subvillage_identity"].tolist(),
            ["misenyi::kijiji", "other lga::kijiji"],
        )

    def test_training_encoding_is_cross_fitted_instead_of_self_encoded(self) -> None:
        frame = _unique_identity_frame(30)
        target = _balanced_target(30)
        policy = TARGET_ENCODING_POLICIES["combined_identity"]
        preprocessor = make_target_encoding_preprocessor(policy)

        cross_fitted = _dense(preprocessor.fit_transform(frame, target))
        self_encoded = _dense(preprocessor.transform(frame))
        encoded_slice = preprocessor.named_steps[
            "column_preprocessing"
        ].output_indices_["target_encoded"]

        self.assertFalse(
            np.allclose(
                cross_fitted[:, encoded_slice],
                self_encoded[:, encoded_slice],
            )
        )
        self.assertTrue(np.isfinite(cross_fitted[:, encoded_slice]).all())

    def test_cross_fitted_encoding_is_deterministic(self) -> None:
        frame = _unique_identity_frame(30)
        target = _balanced_target(30)
        policy = TARGET_ENCODING_POLICIES["location_identity"]

        first = _dense(
            make_target_encoding_preprocessor(policy).fit_transform(frame, target)
        )
        second = _dense(
            make_target_encoding_preprocessor(policy).fit_transform(frame, target)
        )

        np.testing.assert_allclose(first, second)

    def test_all_registered_policies_produce_finite_feature_matrices(self) -> None:
        frame = _unique_identity_frame(30)
        target = _balanced_target(30)
        prediction = frame.iloc[[0]].copy()
        prediction.loc[:, "ward"] = "unseen ward"
        prediction.loc[:, "funder"] = "unseen funder"

        for policy in TARGET_ENCODING_POLICIES.values():
            preprocessor = make_target_encoding_preprocessor(policy)
            training = preprocessor.fit_transform(frame, target)
            predicted = preprocessor.transform(prediction)

            self.assertEqual(training.shape[0], len(frame))
            self.assertEqual(predicted.shape[0], 1)
            self.assertEqual(
                training.shape[1],
                len(preprocessor.get_feature_names_out()),
            )
            self.assertTrue(np.isfinite(_dense(training)).all())
            self.assertTrue(np.isfinite(_dense(predicted)).all())
            features = target_encoding_model_features(policy)
            self.assertEqual(len(features), len(set(features)))


def _source_frame(rows: int) -> pd.DataFrame:
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
            "subvillage": "Kijiji",
            "scheme_name": "Scheme",
        }
    )
    return pd.DataFrame(
        [defaults.copy() for _ in range(rows)],
        columns=EXPECTED_SOURCE_FEATURES,
    )


def _unique_identity_frame(rows: int) -> pd.DataFrame:
    frame = _source_frame(rows)
    for position in range(rows):
        frame.loc[position, "ward"] = f"ward {position}"
        frame.loc[position, "subvillage"] = f"subvillage {position}"
        frame.loc[position, "scheme_name"] = f"scheme {position}"
        frame.loc[position, "funder"] = f"funder {position}"
        frame.loc[position, "installer"] = f"installer {position}"
    return frame


def _balanced_target(rows: int) -> pd.Series:
    return pd.Series(
        [CLASS_LABELS[position % len(CLASS_LABELS)] for position in range(rows)]
    )


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


if __name__ == "__main__":
    unittest.main()

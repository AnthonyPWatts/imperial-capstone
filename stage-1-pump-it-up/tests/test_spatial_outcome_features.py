"""Tests for cross-fitted local spatial outcome features."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import PartitionedData
from feature_engineering import EXPECTED_SOURCE_FEATURES
from spatial_outcome_evaluation import evaluate_spatial_voter
from spatial_outcome_features import CLASS_LABELS
from spatial_outcome_features import SPATIAL_RATE_FEATURES
from spatial_outcome_features import SPATIAL_RADIUS_FEATURE
from spatial_outcome_features import SpatialOutcomeFeatureEngineer
from spatial_outcome_features import fit_spatial_outcome_model
from spatial_outcome_features import make_spatial_outcome_preprocessor
from spatial_outcome_features import spatial_outcome_values


class SpatialOutcomeFeatureTests(unittest.TestCase):
    def test_invalid_coordinates_use_prior_and_missing_radius(self) -> None:
        frame = _source_frame(60)
        target = _balanced_target(60)
        model = fit_spatial_outcome_model(frame, target)
        prediction = frame.iloc[[0]].copy()
        prediction.loc[:, "longitude"] = 0.0
        prediction.loc[:, "latitude"] = -0.00000002

        values = spatial_outcome_values(prediction, model)

        np.testing.assert_allclose(values[0, :3], model.class_prior)
        self.assertTrue(np.isnan(values[0, -1]))

    def test_training_features_are_cross_fitted_and_not_self_neighbours(self) -> None:
        frame = _source_frame(60)
        target = _balanced_target(60)
        transformer = SpatialOutcomeFeatureEngineer()

        cross_fitted = transformer.fit_transform(frame, target)
        self_inclusive = transformer.transform(frame)

        self.assertFalse(
            np.allclose(
                cross_fitted.loc[:, list(SPATIAL_RATE_FEATURES)],
                self_inclusive.loc[:, list(SPATIAL_RATE_FEATURES)],
            )
        )
        np.testing.assert_allclose(
            cross_fitted.loc[:, list(SPATIAL_RATE_FEATURES)].sum(axis=1),
            1.0,
        )

    def test_cross_fitted_spatial_features_are_deterministic(self) -> None:
        frame = _source_frame(60)
        target = _balanced_target(60)

        first = SpatialOutcomeFeatureEngineer().fit_transform(frame, target)
        second = SpatialOutcomeFeatureEngineer().fit_transform(frame, target)

        pd.testing.assert_frame_equal(first, second)

    def test_preprocessor_imputes_invalid_coordinate_radius(self) -> None:
        frame = _source_frame(60)
        target = _balanced_target(60)
        frame.loc[0, "longitude"] = 0.0
        frame.loc[0, "latitude"] = -0.00000002
        preprocessor = make_spatial_outcome_preprocessor()

        matrix = preprocessor.fit_transform(frame, target)

        values = matrix.data if hasattr(matrix, "data") else np.asarray(matrix)
        self.assertTrue(np.isfinite(values).all())
        self.assertIn(
            SPATIAL_RADIUS_FEATURE,
            preprocessor.named_steps[
                "feature_engineering"
            ].get_feature_names_out(),
        )

    def test_outer_validation_probabilities_ignore_its_own_labels(self) -> None:
        frame = _source_frame(75)
        target = _balanced_target(75)
        folds = pd.Series(
            np.tile(np.arange(1, 6), 15),
            name="validation_fold",
        )
        changed = target.copy()
        changed.loc[folds.eq(1)] = "functional"
        original_partition = _partition(frame, target, folds)
        changed_partition = _partition(frame, changed, folds)

        original = evaluate_spatial_voter(
            original_partition,
            PredefinedSplit(test_fold=folds.to_numpy() - 1),
        )
        modified = evaluate_spatial_voter(
            changed_partition,
            PredefinedSplit(test_fold=folds.to_numpy() - 1),
        )

        fold_one = folds.eq(1)
        np.testing.assert_allclose(
            original.out_of_fold_probabilities.loc[fold_one],
            modified.out_of_fold_probabilities.loc[fold_one],
        )


def _source_frame(rows: int) -> pd.DataFrame:
    defaults: dict[str, object] = {
        name: "value" for name in EXPECTED_SOURCE_FEATURES
    }
    defaults.update(
        {
            "amount_tsh": 10.0,
            "date_recorded": "2013-01-01",
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
        }
    )
    frame = pd.DataFrame(
        [defaults.copy() for _ in range(rows)],
        columns=EXPECTED_SOURCE_FEATURES,
    )
    positions = np.arange(rows)
    frame.loc[:, "longitude"] = 31.0 + (positions % 10) * 0.01
    frame.loc[:, "latitude"] = -6.0 - (positions // 10) * 0.01
    return frame


def _balanced_target(rows: int) -> pd.Series:
    return pd.Series(
        [CLASS_LABELS[position % len(CLASS_LABELS)] for position in range(rows)]
    )


def _partition(
    frame: pd.DataFrame,
    target: pd.Series,
    folds: pd.Series,
) -> PartitionedData:
    return PartitionedData(
        development_ids=pd.Series(range(len(frame))),
        X_development=frame,
        y_development=target,
        local_test_ids=pd.Series(dtype="int64"),
        X_local_test=frame.iloc[0:0].copy(),
        y_local_test=pd.Series(dtype="string"),
        validation_folds=folds,
        development_fingerprint="development",
        local_test_fingerprint="local-test",
        cross_validation_fingerprint="cross-validation",
    )


if __name__ == "__main__":
    unittest.main()

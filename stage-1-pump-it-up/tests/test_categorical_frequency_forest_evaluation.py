"""Tests for the all-categorical occurrence-count representation."""

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

from categorical_frequency_forest_evaluation import FREQUENCY_FEATURES
from categorical_frequency_forest_evaluation import FREQUENCY_SOURCE_FEATURES
from categorical_frequency_forest_evaluation import CategoricalFrequencyFeatureEngineer
from categorical_frequency_forest_evaluation import RADIAL_DISTANCE_FEATURE
from categorical_frequency_forest_evaluation import (
    RadialCategoricalFrequencyFeatureEngineer,
)
from categorical_frequency_forest_evaluation import ARCHIVE_FOREST_ESTIMATORS
from categorical_frequency_forest_evaluation import ARCHIVE_FOREST_MAX_FEATURES
from feature_engineering import EXPECTED_SOURCE_FEATURES
from model_evaluation import make_random_forest_pipeline


class CategoricalFrequencyForestEvaluationTests(unittest.TestCase):
    def test_counts_are_fitted_without_prediction_rows_and_rare_values_merge(self) -> None:
        training = _source_frame(["Alpha"] * 5 + ["Beta"] * 4)
        prediction = _source_frame(["Alpha", "Beta", "Gamma"])

        transformed = CategoricalFrequencyFeatureEngineer().fit_transform(training)
        predicted = CategoricalFrequencyFeatureEngineer().fit(training).transform(
            prediction
        )

        self.assertEqual(transformed.loc[0, "wpt_name_occurrence_count"], 5)
        self.assertEqual(transformed.loc[5, "wpt_name_occurrence_count"], -1)
        self.assertEqual(predicted.loc[0, "wpt_name_occurrence_count"], 5)
        self.assertEqual(predicted.loc[1, "wpt_name_occurrence_count"], -1)
        self.assertEqual(predicted.loc[2, "wpt_name_occurrence_count"], -1)

    def test_all_declared_sources_produce_one_ordered_numeric_feature(self) -> None:
        transformed = CategoricalFrequencyFeatureEngineer().fit_transform(
            _source_frame(["Alpha"] * 5)
        )

        self.assertEqual(len(FREQUENCY_SOURCE_FEATURES), len(set(FREQUENCY_SOURCE_FEATURES)))
        self.assertEqual(
            transformed.columns[-len(FREQUENCY_FEATURES) :].tolist(),
            list(FREQUENCY_FEATURES),
        )

    def test_source_frame_is_not_mutated(self) -> None:
        training = _source_frame(["Alpha"] * 5)
        original = training.copy(deep=True)

        CategoricalFrequencyFeatureEngineer().fit_transform(training)

        pd.testing.assert_frame_equal(training, original)

    def test_radial_distance_is_finite_only_for_valid_coordinates(self) -> None:
        training = _source_frame(["Alpha"] * 5)
        training.loc[1, ["longitude", "latitude"]] = [0.0, -2e-08]

        transformed = (
            RadialCategoricalFrequencyFeatureEngineer().fit_transform(training)
        )

        self.assertTrue(np.isfinite(transformed.loc[0, RADIAL_DISTANCE_FEATURE]))
        self.assertGreater(transformed.loc[0, RADIAL_DISTANCE_FEATURE], 3_000.0)
        self.assertTrue(pd.isna(transformed.loc[1, RADIAL_DISTANCE_FEATURE]))

    def test_random_forest_factory_accepts_exact_archive_settings(self) -> None:
        pipeline = make_random_forest_pipeline(
            n_estimators=ARCHIVE_FOREST_ESTIMATORS,
            max_features=ARCHIVE_FOREST_MAX_FEATURES,
        )
        classifier = pipeline.named_steps["classifier"]

        self.assertEqual(classifier.n_estimators, 1000)
        self.assertEqual(classifier.max_features, 5)


def _source_frame(names: list[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        values: dict[str, object] = {
            column: "value" for column in EXPECTED_SOURCE_FEATURES
        }
        values.update(
            {
                "amount_tsh": 10.0,
                "date_recorded": "2013-01-02",
                "gps_height": 100.0,
                "longitude": 31.0,
                "latitude": -6.0,
                "wpt_name": name,
                "num_private": 0.0,
                "region_code": 11,
                "district_code": 4,
                "population": 100.0,
                "construction_year": 2000,
            }
        )
        rows.append(values)
    return pd.DataFrame(rows, columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

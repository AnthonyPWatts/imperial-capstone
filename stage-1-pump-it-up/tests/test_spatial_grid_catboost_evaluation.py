"""Tests for multi-resolution native CatBoost coordinate grids."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_evaluation import DEFERRED_IDENTITY_FEATURES
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import EXPECTED_SOURCE_FEATURES
from spatial_grid_catboost_evaluation import SPATIAL_GRID_FEATURES
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_ACCURACY
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_FOLD_WINS
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_WORST_FOLD
from spatial_grid_catboost_evaluation import SPATIAL_GRID_MISSING_CATEGORY
from spatial_grid_catboost_evaluation import engineer_spatial_grid_catboost_features
from spatial_grid_catboost_evaluation import evaluate_spatial_grid_catboost
from spatial_grid_catboost_evaluation import spatial_grid_policy_signature
from spatial_grid_catboost_evaluation import (
    summarise_spatial_grid_catboost_comparison,
)


class SpatialGridCatBoostEvaluationTests(unittest.TestCase):
    def test_regular_and_half_offset_grids_are_native_categories(self) -> None:
        engineered, categorical = engineer_spatial_grid_catboost_features(
            _source_frame([(31.011, -6.011)])
        )

        self.assertEqual(
            categorical,
            (
                *CATEGORICAL_FEATURES,
                *DEFERRED_IDENTITY_FEATURES,
                *SPATIAL_GRID_FEATURES,
            ),
        )
        self.assertEqual(
            tuple(engineered.columns[-4:]),
            SPATIAL_GRID_FEATURES,
        )
        self.assertEqual(
            engineered.loc[0, "coordinate_grid_0_05_degrees"],
            "620::-121",
        )
        self.assertEqual(
            engineered.loc[0, "coordinate_grid_0_05_degrees_half_offset"],
            "619::-121",
        )
        self.assertEqual(
            engineered.loc[0, "coordinate_grid_0_025_degrees"],
            "1240::-241",
        )
        self.assertEqual(
            engineered.loc[0, "coordinate_grid_0_025_degrees_half_offset"],
            "1239::-241",
        )

    def test_invalid_tanzania_coordinates_use_explicit_missing_category(self) -> None:
        frame = _source_frame(
            [
                (0.0, -0.00000002),
                (27.999, -6.0),
                (31.0, 0.001),
                (None, -6.0),
            ]
        )

        engineered, _ = engineer_spatial_grid_catboost_features(frame)

        self.assertTrue(
            engineered.loc[:, SPATIAL_GRID_FEATURES]
            .eq(SPATIAL_GRID_MISSING_CATEGORY)
            .all()
            .all()
        )

    def test_source_frame_is_not_mutated(self) -> None:
        source = _source_frame([(31.011, -6.011)])
        original = source.copy(deep=True)

        engineer_spatial_grid_catboost_features(source)

        pd.testing.assert_frame_equal(source, original)

    def test_policy_signature_locks_grid_geometry_and_missing_handling(self) -> None:
        signature = spatial_grid_policy_signature()

        self.assertEqual(signature["version"], 1)
        self.assertEqual(
            signature["cells"],
            [
                {
                    "feature": "coordinate_grid_0_05_degrees",
                    "degrees": 0.05,
                    "longitude_offset": 0.0,
                    "latitude_offset": 0.0,
                },
                {
                    "feature": "coordinate_grid_0_05_degrees_half_offset",
                    "degrees": 0.05,
                    "longitude_offset": 0.025,
                    "latitude_offset": 0.025,
                },
                {
                    "feature": "coordinate_grid_0_025_degrees",
                    "degrees": 0.025,
                    "longitude_offset": 0.0,
                    "latitude_offset": 0.0,
                },
                {
                    "feature": "coordinate_grid_0_025_degrees_half_offset",
                    "degrees": 0.025,
                    "longitude_offset": 0.0125,
                    "latitude_offset": 0.0125,
                },
            ],
        )
        self.assertEqual(
            signature["missing_category"],
            SPATIAL_GRID_MISSING_CATEGORY,
        )
        self.assertEqual(
            signature["binning"],
            "floor((coordinate - offset) / degrees)",
        )

    @patch("spatial_grid_catboost_evaluation.evaluate_gpu_candidate")
    def test_evaluation_uses_depth_8_and_spatial_feature_engineer(
        self,
        evaluate_gpu_candidate,
    ) -> None:
        evaluation = object()
        evaluate_gpu_candidate.return_value = evaluation
        partitioned_data = SimpleNamespace(
            X_development=_source_frame(
                [(31.011, -6.011), (31.061, -6.061)]
            )
        )
        cross_validation = _CrossValidation()

        trial = evaluate_spatial_grid_catboost(
            partitioned_data,
            cross_validation,
        )

        spec = evaluate_gpu_candidate.call_args.args[0]
        self.assertEqual(spec.variant, "d8")
        self.assertIn("spatial grids", spec.name)
        self.assertIs(
            evaluate_gpu_candidate.call_args.kwargs[
                "catboost_feature_engineer"
            ],
            engineer_spatial_grid_catboost_features,
        )
        self.assertIs(trial.evaluation, evaluation)
        self.assertEqual(trial.categorical_features, 25)

    def test_summary_reports_paired_fold_deltas_and_boundary_gate_pass(self) -> None:
        identity = _evaluation("complete identity", [0.8] * 5)
        spatial = _evaluation(
            "spatial grid",
            [0.8025, 0.8025, 0.8025, 0.7975, 0.8],
        )

        summary, folds = summarise_spatial_grid_catboost_comparison(
            identity,
            spatial,
        )

        candidate = summary.loc["spatial_grid_catboost"]
        self.assertAlmostEqual(
            candidate["accuracy_change_vs_identity"],
            SPATIAL_GRID_GATE_ACCURACY,
        )
        self.assertEqual(
            candidate["fold_wins_vs_identity"],
            SPATIAL_GRID_GATE_FOLD_WINS,
        )
        self.assertAlmostEqual(
            candidate["worst_fold_change_vs_identity"],
            SPATIAL_GRID_GATE_WORST_FOLD,
        )
        self.assertTrue(candidate["passes_gate"])
        self.assertEqual(
            folds["spatial_grid_wins"].tolist(),
            [True, True, True, False, False],
        )

    def test_gate_rejects_each_failed_requirement(self) -> None:
        identity = _evaluation("complete identity", [0.8] * 5)
        failures = {
            "mean gain": [0.802, 0.802, 0.802, 0.7975, 0.8],
            "fold wins": [0.803, 0.803, 0.8, 0.8, 0.8],
            "worst fold": [0.804, 0.804, 0.804, 0.797, 0.799],
        }

        for name, accuracies in failures.items():
            with self.subTest(name=name):
                summary, _ = summarise_spatial_grid_catboost_comparison(
                    identity,
                    _evaluation("spatial grid", accuracies),
                )
                self.assertFalse(
                    summary.loc["spatial_grid_catboost", "passes_gate"]
                )

    def test_summary_rejects_different_fold_designs(self) -> None:
        identity = _evaluation("complete identity", [0.8] * 5)
        spatial = _evaluation(
            "spatial grid",
            [0.8] * 5,
            fingerprint="different",
        )

        with self.assertRaisesRegex(ValueError, "different cross-validation"):
            summarise_spatial_grid_catboost_comparison(identity, spatial)


class _CrossValidation:
    def split(self):
        yield [0], [1]


def _evaluation(
    model_name: str,
    accuracies: list[float],
    *,
    fingerprint: str = "fresh-full-data-folds",
):
    fold_metrics = pd.DataFrame(
        {"accuracy": accuracies},
        index=pd.Index(range(1, len(accuracies) + 1), name="validation_fold"),
    )
    return SimpleNamespace(
        model_name=model_name,
        cross_validation_fingerprint=fingerprint,
        fold_metrics=fold_metrics,
    )


def _source_frame(
    coordinates: list[tuple[object, object]],
) -> pd.DataFrame:
    rows = []
    for longitude, latitude in coordinates:
        values: dict[str, object] = {
            column: "value" for column in EXPECTED_SOURCE_FEATURES
        }
        values.update(
            {
                "amount_tsh": 10.0,
                "date_recorded": "2013-01-02",
                "gps_height": 100.0,
                "longitude": longitude,
                "latitude": latitude,
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

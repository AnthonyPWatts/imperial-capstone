"""Tests for bounded, fold-fitted training-only outlier removal."""

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
from gpu_model_evaluation import CLASS_LABELS
from model_evaluation import build_candidate_evaluation
from outlier_filtering_evaluation import OUTLIER_FILTER_POLICIES
from outlier_filtering_evaluation import evaluate_outlier_filter
from outlier_filtering_evaluation import fit_training_filter


class _FakeClassifier:
    classes_ = np.asarray(CLASS_LABELS)


class _FakeForest:
    def __init__(self, fit_log: list[set[int]]) -> None:
        self.fit_log = fit_log
        self.named_steps = {"classifier": _FakeClassifier()}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_log.append(set(X.index))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.tile([0.60, 0.10, 0.30], (len(X), 1))


class OutlierFilteringEvaluationTests(unittest.TestCase):
    def test_physical_filter_removes_only_unambiguous_contradictions(self) -> None:
        frame = _source_frame(300)
        target = _balanced_target(300)
        frame.loc[4, "construction_year"] = 2015

        result = fit_training_filter(
            OUTLIER_FILTER_POLICIES["physical_contradictions"],
            frame,
            target,
        )

        self.assertFalse(result.retained.loc[4])
        self.assertEqual(result.diagnostics["removed_rows"], 1)
        self.assertAlmostEqual(result.diagnostics["removed_share"], 1 / 300)

    def test_duplicate_filter_removes_only_the_minority_consensus_label(self) -> None:
        frame = _source_frame(303)
        target = _balanced_target(303)
        frame.loc[:, "amount_tsh"] = np.arange(303, dtype="float64") + 1
        frame.loc[[0, 1, 2], "amount_tsh"] = 10.0
        target.loc[[0, 1, 2]] = [
            "functional",
            "functional",
            "non functional",
        ]

        result = fit_training_filter(
            OUTLIER_FILTER_POLICIES["strict_duplicate_conflicts"],
            frame,
            target,
        )

        self.assertTrue(result.retained.loc[0])
        self.assertTrue(result.retained.loc[1])
        self.assertFalse(result.retained.loc[2])
        self.assertEqual(result.diagnostics["removed_rows"], 1)

    def test_isolation_filter_is_deterministic_and_target_blind(self) -> None:
        frame = _source_frame(300)
        frame.loc[:, "amount_tsh"] = np.linspace(0, 10_000, len(frame))
        frame.loc[:, "population"] = np.arange(len(frame)) + 1
        target = _balanced_target(300)
        permuted = target.sample(frac=1.0, random_state=7)
        permuted.index = target.index
        policy = OUTLIER_FILTER_POLICIES["isolation_forest_010"]

        original = fit_training_filter(policy, frame, target)
        shuffled = fit_training_filter(policy, frame, permuted)

        self.assertEqual(original.diagnostics["removed_rows"], 3)
        pd.testing.assert_series_equal(original.retained, shuffled.retained)

    def test_evaluation_never_filters_or_fits_on_validation_rows(self) -> None:
        rows = 75
        frame = _source_frame(rows)
        frame.loc[0, "construction_year"] = 2015
        target = _balanced_target(rows)
        folds = pd.Series(
            np.tile(np.arange(1, 6), rows // 5),
            name="validation_fold",
        )
        partitioned = PartitionedData(
            development_ids=pd.Series(range(rows)),
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
        cross_validation = PredefinedSplit(test_fold=folds.to_numpy() - 1)
        accepted_xgboost = build_candidate_evaluation(
            model_name="accepted XGBoost",
            partitioned_data=partitioned,
            cross_validation=cross_validation,
            probability_values=np.tile([0.60, 0.10, 0.30], (rows, 1)),
            diagnostic_rows=[
                {"validation_fold": fold, "selected_iterations": 4}
                for fold in range(1, 6)
            ],
        )
        xgboost_fit_log: list[set[int]] = []
        forest_fit_log: list[set[int]] = []
        validation_log: list[set[int]] = []

        def fake_xgboost_fitter(
            spec,
            X_training,
            y_training,
            X_prediction,
            *,
            iterations,
        ):
            xgboost_fit_log.append(set(X_training.index))
            validation_log.append(set(X_prediction.index))
            return np.tile([0.60, 0.10, 0.30], (len(X_prediction), 1)), 0.0

        result = evaluate_outlier_filter(
            OUTLIER_FILTER_POLICIES["physical_contradictions"],
            partitioned,
            cross_validation,
            accepted_xgboost,
            xgboost_fitter=fake_xgboost_fitter,
            random_forest_factory=lambda: _FakeForest(forest_fit_log),
        )

        for xgboost_rows, forest_rows, validation_rows in zip(
            xgboost_fit_log,
            forest_fit_log,
            validation_log,
            strict=True,
        ):
            self.assertTrue(xgboost_rows.isdisjoint(validation_rows))
            self.assertTrue(forest_rows.isdisjoint(validation_rows))
        self.assertFalse(
            result.blend.out_of_fold_probabilities.isna().any().any()
        )
        self.assertEqual(
            result.removal_diagnostics["removed_rows"].tolist(),
            [0, 1, 1, 1, 1],
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
    return pd.DataFrame(
        [defaults.copy() for _ in range(rows)],
        columns=EXPECTED_SOURCE_FEATURES,
    )


def _balanced_target(rows: int) -> pd.Series:
    return pd.Series(
        [CLASS_LABELS[position % len(CLASS_LABELS)] for position in range(rows)]
    )


if __name__ == "__main__":
    unittest.main()

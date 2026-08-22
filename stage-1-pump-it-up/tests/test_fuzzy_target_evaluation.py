"""Focused tests for fold-safe fuzzy target membership evaluation."""

from __future__ import annotations

from dataclasses import replace
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
from fuzzy_target_evaluation import CLASS_LABELS
from fuzzy_target_evaluation import FuzzyMembershipPolicy
from fuzzy_target_evaluation import evaluate_fuzzy_policy
from fuzzy_target_evaluation import evaluate_fuzzy_component_crosses
from fuzzy_target_evaluation import expand_fuzzy_target
from fuzzy_target_evaluation import summarise_memberships
from fuzzy_target_evaluation import triangular_memberships


class _ArrayPreprocessor:
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy(dtype="float64")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy(dtype="float64")


class _FakeClassifier:
    def __init__(
        self,
        fit_records: list[dict[str, object]],
        *,
        iterations: int,
        early_stopping: bool,
    ) -> None:
        self.fit_records = fit_records
        self.n_estimators = iterations
        self.early_stopping = early_stopping
        self.best_iteration = min(iterations, 4) - 1
        self.classes_ = np.arange(3)

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        eval_set=None,
        verbose=None,
    ):
        values = np.asarray(X)
        self.fit_records.append(
            {
                "ids": set(values[:, 0].astype("int64")),
                "rows": len(values),
                "weight": float(np.asarray(sample_weight).sum()),
                "early_stopping": self.early_stopping,
                "has_eval_set": eval_set is not None,
            }
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        hints = np.asarray(X)[:, 1].astype("int64")
        probabilities = np.full((len(hints), 3), 0.05)
        probabilities[np.arange(len(hints)), hints] = 0.90
        return probabilities


class FuzzyTargetEvaluationTests(unittest.TestCase):
    def test_triangular_memberships_are_adjacent_and_normalised(self) -> None:
        policy = FuzzyMembershipPolicy("test", 0.10)
        target = pd.Series(CLASS_LABELS)

        memberships = triangular_memberships(target, policy)

        np.testing.assert_allclose(
            memberships,
            np.asarray(
                [
                    [1 / 1.1, 0.1 / 1.1, 0.0],
                    [0.1 / 1.2, 1 / 1.2, 0.1 / 1.2],
                    [0.0, 0.1 / 1.1, 1 / 1.1],
                ]
            ),
        )
        np.testing.assert_allclose(memberships.sum(axis=1), 1.0)
        self.assertEqual(memberships[0, 2], 0.0)
        self.assertEqual(memberships[2, 0], 0.0)

    def test_weighted_expansion_preserves_each_source_row_mass(self) -> None:
        policy = FuzzyMembershipPolicy("test", 0.05)
        target = pd.Series(list(CLASS_LABELS) * 2)

        expanded = expand_fuzzy_target(target, policy)
        source_weights = np.bincount(
            expanded.source_positions,
            weights=expanded.sample_weights,
            minlength=len(target),
        )
        summary = summarise_memberships(target, policy)

        np.testing.assert_allclose(source_weights, 1.0)
        self.assertEqual(len(expanded.encoded_labels), 14)
        self.assertAlmostEqual(expanded.sample_weights.sum(), len(target))
        self.assertAlmostEqual(summary["effective_share"].sum(), 1.0)

    def test_outer_validation_rows_remain_crisp_and_absent_from_fitting(
        self,
    ) -> None:
        rows_per_fold = 30
        folds = pd.Series(np.repeat(np.arange(1, 6), rows_per_fold))
        encoded = np.tile(np.repeat(np.arange(3), 10), 5)
        target = pd.Series(np.asarray(CLASS_LABELS)[encoded])
        identifiers = np.arange(len(target))
        features = pd.DataFrame({"id": identifiers, "hint": encoded})
        partitioned = PartitionedData(
            development_ids=pd.Series(identifiers),
            X_development=features,
            y_development=target,
            local_test_ids=pd.Series([1000, 1001, 1002]),
            X_local_test=pd.DataFrame(
                {"id": [1000, 1001, 1002], "hint": [0, 1, 2]}
            ),
            y_local_test=pd.Series(CLASS_LABELS),
            validation_folds=folds,
            development_fingerprint="development",
            local_test_fingerprint="local-test",
            cross_validation_fingerprint="cross-validation",
        )
        cross_validation = PredefinedSplit(test_fold=folds.to_numpy() - 1)
        xgboost_records: list[dict[str, object]] = []
        forest_records: list[dict[str, object]] = []

        trial = evaluate_fuzzy_policy(
            FuzzyMembershipPolicy("test", 0.05),
            partitioned,
            cross_validation,
            preprocessor_factory=_ArrayPreprocessor,
            xgboost_factory=lambda iterations, early: _FakeClassifier(
                xgboost_records,
                iterations=iterations,
                early_stopping=early,
            ),
            random_forest_factory=lambda: _FakeClassifier(
                forest_records,
                iterations=7,
                early_stopping=False,
            ),
        )

        self.assertEqual(len(forest_records), 5)
        for fold_number, record in enumerate(forest_records, start=1):
            validation_ids = set(
                identifiers[folds.to_numpy() == fold_number]
            )
            self.assertTrue(record["ids"].isdisjoint(validation_ids))
            self.assertEqual(record["rows"], 280)
            self.assertAlmostEqual(record["weight"], 120.0)
        self.assertAlmostEqual(
            trial.blend.metric_summary.loc["accuracy", "mean"],
            1.0,
        )
        self.assertEqual(trial.blend.confusion_counts.to_numpy().sum(), 150)
        self.assertEqual(
            len(trial.blend.out_of_fold_probabilities),
            len(partitioned.y_development),
        )

        crosses = evaluate_fuzzy_component_crosses(
            partitioned,
            cross_validation,
            hard_xgboost=replace(trial.xgboost, model_name="hard XGBoost"),
            hard_random_forest=replace(
                trial.random_forest,
                model_name="hard Random Forest",
            ),
            trials={"test": trial},
        )
        self.assertEqual(
            set(crosses),
            {
                "test__fuzzy_xgboost_hard_forest",
                "test__hard_xgboost_fuzzy_forest",
            },
        )
        self.assertTrue(
            all(
                len(evaluation.out_of_fold_probabilities) == len(target)
                for evaluation in crosses.values()
            )
        )


if __name__ == "__main__":
    unittest.main()

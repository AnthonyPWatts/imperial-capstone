"""Focused tests for fold-safe regional routing and fallback."""

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
from model_evaluation import build_candidate_evaluation
from regional_specialisation_evaluation import evaluate_regional_specialisation


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


class _FakeRegionalModel:
    def __init__(self, fit_log: list[str]) -> None:
        self.fit_log = fit_log
        self.classes_ = np.asarray(
            ["non functional", "functional", "functional needs repair"]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_log.append(str(X["region"].iloc[0]))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Deliberately return a non-canonical class order.
        return np.tile([0.20, 0.20, 0.60], (len(X), 1))


class RegionalSpecialisationEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        target = []
        folds = []
        for fold in range(1, 6):
            rows.extend([{"region": "A"}] * 3)
            target.extend(CLASS_LABELS)
            rows.extend([{"region": "B"}] * 3)
            target.extend(["functional", "functional", "non functional"])
            folds.extend([fold] * 6)
        self.folds = pd.Series(folds, name="validation_fold")
        self.partitioned = PartitionedData(
            development_ids=pd.Series(range(len(rows))),
            X_development=pd.DataFrame(rows),
            y_development=pd.Series(target),
            local_test_ids=pd.Series(dtype="int64"),
            X_local_test=pd.DataFrame(columns=["region"]),
            y_local_test=pd.Series(dtype="string"),
            validation_folds=self.folds,
            development_fingerprint="development",
            local_test_fingerprint="local-test",
            cross_validation_fingerprint="cross-validation",
        )
        self.cross_validation = PredefinedSplit(
            test_fold=self.folds.to_numpy() - 1
        )
        probabilities = np.tile([0.60, 0.15, 0.25], (len(rows), 1))
        self.accepted = build_candidate_evaluation(
            model_name="accepted",
            partitioned_data=self.partitioned,
            cross_validation=self.cross_validation,
            probability_values=probabilities,
            diagnostic_rows=[
                {"validation_fold": fold} for fold in range(1, 6)
            ],
        )

    def test_ineligible_region_falls_back_and_partial_pooling_is_fixed(self) -> None:
        fit_log: list[str] = []

        result = evaluate_regional_specialisation(
            self.partitioned,
            self.cross_validation,
            self.accepted,
            pipeline_factory=lambda: _FakeRegionalModel(fit_log),
            minimum_training_rows=10,
            minimum_class_rows=4,
            regional_weight=0.20,
            prior_smoothing_levels=(),
            bootstrap_replications=10,
        )

        self.assertEqual(fit_log, ["A"] * 5)
        self.assertTrue(
            result.routing_summary.xs("A", level="region")["eligible"].all()
        )
        self.assertFalse(
            result.routing_summary.xs("B", level="region")["eligible"].any()
        )
        self.assertTrue(
            result.routing_summary.xs("B", level="region")["reason"]
            .eq("too_few_rows_in_at_least_one_class")
            .all()
        )
        self.assertTrue(
            result.partially_pooled_expert.diagnostics["expert_coverage"]
            .eq(0.5)
            .all()
        )

        accepted = self.accepted.out_of_fold_probabilities.to_numpy()
        routed = result.hard_routed_expert.out_of_fold_probabilities.to_numpy()
        pooled = result.partially_pooled_expert.out_of_fold_probabilities.to_numpy()
        region_b = self.partitioned.X_development["region"].eq("B").to_numpy()
        np.testing.assert_allclose(routed[region_b], accepted[region_b])
        np.testing.assert_allclose(pooled, 0.8 * accepted + 0.2 * routed)

    def test_regional_probability_columns_are_aligned_to_canonical_labels(self) -> None:
        result = evaluate_regional_specialisation(
            self.partitioned,
            self.cross_validation,
            self.accepted,
            pipeline_factory=lambda: _FakeRegionalModel([]),
            minimum_training_rows=10,
            minimum_class_rows=4,
            prior_smoothing_levels=(),
            bootstrap_replications=0,
        )
        region_a = self.partitioned.X_development["region"].eq("A").to_numpy()
        probabilities = result.hard_routed_expert.out_of_fold_probabilities.loc[
            region_a
        ]

        np.testing.assert_allclose(probabilities["functional"], 0.20)
        np.testing.assert_allclose(
            probabilities["functional needs repair"],
            0.60,
        )
        np.testing.assert_allclose(probabilities["non functional"], 0.20)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_invalid_regional_weight_is_rejected_before_fitting(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly between"):
            evaluate_regional_specialisation(
                self.partitioned,
                self.cross_validation,
                self.accepted,
                regional_weight=1.0,
                prior_smoothing_levels=(),
            )


if __name__ == "__main__":
    unittest.main()

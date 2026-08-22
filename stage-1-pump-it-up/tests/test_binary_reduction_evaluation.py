"""Focused tests for the two-label training and three-class scoring contract."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binary_reduction_evaluation import BinaryEnsembleSpec
from binary_reduction_evaluation import evaluate_binary_ensembles
from binary_reduction_evaluation import evaluate_binary_pipeline_candidate
from binary_reduction_evaluation import fit_selected_binary_recipe_on_local_test
from binary_reduction_evaluation import score_binary_reduction
from data_partitioning import PartitionedData


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


class _FakeClassifier:
    def __init__(self, fit_targets: list[list[str]]) -> None:
        self.fit_targets = fit_targets
        self.classes_ = np.asarray(["functional", "non functional"])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_targets.append(y.tolist())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        functional = X["signal"].to_numpy(dtype="float64")
        return np.column_stack([functional, 1.0 - functional])


class BinaryReductionEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        folds = pd.Series(np.repeat(np.arange(1, 6), 3))
        target = pd.Series(list(CLASS_LABELS) * 5)
        features = pd.DataFrame(
            {
                "signal": np.tile([0.8, 0.6, 0.2], 5),
            }
        )
        self.partitioned = PartitionedData(
            development_ids=pd.Series(range(15)),
            X_development=features,
            y_development=target,
            local_test_ids=pd.Series([100, 101, 102]),
            X_local_test=pd.DataFrame({"signal": [0.8, 0.6, 0.2]}),
            y_local_test=pd.Series(CLASS_LABELS),
            validation_folds=folds,
            development_fingerprint="development",
            local_test_fingerprint="local-test",
            cross_validation_fingerprint="cross-validation",
        )
        self.cross_validation = PredefinedSplit(test_fold=folds.to_numpy() - 1)

    def test_repair_rows_are_removed_only_from_training(self) -> None:
        fit_targets: list[list[str]] = []

        evaluation = evaluate_binary_pipeline_candidate(
            "fake",
            self.partitioned,
            self.cross_validation,
            pipeline_factory=lambda: _FakeClassifier(fit_targets),
            model_name="fake binary",
        )

        self.assertEqual(len(fit_targets), 5)
        self.assertTrue(
            all(
                "functional needs repair" not in fold_target
                for fold_target in fit_targets
            )
        )
        self.assertTrue(
            evaluation.out_of_fold_probabilities[
                "functional needs repair"
            ].eq(0.0).all()
        )
        self.assertEqual(
            evaluation.confusion_counts.loc[
                "functional needs repair",
                "functional needs repair",
            ],
            0,
        )
        self.assertEqual(
            evaluation.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ],
            0.0,
        )

    def test_full_accuracy_includes_unavoidable_repair_errors(self) -> None:
        probabilities = np.tile([0.8, 0.0, 0.2], (3, 1))
        probabilities[2] = [0.2, 0.0, 0.8]

        metrics = score_binary_reduction(
            pd.Series(CLASS_LABELS),
            probabilities,
        )

        self.assertAlmostEqual(metrics["three_class_accuracy_ceiling"], 2 / 3)
        self.assertAlmostEqual(metrics["three_class_accuracy"], 2 / 3)
        self.assertEqual(metrics["conditional_binary_accuracy"], 1.0)
        self.assertEqual(metrics["repair_recall"], 0.0)
        self.assertEqual(metrics["predicted_repair_share"], 0.0)

    def test_binary_ensemble_preserves_zero_repair_probability(self) -> None:
        fit_targets: list[list[str]] = []
        first = evaluate_binary_pipeline_candidate(
            "first",
            self.partitioned,
            self.cross_validation,
            pipeline_factory=lambda: _FakeClassifier(fit_targets),
            model_name="first",
        )
        second = evaluate_binary_pipeline_candidate(
            "second",
            self.partitioned,
            self.cross_validation,
            pipeline_factory=lambda: _FakeClassifier(fit_targets),
            model_name="second",
        )
        spec = BinaryEnsembleSpec(
            key="pair",
            model_name="binary pair",
            components=("first", "second"),
            weights=(0.60, 0.40),
        )

        result = evaluate_binary_ensembles(
            self.partitioned,
            self.cross_validation,
            {"first": first, "second": second},
            specs=(spec,),
        )["pair"]

        self.assertTrue(
            result.out_of_fold_probabilities[
                "functional needs repair"
            ].eq(0.0).all()
        )
        np.testing.assert_allclose(
            result.out_of_fold_probabilities.sum(axis=1),
            1.0,
        )

    def test_local_test_refit_strips_repair_training_rows_and_scores_all_rows(
        self,
    ) -> None:
        fit_targets: list[list[str]] = []
        development = evaluate_binary_pipeline_candidate(
            "fake",
            self.partitioned,
            self.cross_validation,
            pipeline_factory=lambda: _FakeClassifier([]),
            model_name="fake binary",
        )

        with patch.dict(
            "binary_reduction_evaluation.PIPELINE_FACTORIES",
            {"fake": lambda: _FakeClassifier(fit_targets)},
        ):
            result = fit_selected_binary_recipe_on_local_test(
                self.partitioned,
                selected_key="fake",
                component_evaluations={"fake": development},
                ensemble_specs=(),
            )

        self.assertEqual(len(fit_targets), 1)
        self.assertNotIn("functional needs repair", fit_targets[0])
        self.assertEqual(len(result.predictions), len(self.partitioned.y_local_test))
        self.assertEqual(result.metrics["repair_rows"], 1)
        self.assertEqual(result.metrics["repair_recall"], 0.0)
        self.assertTrue(
            result.probabilities["functional needs repair"].eq(0.0).all()
        )


if __name__ == "__main__":
    unittest.main()

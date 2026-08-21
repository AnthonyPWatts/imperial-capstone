"""Focused tests for weighted and calibrated blend plumbing."""

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

from blend_submission import fit_weighted_vote_for_competition
from data_partitioning import PartitionedData
from final_model import build_competition_prediction
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_equal_weight_soft_vote
from model_evaluation import evaluate_weighted_soft_vote
from model_evaluation import make_calibrated_stack_pipeline
from modelling_data import ModellingData


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


class BlendEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folds = pd.Series(
            np.repeat(np.arange(1, 6), 2),
            name="validation_fold",
        )
        self.target = pd.Series(
            [
                "functional",
                "functional needs repair",
                "non functional",
                "functional",
                "functional needs repair",
                "non functional",
                "functional",
                "functional needs repair",
                "non functional",
                "functional",
            ]
        )
        self.partitioned = PartitionedData(
            development_ids=pd.Series(range(10)),
            X_development=pd.DataFrame(index=range(10)),
            y_development=self.target,
            local_test_ids=pd.Series(dtype="int64"),
            X_local_test=pd.DataFrame(),
            y_local_test=pd.Series(dtype="string"),
            validation_folds=self.folds,
            development_fingerprint="development",
            local_test_fingerprint="local-test",
            cross_validation_fingerprint="cross-validation",
        )
        self.cross_validation = PredefinedSplit(
            test_fold=self.folds.to_numpy() - 1
        )
        first = np.tile([0.7, 0.1, 0.2], (10, 1))
        second = np.tile([0.2, 0.3, 0.5], (10, 1))
        self.first = self._candidate("first", first)
        self.second = self._candidate("second", second)

    def _candidate(
        self,
        name: str,
        probabilities: np.ndarray,
    ) -> CandidateEvaluation:
        return CandidateEvaluation(
            model_name=name,
            cross_validation_fingerprint="cross-validation",
            fold_metrics=pd.DataFrame(),
            metric_summary=pd.DataFrame(),
            diagnostics=pd.DataFrame(
                {"total_seconds": np.ones(5)},
                index=pd.Index(range(1, 6), name="validation_fold"),
            ),
            out_of_fold_probabilities=pd.DataFrame(
                probabilities,
                columns=pd.Index(CLASS_LABELS, name="predicted_class"),
            ),
            confusion_counts=pd.DataFrame(),
            confusion_recall=pd.DataFrame(),
        )

    def test_weighted_vote_uses_explicit_component_weights(self) -> None:
        evaluation = evaluate_weighted_soft_vote(
            self.partitioned,
            self.cross_validation,
            self.first,
            self.second,
            weights=[0.6, 0.4],
            model_name="weighted",
        )

        expected = (
            0.6 * self.first.out_of_fold_probabilities.to_numpy()
            + 0.4 * self.second.out_of_fold_probabilities.to_numpy()
        )
        np.testing.assert_allclose(
            evaluation.out_of_fold_probabilities.to_numpy(),
            expected,
        )
        self.assertTrue(
            evaluation.diagnostics["first weight"].eq(0.6).all()
        )
        self.assertTrue(
            evaluation.diagnostics["second weight"].eq(0.4).all()
        )

    def test_equal_vote_delegates_to_weighted_vote(self) -> None:
        evaluation = evaluate_equal_weight_soft_vote(
            self.partitioned,
            self.cross_validation,
            self.first,
            self.second,
        )
        expected = 0.5 * (
            self.first.out_of_fold_probabilities.to_numpy()
            + self.second.out_of_fold_probabilities.to_numpy()
        )
        np.testing.assert_allclose(
            evaluation.out_of_fold_probabilities.to_numpy(),
            expected,
        )

    def test_weighted_vote_rejects_weights_that_do_not_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to one"):
            evaluate_weighted_soft_vote(
                self.partitioned,
                self.cross_validation,
                self.first,
                self.second,
                weights=[0.6, 0.5],
                model_name="invalid",
            )

    def test_calibrated_stack_declares_inner_cross_fitting(self) -> None:
        classifier = make_calibrated_stack_pipeline().named_steps["classifier"]

        self.assertEqual(classifier.cv.n_splits, 4)
        self.assertEqual(
            [name for name, _ in classifier.estimators],
            ["random_forest", "histogram_boosting"],
        )
        self.assertEqual(classifier.stack_method, "predict_proba")

    def test_competition_vote_rejects_out_of_range_weight_before_fit(self) -> None:
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            fit_weighted_vote_for_competition(
                modelling_data=None,
                submission_template=pd.DataFrame(),
                random_forest_weight=1.1,
            )

    def test_shared_prediction_builder_validates_and_labels_probabilities(self) -> None:
        modelling_data = ModellingData(
            original_ids=pd.Series([1]),
            X_original=pd.DataFrame({"feature": [0]}),
            y_original=pd.Series(["functional"]),
            competition_ids=pd.Series([10, 11]),
            X_competition=pd.DataFrame({"feature": [1, 2]}),
        )
        template = pd.DataFrame(
            {
                "id": [10, 11],
                "status_group": ["", ""],
            }
        )

        prediction = build_competition_prediction(
            modelling_data,
            template,
            np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.2, 0.7],
                ]
            ),
            pd.Series({"component": 1.0}),
        )

        self.assertEqual(
            prediction.submission["status_group"].tolist(),
            ["functional", "non functional"],
        )
        self.assertEqual(prediction.class_shares["functional"], 0.5)
        self.assertEqual(prediction.class_shares["non functional"], 0.5)


if __name__ == "__main__":
    unittest.main()

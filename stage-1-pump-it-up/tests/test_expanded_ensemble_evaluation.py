"""Focused tests for the bounded expanded-voter evaluation."""

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
from expanded_ensemble_evaluation import ExpandedEnsembleSpec
from expanded_ensemble_evaluation import evaluate_expanded_ensembles
from expanded_ensemble_evaluation import probability_quality
from model_evaluation import CandidateEvaluation


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


class ExpandedEnsembleEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        folds = pd.Series(np.repeat(np.arange(1, 6), 2))
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
            validation_folds=folds,
            development_fingerprint="development",
            local_test_fingerprint="local-test",
            cross_validation_fingerprint="cross-validation",
        )
        self.cross_validation = PredefinedSplit(test_fold=folds.to_numpy() - 1)
        self.xgboost = self._candidate(
            "xgboost",
            np.tile([0.70, 0.10, 0.20], (10, 1)),
        )
        self.random_forest = self._candidate(
            "random forest",
            np.tile([0.40, 0.20, 0.40], (10, 1)),
        )
        accepted_values = (
            0.55 * self.xgboost.out_of_fold_probabilities.to_numpy()
            + 0.45 * self.random_forest.out_of_fold_probabilities.to_numpy()
        )
        self.accepted = self._candidate("accepted", accepted_values)
        self.voter = self._candidate(
            "third voter",
            np.tile([0.20, 0.50, 0.30], (10, 1)),
        )

    def _candidate(
        self,
        name: str,
        probabilities: np.ndarray,
    ) -> CandidateEvaluation:
        predictions = np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]
        accuracy = np.mean(predictions == self.target.to_numpy())
        fold_metrics = pd.DataFrame(
            {"accuracy": np.repeat(accuracy, 5)},
            index=pd.Index(range(1, 6), name="validation_fold"),
        )
        metric_summary = pd.DataFrame(
            {
                "mean": [accuracy, 0.25, 0.25, 0.25],
            },
            index=pd.Index(
                [
                    "accuracy",
                    "recall: functional",
                    "recall: functional needs repair",
                    "recall: non functional",
                ],
                name="metric",
            ),
        )
        return CandidateEvaluation(
            model_name=name,
            cross_validation_fingerprint="cross-validation",
            fold_metrics=fold_metrics,
            metric_summary=metric_summary,
            diagnostics=pd.DataFrame(index=range(1, 6)),
            out_of_fold_probabilities=pd.DataFrame(
                probabilities,
                columns=pd.Index(CLASS_LABELS, name="predicted_class"),
            ),
            confusion_counts=pd.DataFrame(),
            confusion_recall=pd.DataFrame(),
        )

    def test_third_voter_preserves_accepted_internal_weight_ratio(self) -> None:
        spec = ExpandedEnsembleSpec(
            key="third",
            label="bounded third voter",
            voter_name=self.voter.model_name,
            voter_weight=0.10,
        )

        result = evaluate_expanded_ensembles(
            self.partitioned,
            self.cross_validation,
            accepted=self.accepted,
            xgboost=self.xgboost,
            random_forest=self.random_forest,
            voters={self.voter.model_name: self.voter},
            specs=(spec,),
        )

        expected = (
            0.495 * self.xgboost.out_of_fold_probabilities.to_numpy()
            + 0.405 * self.random_forest.out_of_fold_probabilities.to_numpy()
            + 0.10 * self.voter.out_of_fold_probabilities.to_numpy()
        )
        np.testing.assert_allclose(
            result.evaluations["third"].out_of_fold_probabilities.to_numpy(),
            expected,
        )
        self.assertAlmostEqual(
            result.candidate_summary.loc["third", "xgboost_weight"],
            0.495,
        )
        self.assertFalse(result.candidate_summary.loc["third", "passes_gate"])

    def test_probability_quality_rejects_rows_that_do_not_sum_to_one(self) -> None:
        probabilities = np.tile([0.3, 0.3, 0.3], (len(self.target), 1))

        with self.assertRaisesRegex(ValueError, "sum to one"):
            probability_quality(self.target, probabilities)

    def test_accepted_recipe_is_validated_before_screening(self) -> None:
        invalid = self._candidate(
            "invalid accepted",
            np.tile([0.2, 0.3, 0.5], (10, 1)),
        )
        spec = ExpandedEnsembleSpec(
            key="third",
            label="bounded third voter",
            voter_name=self.voter.model_name,
            voter_weight=0.10,
        )

        with self.assertRaisesRegex(ValueError, "55:45"):
            evaluate_expanded_ensembles(
                self.partitioned,
                self.cross_validation,
                accepted=invalid,
                xgboost=self.xgboost,
                random_forest=self.random_forest,
                voters={self.voter.model_name: self.voter},
                specs=(spec,),
            )


if __name__ == "__main__":
    unittest.main()

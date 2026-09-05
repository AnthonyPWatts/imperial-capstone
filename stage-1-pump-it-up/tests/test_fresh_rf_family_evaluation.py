"""Focused tests for the locked fresh-fold RF family screen."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from fresh_rf_family_evaluation import CURRENT_RF_VARIANT
from fresh_rf_family_evaluation import LOCKED_MODEL_SEED
from fresh_rf_family_evaluation import LOCKED_RF_VARIANTS
from fresh_rf_family_evaluation import make_full_labelled_partition
from fresh_rf_family_evaluation import make_locked_rf_specs
from fresh_rf_family_evaluation import make_rf_cache_payload
from fresh_rf_family_evaluation import paired_fold_deltas
from fresh_rf_family_evaluation import summarise_rf_screen
from fresh_rf_family_evaluation import validate_rf_cache_payload
from gpu_model_evaluation import SKLEARN_TREE_VARIANTS
from gpu_model_evaluation import resolve_sklearn_tree_parameters
from model_evaluation import CandidateEvaluation
from model_evaluation import RANDOM_FOREST_SEED
from modelling_data import ModellingData


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


class FreshRfFamilyEvaluationTests(unittest.TestCase):
    def test_full_labelled_partition_is_deterministic_and_has_no_holdout(self) -> None:
        modelling_data = self._modelling_data()

        first = make_full_labelled_partition(
            modelling_data,
            cross_validation_seed=101,
        )
        repeated = make_full_labelled_partition(
            modelling_data,
            cross_validation_seed=101,
        )
        different = make_full_labelled_partition(
            modelling_data,
            cross_validation_seed=102,
        )

        self.assertEqual(len(first.y_development), 60)
        self.assertTrue(first.y_local_test.empty)
        self.assertTrue(first.X_local_test.empty)
        self.assertTrue(first.local_test_ids.empty)
        self.assertEqual(set(first.validation_folds), {1, 2, 3, 4, 5})
        self.assertTrue(first.validation_folds.equals(repeated.validation_folds))
        self.assertFalse(first.validation_folds.equals(different.validation_folds))
        self.assertEqual(
            first.cross_validation_fingerprint,
            repeated.cross_validation_fingerprint,
        )
        self.assertNotEqual(
            first.cross_validation_fingerprint,
            different.cross_validation_fingerprint,
        )

    def test_locked_specs_are_exactly_the_incumbent_and_three_challengers(self) -> None:
        specs = make_locked_rf_specs()

        self.assertEqual(tuple(specs), LOCKED_RF_VARIANTS)
        self.assertEqual(LOCKED_MODEL_SEED, RANDOM_FOREST_SEED)
        self.assertTrue(all(spec.seed == LOCKED_MODEL_SEED for spec in specs.values()))
        self.assertEqual(
            SKLEARN_TREE_VARIANTS[CURRENT_RF_VARIANT],
            {
                "kind": "random_forest",
                "n_estimators": 300,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
            },
        )
        self.assertEqual(
            set(LOCKED_RF_VARIANTS[1:]),
            {
                "Extra Trees leaf 2",
                "Random Forest features 0.3",
                "Random Forest features 0.3 leaf 2",
            },
        )
        self.assertEqual(
            resolve_sklearn_tree_parameters(specs[CURRENT_RF_VARIANT]),
            {
                "kind": "random_forest",
                "n_estimators": 300,
                "criterion": "gini",
                "max_features": "sqrt",
                "min_samples_leaf": 1,
                "class_weight": None,
                "n_jobs": 6,
                "random_state": RANDOM_FOREST_SEED,
                "bootstrap": True,
            },
        )

    def test_cache_rejects_changed_resolved_parameters(self) -> None:
        partitioned = make_full_labelled_partition(self._modelling_data())
        spec = make_locked_rf_specs()[CURRENT_RF_VARIANT]
        evaluation = self._candidate_evaluation(
            partitioned.y_development.to_numpy(),
            partitioned.cross_validation_fingerprint,
            model_name=spec.name,
        )
        payload = make_rf_cache_payload(spec, partitioned, evaluation)

        self.assertIs(
            validate_rf_cache_payload(payload, spec, partitioned),
            evaluation,
        )

        stale = dict(payload)
        stale["resolved_model_parameters"] = dict(
            payload["resolved_model_parameters"]
        )
        stale["resolved_model_parameters"]["max_features"] = 0.3
        with self.assertRaisesRegex(ValueError, "resolved_model_parameters"):
            validate_rf_cache_payload(stale, spec, partitioned)

    def test_paired_fold_deltas_count_gains_and_losses(self) -> None:
        partitioned = make_full_labelled_partition(
            self._modelling_data(),
            cross_validation_seed=101,
        )
        cross_validation = make_cross_validation(partitioned)
        actual = partitioned.y_development.to_numpy()
        incumbent_predictions = actual.copy()
        candidate_predictions = actual.copy()
        first_fold = next(cross_validation.split())[1]
        incumbent_predictions[first_fold[0]] = self._different_label(
            actual[first_fold[0]]
        )
        candidate_predictions[first_fold[1]] = self._different_label(
            actual[first_fold[1]]
        )
        candidate_predictions[first_fold[2]] = self._different_label(
            actual[first_fold[2]]
        )
        evaluations = {
            CURRENT_RF_VARIANT: self._evaluation(
                incumbent_predictions,
                partitioned.cross_validation_fingerprint,
            ),
            "challenger": self._evaluation(
                candidate_predictions,
                partitioned.cross_validation_fingerprint,
            ),
        }

        paired = paired_fold_deltas(
            partitioned,
            cross_validation,
            evaluations,
        )
        fold_one = paired.loc[paired["validation_fold"].eq(1)].iloc[0]

        self.assertEqual(fold_one["gained_correct"], 1)
        self.assertEqual(fold_one["lost_correct"], 2)
        self.assertEqual(fold_one["net_additional_correct"], -1)
        self.assertAlmostEqual(
            fold_one["accuracy_delta"],
            -1 / len(first_fold),
        )

    def test_summary_requires_the_full_conservative_gate(self) -> None:
        incumbent = self._summary_evaluation(
            accuracy=0.80,
            accuracy_std=0.01,
            repair_recall=0.30,
        )
        challenger = self._summary_evaluation(
            accuracy=0.802,
            accuracy_std=0.01,
            repair_recall=0.285,
        )
        paired = pd.DataFrame(
            {
                "candidate": ["challenger"] * 5,
                "accuracy_delta": [0.002, 0.002, 0.002, 0.0, -0.002],
                "net_additional_correct": [2, 2, 2, 0, -2],
            }
        )

        summary = summarise_rf_screen(
            {
                CURRENT_RF_VARIANT: incumbent,
                "challenger": challenger,
            },
            paired,
        )

        self.assertTrue(summary.loc["challenger", "passes_gate"])
        self.assertAlmostEqual(
            summary.loc["challenger", "repair_recall_delta"],
            -0.015,
        )

        paired.loc[4, "accuracy_delta"] = -0.003
        failed = summarise_rf_screen(
            {
                CURRENT_RF_VARIANT: incumbent,
                "challenger": challenger,
            },
            paired,
        )
        self.assertFalse(failed.loc["challenger", "passes_gate"])

    @staticmethod
    def _modelling_data() -> ModellingData:
        rows = 60
        return ModellingData(
            original_ids=pd.Series(np.arange(rows) * 7 + 3, name="id"),
            X_original=pd.DataFrame({"feature": np.arange(rows)}),
            y_original=pd.Series(list(CLASS_LABELS) * (rows // 3)),
            competition_ids=pd.Series(dtype="int64", name="id"),
            X_competition=pd.DataFrame({"feature": pd.Series(dtype="int64")}),
        )

    @staticmethod
    def _evaluation(predictions, fingerprint):
        probabilities = np.zeros((len(predictions), len(CLASS_LABELS)))
        positions = {label: index for index, label in enumerate(CLASS_LABELS)}
        for row, prediction in enumerate(predictions):
            probabilities[row, positions[prediction]] = 1.0
        return SimpleNamespace(
            model_name="candidate",
            cross_validation_fingerprint=fingerprint,
            out_of_fold_probabilities=pd.DataFrame(
                probabilities,
                columns=pd.Index(CLASS_LABELS, name="predicted_class"),
            ),
        )

    @staticmethod
    def _summary_evaluation(*, accuracy, accuracy_std, repair_recall):
        return SimpleNamespace(
            metric_summary=pd.DataFrame(
                {
                    "mean": [accuracy, repair_recall],
                    "std": [accuracy_std, 0.02],
                },
                index=["accuracy", "recall: functional needs repair"],
            )
        )

    @staticmethod
    def _candidate_evaluation(predictions, fingerprint, *, model_name):
        simple = FreshRfFamilyEvaluationTests._evaluation(
            predictions,
            fingerprint,
        )
        return CandidateEvaluation(
            model_name=model_name,
            cross_validation_fingerprint=fingerprint,
            fold_metrics=pd.DataFrame(),
            metric_summary=pd.DataFrame(),
            diagnostics=pd.DataFrame(),
            out_of_fold_probabilities=simple.out_of_fold_probabilities,
            confusion_counts=pd.DataFrame(),
            confusion_recall=pd.DataFrame(),
        )

    @staticmethod
    def _different_label(label):
        position = CLASS_LABELS.index(label)
        return CLASS_LABELS[(position + 1) % len(CLASS_LABELS)]


if __name__ == "__main__":
    unittest.main()

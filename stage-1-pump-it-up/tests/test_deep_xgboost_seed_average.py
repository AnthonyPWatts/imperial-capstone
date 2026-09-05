"""Tests for the fixed two-seed deep-XGBoost probability average."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from deep_xgboost_seed_average import AVERAGE_MODEL_NAME
from deep_xgboost_seed_average import classify_seed_average_summary
from deep_xgboost_seed_average import confirm_seed_average
from model_evaluation import build_candidate_evaluation


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


class DeepXgboostSeedAverageTests(unittest.TestCase):
    def test_probability_average_is_exact_and_pair_counts_balance(self) -> None:
        partitioned, cross_validation = self._partition()
        first = self._evaluation(
            partitioned,
            cross_validation,
            "first",
            self._probabilities(0.70, 0.20, 0.10),
        )
        second = self._evaluation(
            partitioned,
            cross_validation,
            "second",
            self._probabilities(0.50, 0.35, 0.15),
        )

        result = confirm_seed_average(
            partitioned,
            cross_validation,
            {
                "seed_20260824": first,
                "seed_20260905": second,
            },
        )

        self.assertEqual(result.average.model_name, AVERAGE_MODEL_NAME)
        np.testing.assert_allclose(
            result.average.out_of_fold_probabilities.to_numpy(),
            0.5
            * (
                first.out_of_fold_probabilities.to_numpy()
                + second.out_of_fold_probabilities.to_numpy()
            ),
        )
        self.assertTrue(
            result.fold_deltas["net_additional_correct"].eq(
                result.fold_deltas["gained_correct"]
                - result.fold_deltas["lost_correct"]
            ).all()
        )
        np.testing.assert_allclose(
            result.fold_deltas["accuracy_delta"],
            result.fold_deltas["net_additional_correct"]
            / result.fold_deltas["rows"],
        )

    def test_tampered_cached_metrics_are_rejected(self) -> None:
        partitioned, cross_validation = self._partition()
        first = self._evaluation(
            partitioned,
            cross_validation,
            "first",
            self._probabilities(0.70, 0.20, 0.10),
        )
        second = self._evaluation(
            partitioned,
            cross_validation,
            "second",
            self._probabilities(0.50, 0.35, 0.15),
        )
        metrics = first.fold_metrics.copy()
        metrics.loc[1, "accuracy"] = 0.0

        with self.assertRaisesRegex(ValueError, "cached fold_metrics"):
            confirm_seed_average(
                partitioned,
                cross_validation,
                {
                    "seed_20260824": replace(first, fold_metrics=metrics),
                    "seed_20260905": second,
                },
            )

    def test_misaligned_fingerprint_is_rejected(self) -> None:
        partitioned, cross_validation = self._partition()
        first = self._evaluation(
            partitioned,
            cross_validation,
            "first",
            self._probabilities(0.70, 0.20, 0.10),
        )
        second = replace(first, cross_validation_fingerprint="different")

        with self.assertRaisesRegex(ValueError, "fold fingerprint"):
            confirm_seed_average(
                partitioned,
                cross_validation,
                {
                    "seed_20260824": first,
                    "seed_20260905": second,
                },
            )

    def test_only_mean_gain_failures_are_a_stable_positive_near_miss(self) -> None:
        summary = pd.DataFrame(
            {
                "mean_accuracy_delta": [0.0005, 0.0004],
                "fold_wins": [3, 4],
                "worst_fold_delta": [-0.0001, -0.0004],
                "repair_recall_delta": [0.0, -0.0005],
                "passes_gate": [False, False],
            },
            index=[
                "average_vs_seed_20260824",
                "average_vs_seed_20260905",
            ],
        )

        failures, passes, near_miss, verdict = (
            classify_seed_average_summary(summary)
        )

        self.assertFalse(passes)
        self.assertTrue(near_miss)
        self.assertEqual(verdict, "stable_positive_near_miss")
        self.assertEqual(
            failures,
            {
                "average_vs_seed_20260824": ("mean_accuracy_gain",),
                "average_vs_seed_20260905": ("mean_accuracy_gain",),
            },
        )

    @staticmethod
    def _partition():
        target = pd.Series(list(CLASS_LABELS) * 10, name="status_group")
        rows = len(target)
        identifiers = pd.Series(range(rows), name="id")
        predictors = pd.DataFrame({"feature": range(rows)})
        folds = pd.Series(
            np.repeat(np.arange(1, 6), rows // 5),
            name="validation_fold",
        )
        partitioned = PartitionedData(
            development_ids=identifiers,
            X_development=predictors,
            y_development=target,
            local_test_ids=identifiers.iloc[0:0].copy(),
            X_local_test=predictors.iloc[0:0].copy(),
            y_local_test=target.iloc[0:0].copy(),
            validation_folds=folds,
            development_fingerprint="development",
            local_test_fingerprint="not-created",
            cross_validation_fingerprint="fresh-folds",
        )
        return partitioned, make_cross_validation(partitioned)

    @staticmethod
    def _probabilities(functional, repair, non_functional):
        values = np.tile([functional, repair, non_functional], (30, 1))
        for row in range(30):
            values[row] = np.roll(values[row], row % 3)
        return values

    @staticmethod
    def _evaluation(
        partitioned,
        cross_validation,
        name,
        probabilities,
    ):
        return build_candidate_evaluation(
            model_name=name,
            partitioned_data=partitioned,
            cross_validation=cross_validation,
            probability_values=probabilities,
            diagnostic_rows=[
                {"validation_fold": fold}
                for fold in range(1, 6)
            ],
        )


if __name__ == "__main__":
    unittest.main()

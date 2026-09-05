"""Tests for the deep-XGBoost frozen-ensemble confirmation."""

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
from deep_xgboost_frozen_ensemble_confirmation import (
    compare_deep_seed_average_ensemble,
)
from deep_xgboost_frozen_ensemble_confirmation import (
    CONFIRMATION_CACHE_VERSION,
)
from deep_xgboost_frozen_ensemble_confirmation import ENSEMBLE_FOLD_WIN_GATE
from deep_xgboost_frozen_ensemble_confirmation import (
    ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
)
from deep_xgboost_frozen_ensemble_confirmation import (
    ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
)
from deep_xgboost_frozen_ensemble_confirmation import (
    ENSEMBLE_WORST_FOLD_DELTA_GATE,
)
from deep_xgboost_frozen_ensemble_confirmation import (
    passes_deep_frozen_ensemble_gate,
)
from deep_xgboost_frozen_ensemble_confirmation import (
    validate_deep_average_component_cache,
)
from deep_xgboost_frozen_ensemble_confirmation import validate_seed_fold_cache
from model_evaluation import build_candidate_evaluation


class DeepXgboostFrozenEnsembleConfirmationTests(unittest.TestCase):
    def test_exact_ensemble_comparison_uses_only_the_half_seed_deep_bag(
        self,
    ) -> None:
        partitioned = _partitioned()
        seed_24_values = _perfect_probabilities(partitioned.y_development)
        seed_05_values = seed_24_values.copy()
        seed_24 = _evaluation(partitioned, "seed 24", seed_24_values)
        seed_05 = _evaluation(partitioned, "seed 05", seed_05_values)
        components = _components(seed_24_values)

        comparison = compare_deep_seed_average_ensemble(
            partitioned,
            components,
            seed_24,
            seed_05,
        )

        np.testing.assert_allclose(
            comparison.deep_seed_average.out_of_fold_probabilities,
            seed_24_values,
        )
        np.testing.assert_allclose(
            comparison.candidate.out_of_fold_probabilities,
            comparison.incumbent.out_of_fold_probabilities,
        )
        row = comparison.summary.iloc[0]
        self.assertEqual(row["mean_accuracy_delta"], 0.0)
        self.assertEqual(row["prediction_disagreements"], 0)
        self.assertFalse(row["passes_gate"])

        metadata = {"pair": [20260824, 20260905], "weights": [0.5, 0.5]}
        replayed = validate_deep_average_component_cache(
            {
                "cache_version": CONFIRMATION_CACHE_VERSION,
                "metadata": metadata,
                "evaluation": comparison.deep_seed_average,
            },
            metadata,
            partitioned,
        )
        np.testing.assert_allclose(
            replayed.out_of_fold_probabilities,
            seed_24_values,
        )

    def test_mismatched_incumbent_deep_slot_is_rejected(self) -> None:
        partitioned = _partitioned()
        values = _perfect_probabilities(partitioned.y_development)
        evaluation = _evaluation(partitioned, "seed", values)
        components = _components(values)
        components["deep_xgboost"] = np.roll(values, 1, axis=1)

        with self.assertRaisesRegex(ValueError, "incumbent deep slot"):
            compare_deep_seed_average_ensemble(
                partitioned,
                components,
                evaluation,
                evaluation,
            )

    def test_exact_gate_boundaries_pass(self) -> None:
        self.assertTrue(
            passes_deep_frozen_ensemble_gate(
                mean_accuracy_delta=ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
                fold_wins=ENSEMBLE_FOLD_WIN_GATE,
                worst_fold_delta=ENSEMBLE_WORST_FOLD_DELTA_GATE,
                repair_recall_delta=ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
            )
        )

    def test_each_gate_failure_rejects(self) -> None:
        passing = {
            "mean_accuracy_delta": ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
            "fold_wins": ENSEMBLE_FOLD_WIN_GATE,
            "worst_fold_delta": ENSEMBLE_WORST_FOLD_DELTA_GATE,
            "repair_recall_delta": ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
        }
        failures = (
            {"mean_accuracy_delta": ENSEMBLE_MEAN_ACCURACY_GAIN_GATE - 1e-5},
            {"fold_wins": ENSEMBLE_FOLD_WIN_GATE - 1},
            {"worst_fold_delta": ENSEMBLE_WORST_FOLD_DELTA_GATE - 1e-5},
            {
                "repair_recall_delta": (
                    ENSEMBLE_REPAIR_RECALL_DELTA_GATE - 1e-5
                )
            },
        )
        for failure in failures:
            with self.subTest(failure=failure):
                self.assertFalse(
                    passes_deep_frozen_ensemble_gate(
                        **{**passing, **failure}
                    )
                )

    def test_fold_cache_requires_exact_metadata_ids_and_shape(self) -> None:
        metadata = {"recipe": "locked", "fold": 1}
        identifiers = np.asarray([10, 20])
        payload = {
            "cache_version": CONFIRMATION_CACHE_VERSION,
            "metadata": metadata,
            "validation_ids": identifiers.copy(),
            "probabilities": np.asarray(
                [[0.7, 0.1, 0.2], [0.2, 0.3, 0.5]]
            ),
            "fit_and_predict_seconds": 1.0,
        }
        self.assertIs(
            validate_seed_fold_cache(payload, metadata, identifiers),
            payload,
        )

        with self.assertRaisesRegex(ValueError, "metadata is stale"):
            validate_seed_fold_cache(
                {**payload, "metadata": {"fold": 2}},
                metadata,
                identifiers,
            )
        with self.assertRaisesRegex(ValueError, "misaligned"):
            validate_seed_fold_cache(
                {**payload, "validation_ids": identifiers[::-1]},
                metadata,
                identifiers,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_seed_fold_cache(
                {**payload, "probabilities": np.asarray([[0.7, 0.3]])},
                metadata,
                identifiers,
            )

    def test_tampered_seed_metrics_are_replayed_and_rejected(self) -> None:
        partitioned = _partitioned()
        values = _perfect_probabilities(partitioned.y_development)
        seed = _evaluation(partitioned, "seed", values)
        metrics = seed.fold_metrics.copy()
        metrics.loc[1, "accuracy"] = 0.0

        with self.assertRaisesRegex(ValueError, "OOF evidence"):
            compare_deep_seed_average_ensemble(
                partitioned,
                _components(values),
                replace(seed, fold_metrics=metrics),
                seed,
            )


def _partitioned() -> PartitionedData:
    labels = (
        "functional",
        "functional needs repair",
        "non functional",
    )
    target = pd.Series(labels * 5, name="status_group")
    rows = len(target)
    identifiers = pd.Series(range(rows), name="id")
    predictors = pd.DataFrame({"feature": range(rows)})
    return PartitionedData(
        development_ids=identifiers,
        X_development=predictors,
        y_development=target,
        local_test_ids=identifiers.iloc[0:0].copy(),
        X_local_test=predictors.iloc[0:0].copy(),
        y_local_test=target.iloc[0:0].copy(),
        validation_folds=pd.Series(
            np.repeat(np.arange(1, 6), 3),
            name="validation_fold",
        ),
        development_fingerprint="development",
        local_test_fingerprint="not-opened",
        cross_validation_fingerprint="old-frozen-folds",
    )


def _perfect_probabilities(target: pd.Series) -> np.ndarray:
    labels = (
        "functional",
        "functional needs repair",
        "non functional",
    )
    positions = {label: index for index, label in enumerate(labels)}
    probabilities = np.full((len(target), 3), 0.05)
    probabilities[
        np.arange(len(target)),
        [positions[label] for label in target],
    ] = 0.90
    return probabilities


def _evaluation(
    partitioned: PartitionedData,
    name: str,
    probabilities: np.ndarray,
):
    return build_candidate_evaluation(
        model_name=name,
        partitioned_data=partitioned,
        cross_validation=make_cross_validation(partitioned),
        probability_values=probabilities,
        diagnostic_rows=[
            {"validation_fold": fold}
            for fold in range(1, 6)
        ],
    )


def _components(deep: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "deep_xgboost": deep,
        "spatial_xgboost": deep.copy(),
        "random_forest": deep.copy(),
        "frequency_random_forest": deep.copy(),
        "spatial_random_forest": deep.copy(),
        "identity_catboost": deep.copy(),
    }


if __name__ == "__main__":
    unittest.main()

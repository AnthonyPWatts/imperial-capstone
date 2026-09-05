"""Tests for the fixed identity/grid CatBoost probability-bag replay."""

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
from catboost_identity_evaluation import CatBoostIdentityTrial
from locked_catboost_probability_bag import CATBOOST_BAG_CANDIDATE
from locked_catboost_probability_bag import CATBOOST_OLD_FOLD_CACHE_VERSION
from locked_catboost_probability_bag import compare_fresh_catboost_probability_bag
from locked_catboost_probability_bag import compare_locked_catboost_bag_ensemble
from locked_catboost_probability_bag import make_equal_catboost_probability_bag
from locked_catboost_probability_bag import passes_fresh_catboost_bag_gate
from locked_catboost_probability_bag import validate_old_fold_spatial_grid_cache
from model_evaluation import build_candidate_evaluation
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_ACCURACY
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_FOLD_WINS
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_WORST_FOLD


class LockedCatBoostProbabilityBagTests(unittest.TestCase):
    def test_probability_bag_uses_exact_half_weights(self) -> None:
        identity = np.asarray([[0.6, 0.1, 0.3]])
        spatial = np.asarray([[0.2, 0.3, 0.5]])

        bag = make_equal_catboost_probability_bag(identity, spatial)

        np.testing.assert_allclose(bag, 0.5 * identity + 0.5 * spatial)

    def test_existing_spatial_gate_boundaries_pass(self) -> None:
        self.assertTrue(
            passes_fresh_catboost_bag_gate(
                mean_accuracy_delta=SPATIAL_GRID_GATE_ACCURACY,
                fold_wins=SPATIAL_GRID_GATE_FOLD_WINS,
                worst_fold_delta=SPATIAL_GRID_GATE_WORST_FOLD,
            )
        )

    def test_each_failed_gate_requirement_rejects(self) -> None:
        passing = {
            "mean_accuracy_delta": SPATIAL_GRID_GATE_ACCURACY,
            "fold_wins": SPATIAL_GRID_GATE_FOLD_WINS,
            "worst_fold_delta": SPATIAL_GRID_GATE_WORST_FOLD,
        }
        failures = (
            {"mean_accuracy_delta": SPATIAL_GRID_GATE_ACCURACY - 1e-6},
            {"fold_wins": SPATIAL_GRID_GATE_FOLD_WINS - 1},
            {"worst_fold_delta": SPATIAL_GRID_GATE_WORST_FOLD - 1e-6},
        )
        for failure in failures:
            with self.subTest(failure=failure):
                self.assertFalse(
                    passes_fresh_catboost_bag_gate(
                        **{**passing, **failure}
                    )
                )

    def test_replay_reports_fold_and_repair_metrics(self) -> None:
        partitioned, identity = _evidence()

        comparison = compare_fresh_catboost_probability_bag(
            partitioned,
            identity,
            identity,
        )

        row = comparison.summary.loc[CATBOOST_BAG_CANDIDATE]
        self.assertEqual(row["mean_accuracy_delta"], 0.0)
        self.assertEqual(row["repair_recall_delta"], 0.0)
        self.assertFalse(row["passes_gate"])
        self.assertEqual(len(comparison.fold_deltas), 5)
        self.assertIn("repair_recall_delta", comparison.fold_deltas)

    def test_tampered_cached_metrics_are_rejected(self) -> None:
        partitioned, identity = _evidence()
        metrics = identity.fold_metrics.copy()
        metrics.loc[1, "accuracy"] = 0.5

        with self.assertRaisesRegex(ValueError, "cached fold_metrics"):
            compare_fresh_catboost_probability_bag(
                partitioned,
                identity,
                replace(identity, fold_metrics=metrics),
            )

    def test_old_fold_bag_is_compared_inside_locked_ensemble(self) -> None:
        partitioned, identity = _evidence()
        probabilities = identity.out_of_fold_probabilities.to_numpy()
        components = {
            name: probabilities.copy()
            for name in (
                "deep_xgboost",
                "spatial_xgboost",
                "random_forest",
                "frequency_random_forest",
                "spatial_random_forest",
                "identity_catboost",
            )
        }

        comparison = compare_locked_catboost_bag_ensemble(
            partitioned,
            components,
            probabilities,
        )

        row = comparison.summary.loc[CATBOOST_BAG_CANDIDATE]
        self.assertEqual(row["mean_accuracy_delta"], 0.0)
        self.assertEqual(row["repair_recall_delta"], 0.0)
        self.assertFalse(row["passes_gate"])
        self.assertEqual(len(comparison.fold_deltas), 5)

    def test_old_fold_cache_rejects_stale_recipe(self) -> None:
        partitioned, evaluation = _evidence()
        metadata = {"recipe": "locked", "rows": len(partitioned.y_development)}
        payload = _old_fold_payload(evaluation, metadata)

        with self.assertRaisesRegex(ValueError, "metadata is stale"):
            validate_old_fold_spatial_grid_cache(
                payload,
                {**metadata, "recipe": "changed"},
                partitioned,
            )

    def test_old_fold_cache_recomputes_and_rejects_tampered_metrics(self) -> None:
        partitioned, evaluation = _evidence()
        metadata = {"recipe": "locked", "rows": len(partitioned.y_development)}
        metrics = evaluation.fold_metrics.copy()
        metrics.loc[1, "accuracy"] = 0.5
        payload = _old_fold_payload(
            replace(evaluation, fold_metrics=metrics),
            metadata,
        )

        with self.assertRaisesRegex(ValueError, "cached fold_metrics"):
            validate_old_fold_spatial_grid_cache(
                payload,
                metadata,
                partitioned,
            )


def _evidence():
    labels = (
        "functional",
        "functional needs repair",
        "non functional",
    )
    target = pd.Series(labels * 5, name="status_group")
    rows = len(target)
    identifiers = pd.Series(range(rows), name="id")
    X = pd.DataFrame(index=target.index)
    partitioned = PartitionedData(
        development_ids=identifiers,
        X_development=X,
        y_development=target,
        local_test_ids=identifiers.iloc[0:0].copy(),
        X_local_test=X.iloc[0:0].copy(),
        y_local_test=target.iloc[0:0].copy(),
        validation_folds=pd.Series(
            np.repeat(np.arange(1, 6), 3),
            name="validation_fold",
        ),
        development_fingerprint="development",
        local_test_fingerprint="not-created",
        cross_validation_fingerprint="fresh-folds",
    )
    probabilities = np.zeros((rows, len(labels)), dtype="float64")
    probabilities[np.arange(rows), np.tile(np.arange(3), 5)] = 1.0
    evaluation = build_candidate_evaluation(
        model_name="candidate",
        partitioned_data=partitioned,
        cross_validation=make_cross_validation(partitioned),
        probability_values=probabilities,
        diagnostic_rows=[
            {"validation_fold": fold, "value": 1.0}
            for fold in range(1, 6)
        ],
    )
    return partitioned, evaluation


def _old_fold_payload(evaluation, metadata):
    return {
        "cache_version": CATBOOST_OLD_FOLD_CACHE_VERSION,
        "metadata": metadata,
        "trial": CatBoostIdentityTrial(
            evaluation=evaluation,
            engineered_features=42,
            categorical_features=17,
        ),
    }


if __name__ == "__main__":
    unittest.main()

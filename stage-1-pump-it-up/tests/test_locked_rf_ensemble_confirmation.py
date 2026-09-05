"""Tests for the locked RF exact-ensemble confirmation."""

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
from locked_rf_ensemble_confirmation import CONFIRMATION_CACHE_VERSION
from locked_rf_ensemble_confirmation import ENSEMBLE_FOLD_WIN_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_MEAN_ACCURACY_GAIN_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_REPAIR_RECALL_DELTA_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_WORST_FOLD_DELTA_GATE
from locked_rf_ensemble_confirmation import ACCURACY_FIRST_RF_CHALLENGER
from locked_rf_ensemble_confirmation import FORMAL_RF_CHALLENGER
from locked_rf_ensemble_confirmation import make_equal_rf_probability_bags
from locked_rf_ensemble_confirmation import passes_locked_rf_ensemble_gate
from locked_rf_ensemble_confirmation import validate_confirmation_cache_payload
from model_evaluation import build_candidate_evaluation


class LockedRfEnsembleConfirmationTests(unittest.TestCase):
    def test_equal_probability_bags_use_fixed_half_weights(self) -> None:
        incumbent = np.asarray([[0.6, 0.1, 0.3]])
        formal = np.asarray([[0.4, 0.2, 0.4]])
        accuracy_first = np.asarray([[0.2, 0.3, 0.5]])

        bags = make_equal_rf_probability_bags(
            incumbent,
            {
                FORMAL_RF_CHALLENGER: formal,
                ACCURACY_FIRST_RF_CHALLENGER: accuracy_first,
            },
        )

        np.testing.assert_allclose(
            bags[FORMAL_RF_CHALLENGER],
            0.5 * incumbent + 0.5 * formal,
        )
        np.testing.assert_allclose(
            bags[ACCURACY_FIRST_RF_CHALLENGER],
            0.5 * incumbent + 0.5 * accuracy_first,
        )

    def test_probability_bags_reject_an_unplanned_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate set changed"):
            make_equal_rf_probability_bags(
                np.asarray([[0.6, 0.1, 0.3]]),
                {"unplanned": np.asarray([[0.4, 0.2, 0.4]])},
            )

    def test_exact_gate_boundaries_pass(self) -> None:
        self.assertTrue(
            passes_locked_rf_ensemble_gate(
                mean_accuracy_delta=ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
                fold_wins=ENSEMBLE_FOLD_WIN_GATE,
                worst_fold_delta=ENSEMBLE_WORST_FOLD_DELTA_GATE,
                repair_recall_delta=ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
            )
        )

    def test_each_failed_gate_requirement_rejects(self) -> None:
        cases = {
            "mean accuracy": {
                "mean_accuracy_delta": ENSEMBLE_MEAN_ACCURACY_GAIN_GATE - 1e-6,
            },
            "fold wins": {"fold_wins": ENSEMBLE_FOLD_WIN_GATE - 1},
            "worst fold": {
                "worst_fold_delta": ENSEMBLE_WORST_FOLD_DELTA_GATE - 1e-6,
            },
            "repair recall": {
                "repair_recall_delta": ENSEMBLE_REPAIR_RECALL_DELTA_GATE - 1e-6,
            },
        }
        passing = {
            "mean_accuracy_delta": ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
            "fold_wins": ENSEMBLE_FOLD_WIN_GATE,
            "worst_fold_delta": ENSEMBLE_WORST_FOLD_DELTA_GATE,
            "repair_recall_delta": ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
        }
        for name, change in cases.items():
            with self.subTest(name=name):
                self.assertFalse(
                    passes_locked_rf_ensemble_gate(**{**passing, **change})
                )

    def test_stale_recipe_or_evidence_metadata_is_rejected(self) -> None:
        partitioned, evaluation = _evidence()
        expected = {
            "recipe": {"seed": 20260821, "max_features": 0.3},
            "evidence_sha256": {"screen": "abc", "incumbent": "def"},
        }
        cases = (
            {
                "recipe": {"seed": 7, "max_features": 0.3},
                "evidence_sha256": expected["evidence_sha256"],
            },
            {
                "recipe": expected["recipe"],
                "evidence_sha256": {"screen": "changed", "incumbent": "def"},
            },
        )
        for metadata in cases:
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "metadata is stale"):
                    validate_confirmation_cache_payload(
                        {
                            "cache_version": CONFIRMATION_CACHE_VERSION,
                            "metadata": metadata,
                            "evaluation": evaluation,
                        },
                        expected,
                        partitioned,
                        candidate="candidate",
                    )

    def test_tampered_cached_oof_evidence_is_rejected(self) -> None:
        partitioned, evaluation = _evidence()
        probabilities = evaluation.out_of_fold_probabilities.copy()
        probabilities.iloc[0] = [0.0, 1.0, 0.0]
        payload = {
            "cache_version": CONFIRMATION_CACHE_VERSION,
            "metadata": {"recipe": "locked"},
            "evaluation": replace(
                evaluation,
                out_of_fold_probabilities=probabilities,
            ),
        }

        with self.assertRaisesRegex(ValueError, "OOF evidence"):
            validate_confirmation_cache_payload(
                payload,
                payload["metadata"],
                partitioned,
                candidate="candidate",
            )

    def test_confirmation_cache_replays_untampered_metrics(self) -> None:
        partitioned, evaluation = _evidence()
        metadata = {"recipe": "locked", "evidence": "locked"}

        recomputed = validate_confirmation_cache_payload(
            {
                "cache_version": CONFIRMATION_CACHE_VERSION,
                "metadata": metadata,
                "evaluation": evaluation,
            },
            metadata,
            partitioned,
            candidate="candidate",
        )

        self.assertEqual(recomputed.metric_summary.loc["accuracy", "mean"], 1.0)


def _evidence():
    labels = (
        "functional",
        "functional needs repair",
        "non functional",
    )
    y = pd.Series(labels * 5, name="status_group")
    rows = len(y)
    identifiers = pd.Series(range(rows), name="id")
    X = pd.DataFrame({"feature": range(rows)})
    partitioned = PartitionedData(
        development_ids=identifiers,
        X_development=X,
        y_development=y,
        local_test_ids=identifiers.iloc[0:0].copy(),
        X_local_test=X.iloc[0:0].copy(),
        y_local_test=y.iloc[0:0].copy(),
        validation_folds=pd.Series(
            np.repeat(np.arange(1, 6), 3),
            name="validation_fold",
        ),
        development_fingerprint="development",
        local_test_fingerprint="not-opened",
        cross_validation_fingerprint="locked-folds",
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


if __name__ == "__main__":
    unittest.main()

"""Tests for replaying locked screens from cached OOF probabilities."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from fresh_rf_family_evaluation import CURRENT_RF_VARIANT
from fresh_rf_family_evaluation import FOLD_WIN_GATE
from fresh_rf_family_evaluation import FRESH_CROSS_VALIDATION_SEED
from fresh_rf_family_evaluation import LOCKED_MODEL_SEED
from fresh_rf_family_evaluation import LOCKED_RF_VARIANTS
from fresh_rf_family_evaluation import MEAN_ACCURACY_GAIN_GATE
from fresh_rf_family_evaluation import REPAIR_RECALL_DELTA_GATE
from fresh_rf_family_evaluation import RF_SCREEN_CACHE_VERSION
from fresh_rf_family_evaluation import WORST_FOLD_DELTA_GATE
from fresh_rf_family_evaluation import make_locked_rf_specs
from fresh_rf_family_evaluation import make_rf_cache_payload
from locked_screen_evidence import FRESH_FOLD_FINGERPRINT
from locked_screen_evidence import load_rf_screen_evidence
from locked_screen_evidence import recompute_and_validate_oof_evaluation
from locked_architecture_candidates import sha256_file
from model_evaluation import build_candidate_evaluation


class LockedScreenEvidenceTests(unittest.TestCase):
    def test_rf_loader_replays_all_caches_before_returning(self) -> None:
        partitioned, _, evaluation = _evidence()
        partitioned = replace(
            partitioned,
            cross_validation_fingerprint=FRESH_FOLD_FINGERPRINT,
        )
        specs = make_locked_rf_specs()
        result = {
            "cache_version": RF_SCREEN_CACHE_VERSION,
            "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
            "model_seed": LOCKED_MODEL_SEED,
            "folds": 5,
            "labelled_rows": len(partitioned.y_development),
            "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
            "incumbent": CURRENT_RF_VARIANT,
            "locked_candidates": list(LOCKED_RF_VARIANTS),
            "locked_specs": {
                name: asdict(spec) for name, spec in specs.items()
            },
            "gate_thresholds": {
                "mean_accuracy_gain": MEAN_ACCURACY_GAIN_GATE,
                "minimum_fold_wins": FOLD_WIN_GATE,
                "worst_fold_delta": WORST_FOLD_DELTA_GATE,
                "minimum_repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
            },
            "passes_gate": False,
            "selected_candidate": CURRENT_RF_VARIANT,
            "selected_candidate_is_challenger": False,
            "old_local_test_scored_separately": False,
            "competition_predictions_generated": False,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "result.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            (directory / "candidate-summary.csv").write_text(
                "candidate\n",
                encoding="utf-8",
            )
            (directory / "fold-paired-deltas.csv").write_text(
                "candidate\n",
                encoding="utf-8",
            )
            provenance_records = {}
            for name, spec in specs.items():
                candidate_evaluation = replace(
                    evaluation,
                    model_name=spec.name,
                    cross_validation_fingerprint=FRESH_FOLD_FINGERPRINT,
                )
                payload = make_rf_cache_payload(
                    spec,
                    partitioned,
                    candidate_evaluation,
                )
                filename = (
                    name.casefold()
                    .replace(" ", "-")
                    .replace(".", "-")
                )
                versioned_path = directory / f"{filename}.joblib"
                bare_path = directory / f"{filename}.bare-v0.joblib"
                joblib.dump(payload, versioned_path)
                joblib.dump(candidate_evaluation, bare_path)
                provenance_records[name] = {
                    "bare_cache": bare_path.name,
                    "bare_sha256": sha256_file(bare_path),
                    "versioned_cache": versioned_path.name,
                    "versioned_sha256": sha256_file(versioned_path),
                }
            (directory / "cache-provenance.json").write_text(
                json.dumps(
                    {
                        "transition": (
                            "Same-run bare CandidateEvaluation wrapped after "
                            "cache contract was hardened"
                        ),
                        "metadata_attached_after_training": True,
                        "wrapped_at_utc": "2026-09-05T00:00:00+00:00",
                        "cross_validation_fingerprint": (
                            FRESH_FOLD_FINGERPRINT
                        ),
                        "caches": provenance_records,
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "locked_screen_evidence.make_full_labelled_partition",
                return_value=partitioned,
            ):
                loaded = load_rf_screen_evidence(
                    directory,
                    SimpleNamespace(y_original=partitioned.y_development),
                )
                first_bare = directory / "current-random-forest.bare-v0.joblib"
                first_bare.write_bytes(b"tampered")
                with self.assertRaisesRegex(
                    ValueError,
                    "RF cache provenance changed",
                ):
                    load_rf_screen_evidence(
                        directory,
                        SimpleNamespace(
                            y_original=partitioned.y_development
                        ),
                    )

        self.assertEqual(loaded.spec, specs[CURRENT_RF_VARIANT])
        self.assertEqual(loaded.iterations, 300)
        self.assertEqual(
            set(loaded.hashes),
            {
                "result",
                "summary",
                "folds",
                "provenance",
                *(f"evaluation:{name}" for name in LOCKED_RF_VARIANTS),
                *(f"bare:{name}" for name in LOCKED_RF_VARIANTS),
            },
        )

    def test_valid_metrics_are_recomputed_from_oof_rows(self) -> None:
        partitioned, cross_validation, evaluation = _evidence()

        recomputed = recompute_and_validate_oof_evaluation(
            evaluation,
            partitioned,
            cross_validation,
            candidate="candidate",
        )

        self.assertTrue(recomputed.fold_metrics["accuracy"].eq(1.0).all())
        self.assertEqual(
            recomputed.metric_summary.loc["accuracy", "mean"],
            1.0,
        )

    def test_tampered_cached_fold_metrics_are_rejected(self) -> None:
        partitioned, cross_validation, evaluation = _evidence()
        fold_metrics = evaluation.fold_metrics.copy()
        fold_metrics.loc[1, "accuracy"] = 0.5

        with self.assertRaisesRegex(ValueError, "cached fold_metrics"):
            recompute_and_validate_oof_evaluation(
                replace(evaluation, fold_metrics=fold_metrics),
                partitioned,
                cross_validation,
                candidate="candidate",
            )

    def test_tampered_cached_summary_is_rejected(self) -> None:
        partitioned, cross_validation, evaluation = _evidence()
        summary = evaluation.metric_summary.copy()
        summary.loc["accuracy", "mean"] = 0.5

        with self.assertRaisesRegex(ValueError, "cached metric_summary"):
            recompute_and_validate_oof_evaluation(
                replace(evaluation, metric_summary=summary),
                partitioned,
                cross_validation,
                candidate="candidate",
            )

    def test_tampered_oof_probabilities_are_rejected_by_replay(self) -> None:
        partitioned, cross_validation, evaluation = _evidence()
        probabilities = evaluation.out_of_fold_probabilities.copy()
        probabilities.iloc[0] = [0.0, 1.0, 0.0]

        with self.assertRaisesRegex(ValueError, "does not match its OOF evidence"):
            recompute_and_validate_oof_evaluation(
                replace(
                    evaluation,
                    out_of_fold_probabilities=probabilities,
                ),
                partitioned,
                cross_validation,
                candidate="candidate",
            )


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
    folds = pd.Series(
        np.repeat(np.arange(1, 6), 3),
        name="validation_fold",
    )
    partitioned = PartitionedData(
        development_ids=identifiers,
        X_development=X,
        y_development=y,
        local_test_ids=identifiers.iloc[0:0].copy(),
        X_local_test=X.iloc[0:0].copy(),
        y_local_test=y.iloc[0:0].copy(),
        validation_folds=folds,
        development_fingerprint="development",
        local_test_fingerprint="not-created",
        cross_validation_fingerprint="locked-folds",
    )
    cross_validation = make_cross_validation(partitioned)
    probabilities = np.zeros((rows, len(labels)), dtype="float64")
    probabilities[np.arange(rows), np.tile(np.arange(3), 5)] = 1.0
    evaluation = build_candidate_evaluation(
        model_name="candidate",
        partitioned_data=partitioned,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=[
            {"validation_fold": fold, "value": 1.0}
            for fold in range(1, 6)
        ],
    )
    return partitioned, cross_validation, evaluation


if __name__ == "__main__":
    unittest.main()

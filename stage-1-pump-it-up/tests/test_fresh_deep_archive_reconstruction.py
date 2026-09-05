"""Tests for strict fresh-fold deep-archive reconstruction."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from deep_xgboost_ten_seed_bag import TEN_BAG_NAME
from fresh_deep_archive_reconstruction import assemble_fold_evaluation
from fresh_deep_archive_reconstruction import CATBOOST_IDENTITY_GRID_BAG
from fresh_deep_archive_reconstruction import COMBINED_FALLBACK_CANDIDATE
from fresh_deep_archive_reconstruction import combine_fresh_deep_archive_fallback
from fresh_deep_archive_reconstruction import compare_fresh_deep_archive_candidates
from fresh_deep_archive_reconstruction import (
    compare_fresh_deep_ten_seed_direct_replacement,
)
from fresh_deep_archive_reconstruction import DEEP_TEN_SEED_DIRECT_REPLACEMENT
from fresh_deep_archive_reconstruction import DEEP_SEED_BAG
from fresh_deep_archive_reconstruction import DEEP_SEED_BAG_COMPONENT
from fresh_deep_archive_reconstruction import ENSEMBLE_FOLD_WIN_GATE
from fresh_deep_archive_reconstruction import ENSEMBLE_MEAN_ACCURACY_GAIN_GATE
from fresh_deep_archive_reconstruction import ENSEMBLE_REPAIR_RECALL_DELTA_GATE
from fresh_deep_archive_reconstruction import ENSEMBLE_WORST_FOLD_DELTA_GATE
from fresh_deep_archive_reconstruction import FRESH_CROSS_VALIDATION_FINGERPRINT
from fresh_deep_archive_reconstruction import FRESH_CROSS_VALIDATION_FOLDS
from fresh_deep_archive_reconstruction import FRESH_CROSS_VALIDATION_SEED
from fresh_deep_archive_reconstruction import FRESH_LABELLED_ROWS
from fresh_deep_archive_reconstruction import LOCKED_ALTERNATIVE_COMPONENT_KEYS
from fresh_deep_archive_reconstruction import LOCKED_FRESH_CANDIDATES
from fresh_deep_archive_reconstruction import make_fold_cache_payload
from fresh_deep_archive_reconstruction import (
    passes_fresh_deep_archive_exploratory_guard,
)
from fresh_deep_archive_reconstruction import passes_fresh_deep_archive_gate
from fresh_deep_archive_reconstruction import RF_CURRENT_LEAF_1_BAG
from fresh_deep_archive_reconstruction import RF_CURRENT_LEAF_2_BAG
from fresh_deep_archive_reconstruction import RF_FEATURES_0_3_COMPONENT
from fresh_deep_archive_reconstruction import RF_FEATURES_0_3_LEAF_2_COMPONENT
from fresh_deep_archive_reconstruction import SPATIAL_GRID_COMPONENT
from fresh_deep_archive_reconstruction import validate_fold_cache_payload
from model_evaluation import build_candidate_evaluation


class FreshDeepArchiveReconstructionTests(unittest.TestCase):
    def test_fold_cache_rejects_stale_metadata_misordered_ids_and_tampering(
        self,
    ) -> None:
        metadata = _metadata(1)
        identifiers = np.asarray([101, 202, 303])
        probabilities = np.asarray(
            [
                [0.70, 0.20, 0.10],
                [0.10, 0.65, 0.25],
                [0.15, 0.20, 0.65],
            ]
        )
        payload = make_fold_cache_payload(
            metadata,
            identifiers,
            probabilities,
            {"validation_fold": 1, "total_seconds": 2.5},
        )

        replayed, diagnostics = validate_fold_cache_payload(
            payload,
            metadata,
            identifiers,
        )
        np.testing.assert_array_equal(replayed, probabilities)
        self.assertEqual(diagnostics["validation_fold"], 1)

        stale = {**metadata, "recipe": {"name": "changed"}}
        with self.assertRaisesRegex(ValueError, "metadata is stale"):
            validate_fold_cache_payload(payload, stale, identifiers)
        with self.assertRaisesRegex(ValueError, "misordered"):
            validate_fold_cache_payload(payload, metadata, identifiers[::-1])

        tampered = {**payload, "probabilities": probabilities.copy()}
        tampered["probabilities"][0] = [0.60, 0.30, 0.10]
        with self.assertRaisesRegex(ValueError, "tampered"):
            validate_fold_cache_payload(tampered, metadata, identifiers)

    def test_fold_cache_rejects_payload_schema_and_diagnostic_tampering(
        self,
    ) -> None:
        metadata = _metadata(1)
        identifiers = np.asarray([1, 2])
        payload = make_fold_cache_payload(
            metadata,
            identifiers,
            np.asarray([[0.6, 0.2, 0.2], [0.2, 0.3, 0.5]]),
            {"total_seconds": 1.0},
        )
        with self.assertRaisesRegex(ValueError, "schema changed"):
            validate_fold_cache_payload(
                {**payload, "unexpected": True},
                metadata,
                identifiers,
            )
        tampered = {
            **payload,
            "diagnostics": {"total_seconds": 99.0},
        }
        with self.assertRaisesRegex(ValueError, "tampered"):
            validate_fold_cache_payload(tampered, metadata, identifiers)

    def test_five_fold_payloads_assemble_in_original_row_order(self) -> None:
        partitioned = _partitioned()
        probabilities = _perfect_probabilities(partitioned.y_development)
        payloads = {}
        metadata_by_fold = {}
        for fold, (_, validation_positions) in enumerate(
            make_cross_validation(partitioned).split(),
            start=1,
        ):
            metadata = _metadata(fold)
            metadata_by_fold[fold] = metadata
            payloads[fold] = make_fold_cache_payload(
                metadata,
                partitioned.development_ids.iloc[
                    validation_positions
                ].to_numpy(),
                probabilities[validation_positions],
                {"validation_fold": fold, "total_seconds": float(fold)},
            )

        evaluation = assemble_fold_evaluation(
            partitioned,
            payloads,
            metadata_by_fold,
            "assembled candidate",
        )

        np.testing.assert_array_equal(
            evaluation.out_of_fold_probabilities.to_numpy(),
            probabilities,
        )
        self.assertEqual(
            evaluation.cross_validation_fingerprint,
            FRESH_CROSS_VALIDATION_FINGERPRINT,
        )
        self.assertEqual(
            evaluation.diagnostics.index.tolist(),
            [1, 2, 3, 4, 5],
        )

    def test_comparison_builds_only_four_exact_one_factor_bags(self) -> None:
        partitioned = _partitioned()
        incumbent = _incumbent_components()
        alternatives = _alternative_components()

        comparison = compare_fresh_deep_archive_candidates(
            partitioned,
            incumbent,
            alternatives,
        )

        self.assertEqual(
            tuple(comparison.candidate_evaluations),
            LOCKED_FRESH_CANDIDATES,
        )
        self.assertEqual(
            comparison.summary.index.tolist(),
            list(LOCKED_FRESH_CANDIDATES),
        )
        incumbent_blend = _blend(incumbent)
        np.testing.assert_allclose(
            comparison.incumbent_evaluation.out_of_fold_probabilities,
            incumbent_blend,
        )
        expected = {
            CATBOOST_IDENTITY_GRID_BAG: (
                incumbent_blend
                + 0.20
                * (
                    0.5 * incumbent["identity_catboost"]
                    + 0.5 * alternatives[SPATIAL_GRID_COMPONENT]
                    - incumbent["identity_catboost"]
                )
            ),
            RF_CURRENT_LEAF_1_BAG: (
                incumbent_blend
                + 0.18
                * (
                    0.5 * incumbent["random_forest"]
                    + 0.5 * alternatives[RF_FEATURES_0_3_COMPONENT]
                    - incumbent["random_forest"]
                )
            ),
            RF_CURRENT_LEAF_2_BAG: (
                incumbent_blend
                + 0.18
                * (
                    0.5 * incumbent["random_forest"]
                    + 0.5 * alternatives[RF_FEATURES_0_3_LEAF_2_COMPONENT]
                    - incumbent["random_forest"]
                )
            ),
            DEEP_SEED_BAG: (
                incumbent_blend
                + 0.33
                * (
                    alternatives[DEEP_SEED_BAG_COMPONENT]
                    - incumbent["deep_xgboost"]
                )
            ),
        }
        for candidate, probabilities in expected.items():
            with self.subTest(candidate=candidate):
                np.testing.assert_allclose(
                    comparison.candidate_evaluations[
                        candidate
                    ].out_of_fold_probabilities,
                    probabilities,
                )

        expected_slot_weights = [0.20, 0.18, 0.18, 0.33]
        np.testing.assert_allclose(
            comparison.summary["slot_weight"],
            expected_slot_weights,
        )
        np.testing.assert_allclose(
            comparison.summary["within_slot_incumbent_weight"],
            0.5,
        )
        np.testing.assert_allclose(
            comparison.summary["within_slot_alternative_weight"],
            0.5,
        )

    def test_comparison_rejects_changed_component_sets_weights_and_folds(
        self,
    ) -> None:
        partitioned = _partitioned()
        incumbent = _incumbent_components()
        alternatives = _alternative_components()

        with self.assertRaisesRegex(ValueError, "alternative component set"):
            compare_fresh_deep_archive_candidates(
                partitioned,
                incumbent,
                {**alternatives, "unplanned": alternatives[SPATIAL_GRID_COMPONENT]},
            )
        with self.assertRaisesRegex(ValueError, "incumbent component set"):
            compare_fresh_deep_archive_candidates(
                partitioned,
                {
                    key: value
                    for key, value in incumbent.items()
                    if key != "spatial_xgboost"
                },
                alternatives,
            )
        with patch.dict(DEEP_ARCHIVE_WEIGHTS, {"identity_catboost": 0.19}):
            with self.assertRaisesRegex(ValueError, "weights changed"):
                compare_fresh_deep_archive_candidates(
                    partitioned,
                    incumbent,
                    alternatives,
                )
        stale_partition = PartitionedData(
            **{
                **vars(partitioned),
                "cross_validation_fingerprint": "changed",
            }
        )
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            compare_fresh_deep_archive_candidates(
                stale_partition,
                incumbent,
                alternatives,
            )

    def test_exact_gate_boundaries_pass(self) -> None:
        self.assertTrue(
            passes_fresh_deep_archive_gate(
                mean_accuracy_delta=ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
                fold_wins=ENSEMBLE_FOLD_WIN_GATE,
                worst_fold_delta=ENSEMBLE_WORST_FOLD_DELTA_GATE,
                repair_recall_delta=ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
            )
        )

    def test_each_failed_gate_boundary_rejects(self) -> None:
        passing = {
            "mean_accuracy_delta": ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
            "fold_wins": ENSEMBLE_FOLD_WIN_GATE,
            "worst_fold_delta": ENSEMBLE_WORST_FOLD_DELTA_GATE,
            "repair_recall_delta": ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
        }
        failures = (
            {"mean_accuracy_delta": ENSEMBLE_MEAN_ACCURACY_GAIN_GATE - 1e-6},
            {"fold_wins": ENSEMBLE_FOLD_WIN_GATE - 1},
            {"worst_fold_delta": ENSEMBLE_WORST_FOLD_DELTA_GATE - 1e-6},
            {
                "repair_recall_delta": (
                    ENSEMBLE_REPAIR_RECALL_DELTA_GATE - 1e-6
                )
            },
        )
        for failure in failures:
            with self.subTest(failure=failure):
                self.assertFalse(
                    passes_fresh_deep_archive_gate(
                        **{**passing, **failure}
                    )
                )

    def test_exploratory_guard_requires_strict_gain_and_evidence(self) -> None:
        stable = {
            "fold_wins": 3,
            "worst_fold_delta": -0.001,
            "repair_recall_delta": -0.01,
        }
        self.assertTrue(
            passes_fresh_deep_archive_exploratory_guard(
                mean_accuracy_delta=1e-12,
                component_evidence_eligible=True,
                **stable,
            )
        )
        self.assertFalse(
            passes_fresh_deep_archive_exploratory_guard(
                mean_accuracy_delta=0.0,
                component_evidence_eligible=True,
                **stable,
            )
        )
        self.assertFalse(
            passes_fresh_deep_archive_exploratory_guard(
                mean_accuracy_delta=1e-4,
                component_evidence_eligible=False,
                **stable,
            )
        )

    def test_component_evidence_eligibility_requires_exact_candidate_set(
        self,
    ) -> None:
        eligibility = {
            candidate: True for candidate in LOCKED_FRESH_CANDIDATES
        }
        eligibility.pop(LOCKED_FRESH_CANDIDATES[-1])
        with self.assertRaisesRegex(ValueError, "eligibility set changed"):
            compare_fresh_deep_archive_candidates(
                _partitioned(),
                _incumbent_components(),
                _alternative_components(),
                component_evidence_eligibility=eligibility,
            )

    def test_combined_fallback_adds_exactly_the_selected_one_factor_deltas(
        self,
    ) -> None:
        partitioned = _partitioned()
        comparison = _admissible_comparison(partitioned)

        combined = combine_fresh_deep_archive_fallback(
            partitioned,
            comparison,
            (RF_CURRENT_LEAF_1_BAG, CATBOOST_IDENTITY_GRID_BAG),
        )

        self.assertEqual(
            combined.selected_candidates,
            (CATBOOST_IDENTITY_GRID_BAG, RF_CURRENT_LEAF_1_BAG),
        )
        self.assertEqual(
            combined.selected_families,
            ("catboost", "random_forest"),
        )
        incumbent = comparison.incumbent_evaluation.out_of_fold_probabilities
        expected = incumbent.copy()
        for candidate in combined.selected_candidates:
            expected += (
                comparison.candidate_evaluations[
                    candidate
                ].out_of_fold_probabilities
                - incumbent
            )
        np.testing.assert_allclose(
            combined.candidate_evaluation.out_of_fold_probabilities,
            expected,
        )
        self.assertEqual(
            combined.summary.index.tolist(),
            [COMBINED_FALLBACK_CANDIDATE],
        )
        row = combined.summary.iloc[0]
        self.assertTrue(row["passes_formal_gate"])
        self.assertTrue(row["passes_exploratory_guard"])
        self.assertEqual(set(combined.fold_deltas["validation_fold"]), set(range(1, 6)))

    def test_combined_fallback_rejects_invalid_or_non_admissible_selections(
        self,
    ) -> None:
        partitioned = _partitioned()
        admissible = _admissible_comparison(partitioned)
        invalid = (
            ((CATBOOST_IDENTITY_GRID_BAG,), "two or three"),
            (
                (RF_CURRENT_LEAF_1_BAG, RF_CURRENT_LEAF_2_BAG),
                "duplicate model family",
            ),
            ((CATBOOST_IDENTITY_GRID_BAG, "unknown"), "unknown"),
        )
        for selected, message in invalid:
            with self.subTest(selected=selected):
                with self.assertRaisesRegex(ValueError, message):
                    combine_fresh_deep_archive_fallback(
                        partitioned,
                        admissible,
                        selected,
                    )

        neutral_components = {
            key: _constant_probabilities((0.34, 0.33, 0.33))
            for key in DEEP_ARCHIVE_WEIGHTS
        }
        neutral_alternatives = {
            key: _constant_probabilities((0.34, 0.33, 0.33))
            for key in LOCKED_ALTERNATIVE_COMPONENT_KEYS
        }
        non_admissible = compare_fresh_deep_archive_candidates(
            partitioned,
            neutral_components,
            neutral_alternatives,
        )
        with self.assertRaisesRegex(ValueError, "not admissible"):
            combine_fresh_deep_archive_fallback(
                partitioned,
                non_admissible,
                (CATBOOST_IDENTITY_GRID_BAG, DEEP_SEED_BAG),
            )

    def test_ten_seed_bag_directly_replaces_only_the_point_33_deep_slot(
        self,
    ) -> None:
        partitioned = _partitioned()
        incumbent = {
            key: _constant_probabilities((0.34, 0.33, 0.33))
            for key in DEEP_ARCHIVE_WEIGHTS
        }
        ten_seed_probabilities = _perfect_probabilities(
            partitioned.y_development
        )
        ten_seed = _evaluation(
            partitioned,
            TEN_BAG_NAME,
            ten_seed_probabilities,
        )

        result = compare_fresh_deep_ten_seed_direct_replacement(
            partitioned,
            incumbent,
            ten_seed,
        )

        incumbent_blend = _blend(incumbent)
        expected = (
            incumbent_blend
            + 0.33
            * (ten_seed_probabilities - incumbent["deep_xgboost"])
        )
        np.testing.assert_allclose(
            result.candidate_evaluation.out_of_fold_probabilities,
            expected,
        )
        self.assertEqual(
            result.summary.index.tolist(),
            [DEEP_TEN_SEED_DIRECT_REPLACEMENT],
        )
        row = result.summary.iloc[0]
        self.assertEqual(row["replaced_slot"], "deep_xgboost")
        self.assertEqual(row["slot_weight"], 0.33)
        self.assertFalse(row["component_evidence_eligible"])
        self.assertFalse(row["passes_exploratory_guard"])
        self.assertTrue(row["passes_formal_gate"])

    def test_ten_seed_direct_replacement_replays_evidence_and_model_identity(
        self,
    ) -> None:
        partitioned = _partitioned()
        incumbent = {
            key: _constant_probabilities((0.34, 0.33, 0.33))
            for key in DEEP_ARCHIVE_WEIGHTS
        }
        values = _perfect_probabilities(partitioned.y_development)
        wrong_name = _evaluation(partitioned, "not the ten-seed bag", values)
        with self.assertRaisesRegex(ValueError, "model identity changed"):
            compare_fresh_deep_ten_seed_direct_replacement(
                partitioned,
                incumbent,
                wrong_name,
            )

        ten_seed = _evaluation(partitioned, TEN_BAG_NAME, values)
        changed_metrics = ten_seed.fold_metrics.copy()
        changed_metrics.loc[1, "accuracy"] = 0.0
        with self.assertRaisesRegex(ValueError, "cached metrics changed"):
            compare_fresh_deep_ten_seed_direct_replacement(
                partitioned,
                incumbent,
                replace(ten_seed, fold_metrics=changed_metrics),
            )


def _metadata(fold: int) -> dict[str, object]:
    return {
        "candidate": "spatial_height_components",
        "recipe": {"version": 1, "fold_fitted": True},
        "source_sha256": {"TrainingSetValues.csv": "a" * 64},
        "labelled_rows": FRESH_LABELLED_ROWS,
        "cross_validation_folds": FRESH_CROSS_VALIDATION_FOLDS,
        "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": FRESH_CROSS_VALIDATION_FINGERPRINT,
        "validation_fold": fold,
    }


def _partitioned() -> PartitionedData:
    labels = np.asarray(
        ("functional", "functional needs repair", "non functional")
    )
    target = pd.Series(
        labels[np.arange(FRESH_LABELLED_ROWS) % len(labels)],
        name="status_group",
    )
    identifiers = pd.Series(np.arange(FRESH_LABELLED_ROWS), name="id")
    predictors = pd.DataFrame(index=target.index)
    folds = pd.Series(
        np.arange(FRESH_LABELLED_ROWS) % FRESH_CROSS_VALIDATION_FOLDS + 1,
        name="validation_fold",
    )
    return PartitionedData(
        development_ids=identifiers,
        X_development=predictors,
        y_development=target,
        local_test_ids=identifiers.iloc[0:0].copy(),
        X_local_test=predictors.iloc[0:0].copy(),
        y_local_test=target.iloc[0:0].copy(),
        validation_folds=folds,
        development_fingerprint="all-labelled",
        local_test_fingerprint="not-created",
        cross_validation_fingerprint=FRESH_CROSS_VALIDATION_FINGERPRINT,
    )


def _perfect_probabilities(target: pd.Series) -> np.ndarray:
    labels = {
        "functional": 0,
        "functional needs repair": 1,
        "non functional": 2,
    }
    probabilities = np.full((len(target), 3), 0.05)
    probabilities[
        np.arange(len(target)),
        [labels[value] for value in target],
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
            for fold in range(1, FRESH_CROSS_VALIDATION_FOLDS + 1)
        ],
    )


def _constant_probabilities(values: tuple[float, float, float]) -> np.ndarray:
    return np.tile(np.asarray(values, dtype="float64"), (FRESH_LABELLED_ROWS, 1))


def _incumbent_components() -> dict[str, np.ndarray]:
    return {
        "deep_xgboost": _constant_probabilities((0.61, 0.19, 0.20)),
        "spatial_xgboost": _constant_probabilities((0.55, 0.30, 0.15)),
        "random_forest": _constant_probabilities((0.52, 0.18, 0.30)),
        "frequency_random_forest": _constant_probabilities((0.48, 0.22, 0.30)),
        "spatial_random_forest": _constant_probabilities((0.45, 0.25, 0.30)),
        "identity_catboost": _constant_probabilities((0.58, 0.12, 0.30)),
    }


def _alternative_components() -> dict[str, np.ndarray]:
    self_check = (
        SPATIAL_GRID_COMPONENT,
        RF_FEATURES_0_3_COMPONENT,
        RF_FEATURES_0_3_LEAF_2_COMPONENT,
        DEEP_SEED_BAG_COMPONENT,
    )
    if self_check != LOCKED_ALTERNATIVE_COMPONENT_KEYS:
        raise AssertionError("Test alternative component order changed.")
    return {
        SPATIAL_GRID_COMPONENT: _constant_probabilities((0.40, 0.40, 0.20)),
        RF_FEATURES_0_3_COMPONENT: _constant_probabilities((0.30, 0.20, 0.50)),
        RF_FEATURES_0_3_LEAF_2_COMPONENT: _constant_probabilities(
            (0.20, 0.50, 0.30)
        ),
        DEEP_SEED_BAG_COMPONENT: _constant_probabilities((0.65, 0.15, 0.20)),
    }


def _admissible_comparison(
    partitioned: PartitionedData,
):
    incumbent = {
        key: _constant_probabilities((0.34, 0.33, 0.33))
        for key in DEEP_ARCHIVE_WEIGHTS
    }
    perfect = _perfect_probabilities(partitioned.y_development)
    alternatives = {
        key: perfect.copy() for key in LOCKED_ALTERNATIVE_COMPONENT_KEYS
    }
    return compare_fresh_deep_archive_candidates(
        partitioned,
        incumbent,
        alternatives,
    )


def _blend(components: dict[str, np.ndarray]) -> np.ndarray:
    return sum(
        DEEP_ARCHIVE_WEIGHTS[key] * components[key]
        for key in DEEP_ARCHIVE_WEIGHTS
    )


if __name__ == "__main__":
    unittest.main()

"""Tests for the strictly gated fresh-reconstruction submission portfolio."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
SCRIPTS_DIR = STAGE_DIR / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from fresh_deep_archive_reconstruction import CATBOOST_IDENTITY_GRID_BAG
from fresh_deep_archive_reconstruction import COMBINED_FALLBACK_CANDIDATE
from fresh_deep_archive_reconstruction import DEEP_SEED_BAG
from fresh_deep_archive_reconstruction import RF_CURRENT_LEAF_1_BAG
from fresh_deep_archive_reconstruction import RF_CURRENT_LEAF_2_BAG
from fresh_submission_portfolio import build_fresh_portfolio_probabilities
from fresh_submission_portfolio import COMPETITION_ROWS
from fresh_submission_portfolio import analyse_portfolio_label_distinctness
from fresh_submission_portfolio import make_component_cache_metadata
from fresh_submission_portfolio import make_component_cache_payload
from fresh_submission_portfolio import OPT_IN_TOKEN
from fresh_submission_portfolio import PORTFOLIO_CANDIDATES
from fresh_submission_portfolio import require_explicit_opt_in
from fresh_submission_portfolio import require_materially_distinct_portfolio
from fresh_submission_portfolio import validate_component_cache
from fresh_submission_portfolio import validate_fresh_portfolio_evidence
from fresh_submission_portfolio import validate_submission_frame
from fresh_submission_portfolio import write_submission_without_overwrite
import generate_fresh_reconstruction_portfolio as generator


class FreshSubmissionPortfolioTests(unittest.TestCase):
    def test_exact_opt_in_token_is_required(self) -> None:
        require_explicit_opt_in(OPT_IN_TOKEN)
        for value in (None, "generate", "fresh-reconstruction"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(PermissionError, "opt-in token"):
                    require_explicit_opt_in(value)

    def test_runner_cannot_open_competition_data_without_opt_in(self) -> None:
        with patch.object(generator, "_load_data") as load_data:
            with self.assertRaisesRegex(PermissionError, "opt-in token"):
                generator.main(opt_in=None)
        load_data.assert_not_called()

    def test_runner_accepts_real_cached_leaf_1_screen_recipe(self) -> None:
        result_path = (
            STAGE_DIR.parent
            / ".runtime/fresh-rf-family-screen-seed-20260905/result.json"
        )
        screen_result = json.loads(result_path.read_text(encoding="utf-8"))

        spec = generator.validate_rf_leaf_1_refit_recipe(screen_result)

        self.assertEqual(spec.variant, "Random Forest features 0.3")

    def test_only_locked_exploratory_slate_is_admitted(self) -> None:
        evidence = validate_fresh_portfolio_evidence(
            _base_result(),
        )

        self.assertEqual(
            tuple(evidence.base_rows),
            (CATBOOST_IDENTITY_GRID_BAG, RF_CURRENT_LEAF_1_BAG),
        )
        self.assertEqual(
            evidence.combined_row["candidate"],
            COMBINED_FALLBACK_CANDIDATE,
        )
        self.assertEqual(tuple(evidence.evidence_tier), PORTFOLIO_CANDIDATES)

    def test_evidence_recomputes_guards_and_rejects_changed_slate(self) -> None:
        changed = _base_result()
        changed["results"][0]["passes_exploratory_guard"] = False
        with self.assertRaisesRegex(ValueError, "exploratory gate flag"):
            validate_fresh_portfolio_evidence(changed)

        changed = _base_result()
        changed["exploratory_admissible_candidates"] = [
            CATBOOST_IDENTITY_GRID_BAG,
            RF_CURRENT_LEAF_2_BAG,
        ]
        with self.assertRaisesRegex(ValueError, "exploratory-pass list"):
            validate_fresh_portfolio_evidence(changed)

        changed = _base_result()
        changed["factorial_combination"]["selected_candidates"] = [
            CATBOOST_IDENTITY_GRID_BAG,
            DEEP_SEED_BAG,
        ]
        with self.assertRaisesRegex(ValueError, "different candidate set"):
            validate_fresh_portfolio_evidence(changed)

    def test_probability_build_uses_only_exact_half_slot_bags(self) -> None:
        incumbent = _incumbent_components()
        spatial = _probabilities(0.15, 0.20, 0.65)
        forest = _probabilities(0.25, 0.25, 0.50)

        result = build_fresh_portfolio_probabilities(
            incumbent,
            spatial_grid_catboost=spatial,
            rf_features_0_3=forest,
        )

        expected_incumbent = sum(
            weight * incumbent[name]
            for name, weight in {
                "deep_xgboost": 0.33,
                "spatial_xgboost": 0.11,
                "random_forest": 0.18,
                "frequency_random_forest": 0.09,
                "spatial_random_forest": 0.09,
                "identity_catboost": 0.20,
            }.items()
        )
        expected_cat = expected_incumbent + 0.10 * (
            spatial - incumbent["identity_catboost"]
        )
        expected_rf = expected_incumbent + 0.09 * (
            forest - incumbent["random_forest"]
        )
        np.testing.assert_allclose(result.incumbent, expected_incumbent)
        np.testing.assert_allclose(
            result.candidates[CATBOOST_IDENTITY_GRID_BAG],
            expected_cat,
        )
        np.testing.assert_allclose(
            result.candidates[RF_CURRENT_LEAF_1_BAG],
            expected_rf,
        )
        np.testing.assert_allclose(
            result.candidates[COMBINED_FALLBACK_CANDIDATE],
            expected_cat + expected_rf - expected_incumbent,
        )
        self.assertEqual(tuple(result.candidates), PORTFOLIO_CANDIDATES)

    def test_component_cache_rejects_stale_metadata_and_misordered_ids(self) -> None:
        ids = _competition_ids()
        metadata = make_component_cache_metadata(
            candidate="spatial_grid_catboost",
            recipe={"model": "locked"},
            data_sha256={"data": "a" * 64},
            source_sha256={"source": "b" * 64},
            evidence_sha256={"evidence": "c" * 64},
        )
        payload = make_component_cache_payload(
            metadata,
            ids,
            _full_probabilities(),
            1.5,
        )
        self.assertIs(validate_component_cache(payload, metadata, ids), payload)

        stale = {**payload, "metadata": {**metadata, "training_rows": 1}}
        with self.assertRaisesRegex(ValueError, "metadata is stale"):
            validate_component_cache(stale, metadata, ids)

        misordered = {**payload, "competition_ids": ids.iloc[::-1]}
        with self.assertRaisesRegex(ValueError, "misordered"):
            validate_component_cache(misordered, metadata, ids)

        tampered = deepcopy(payload)
        tampered["probabilities"][0] = [0.6, 0.2, 0.2]
        with self.assertRaisesRegex(ValueError, "tampered"):
            validate_component_cache(tampered, metadata, ids)

    def test_zero_disagreement_with_incumbent_or_peer_fails_closed(self) -> None:
        incumbent = np.resize(
            ["functional", "functional needs repair", "non functional"],
            COMPETITION_ROWS,
        )
        candidates = {
            COMBINED_FALLBACK_CANDIDATE: incumbent.copy(),
            CATBOOST_IDENTITY_GRID_BAG: np.roll(incumbent, 1),
            RF_CURRENT_LEAF_1_BAG: np.roll(incumbent, 1),
        }

        result = analyse_portfolio_label_distinctness(incumbent, candidates)

        self.assertEqual(
            result.redundant_candidates,
            (COMBINED_FALLBACK_CANDIDATE,),
        )
        self.assertEqual(len(result.redundant_pairs), 1)
        with self.assertRaisesRegex(ValueError, "redundant hard-label"):
            require_materially_distinct_portfolio(result)

    def test_nonzero_pairwise_and_incumbent_disagreements_are_accepted(self) -> None:
        incumbent = np.resize(
            ["functional", "functional needs repair", "non functional"],
            COMPETITION_ROWS,
        )
        candidates = {
            COMBINED_FALLBACK_CANDIDATE: np.roll(incumbent, 1),
            CATBOOST_IDENTITY_GRID_BAG: np.roll(incumbent, 2),
            RF_CURRENT_LEAF_1_BAG: incumbent.copy(),
        }
        candidates[RF_CURRENT_LEAF_1_BAG][0] = "non functional"

        result = analyse_portfolio_label_distinctness(incumbent, candidates)

        require_materially_distinct_portfolio(result)
        self.assertTrue(all(value > 0 for value in result.vs_incumbent.values()))
        self.assertTrue(all(value > 0 for value in result.pairwise.values()))

    def test_submission_requires_all_rows_ordered_unique_and_three_labels(self) -> None:
        ids = _competition_ids()
        labels = np.resize(
            np.asarray(
                [
                    "functional",
                    "functional needs repair",
                    "non functional",
                ]
            ),
            COMPETITION_ROWS,
        )
        submission = pd.DataFrame({"id": ids, "status_group": labels})
        validate_submission_frame(submission, ids)

        missing_label = submission.copy()
        missing_label["status_group"] = "functional"
        with self.assertRaisesRegex(ValueError, "exactly three labels"):
            validate_submission_frame(missing_label, ids)

        misordered = submission.iloc[::-1].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "misordered"):
            validate_submission_frame(misordered, ids)

    def test_submission_writer_refuses_even_identical_existing_output(self) -> None:
        ids = _competition_ids()
        submission = pd.DataFrame(
            {
                "id": ids,
                "status_group": np.resize(
                    [
                        "functional",
                        "functional needs repair",
                        "non functional",
                    ],
                    COMPETITION_ROWS,
                ),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            write_submission_without_overwrite(submission, ids, path)
            with self.assertRaisesRegex(FileExistsError, "Refusing"):
                write_submission_without_overwrite(submission, ids, path)


def _base_result() -> dict[str, object]:
    rows = [
        _base_row(
            CATBOOST_IDENTITY_GRID_BAG,
            "identity_catboost",
            "spatial_grid_catboost",
            0.20,
            0.10,
            mean_delta=0.000218855218855,
            wins=4,
            losses=1,
            worst=-0.000252525252525,
            best=0.000589225589226,
            repair_delta=-0.001621175056864,
            disagreements=272,
            gained=130,
            lost=117,
            formal=False,
            exploratory=True,
        ),
        _base_row(
            RF_CURRENT_LEAF_1_BAG,
            "random_forest",
            "rf_features_0_3",
            0.18,
            0.09,
            mean_delta=0.000151515151515,
            wins=3,
            losses=2,
            worst=-0.000252525252525,
            best=0.000673400673401,
            repair_delta=-0.000463231191794,
            disagreements=128,
            gained=60,
            lost=51,
            formal=False,
            exploratory=True,
        ),
        _base_row(
            RF_CURRENT_LEAF_2_BAG,
            "random_forest",
            "rf_features_0_3_leaf_2",
            0.18,
            0.09,
            mean_delta=0.000185185185185,
            wins=4,
            losses=1,
            worst=-0.001094276094276,
            best=0.000841750841751,
            repair_delta=-0.001854265911334,
            disagreements=297,
            gained=138,
            lost=127,
            formal=False,
            exploratory=False,
        ),
        _base_row(
            DEEP_SEED_BAG,
            "deep_xgboost",
            "deep_seed_20260824_20260905_bag",
            0.33,
            0.165,
            mean_delta=-0.000252525252525,
            wins=1,
            losses=4,
            worst=-0.000673400673401,
            best=0.000336700336700,
            repair_delta=-0.000926998841251,
            disagreements=151,
            gained=61,
            lost=76,
            formal=False,
            exploratory=False,
        ),
    ]
    combined = _combined_result()
    ten_seed = _ten_seed_result()
    return {
        "screen": "fresh exact deep-archive one-factor reconstruction",
        "status": "comparison_complete",
        "formal_passing_candidates": [],
        "exploratory_admissible_candidates": [
            CATBOOST_IDENTITY_GRID_BAG,
            RF_CURRENT_LEAF_1_BAG,
        ],
        "admissible_candidates": [
            CATBOOST_IDENTITY_GRID_BAG,
            RF_CURRENT_LEAF_1_BAG,
            COMBINED_FALLBACK_CANDIDATE,
        ],
        "ten_seed_formal_only_append": ten_seed,
        "factorial_combination": combined,
        "deep_archive_weights": {
            "deep_xgboost": 0.33,
            "spatial_xgboost": 0.11,
            "random_forest": 0.18,
            "frequency_random_forest": 0.09,
            "spatial_random_forest": 0.09,
            "identity_catboost": 0.20,
        },
        "source_sha256": {
            "TrainingSetValues.csv": (
                "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
            ),
            "TrainingSetLabels.csv": (
                "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
            ),
        },
        "evidence_sha256": {"component": "c" * 64},
        "component_evidence": _component_evidence(),
        "protocol": {
            "labelled_rows": 59_400,
            "cross_validation_seed": 20260905,
            "cross_validation_fingerprint": (
                "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
            ),
            "historical_local_subset": "not reconstructed or consulted",
            "competition_data_opened": False,
            "competition_predictions_generated": False,
        },
        "results": [
            *rows,
            ten_seed["result"],
            combined["result"],
        ],
        "output_sha256": {
            name: "d" * 64
            for name in (
                "candidate-summary.csv",
                "fold-paired-deltas.csv",
                "combined-catboost-rf-leaf-1-summary.csv",
                "combined-catboost-rf-leaf-1-fold-paired-deltas.csv",
                "ten-seed-direct-replacement-summary.csv",
                "ten-seed-direct-replacement-fold-paired-deltas.csv",
                "all-candidate-summary.csv",
                "all-fold-paired-deltas.csv",
            )
        },
    }


def _base_row(
    candidate: str,
    slot: str,
    alternative: str,
    slot_weight: float,
    effective_weight: float,
    *,
    mean_delta: float,
    wins: int,
    losses: int,
    worst: float,
    best: float,
    repair_delta: float,
    disagreements: int,
    gained: int,
    lost: int,
    formal: bool,
    exploratory: bool,
) -> dict[str, object]:
    incumbent_accuracy = 0.8203030303030303
    incumbent_repair = 0.3349547766190292
    return {
        "candidate": candidate,
        "replaced_slot": slot,
        "alternative_component": alternative,
        "slot_weight": slot_weight,
        "within_slot_incumbent_weight": 0.5,
        "within_slot_alternative_weight": 0.5,
        "effective_incumbent_component_weight": effective_weight,
        "effective_alternative_component_weight": effective_weight,
        "incumbent_mean_accuracy": incumbent_accuracy,
        "candidate_mean_accuracy": incumbent_accuracy + mean_delta,
        "mean_accuracy_delta": mean_delta,
        "fold_wins": wins,
        "fold_losses": losses,
        "worst_fold_delta": worst,
        "best_fold_delta": best,
        "incumbent_repair_recall": incumbent_repair,
        "candidate_repair_recall": incumbent_repair + repair_delta,
        "repair_recall_delta": repair_delta,
        "prediction_disagreements": disagreements,
        "gained_correct": gained,
        "lost_correct": lost,
        "net_additional_correct": gained - lost,
        "component_evidence_eligible": True,
        "passes_formal_gate": formal,
        "passes_exploratory_guard": exploratory,
    }


def _combined_result() -> dict[str, object]:
    incumbent_accuracy = 0.8203030303030303
    incumbent_repair = 0.3349547766190292
    fold_nets = [12, -7, -3, 7, 4]
    deltas = [value / 11_880 for value in fold_nets]
    delta = sum(fold_nets) / 59_400
    repair_delta = 0.0
    selected = [CATBOOST_IDENTITY_GRID_BAG, RF_CURRENT_LEAF_1_BAG]
    families = ["catboost", "random_forest"]
    folds = []
    for fold, (net, accuracy_delta) in enumerate(zip(fold_nets, deltas), start=1):
        gained = max(net, 0)
        lost = max(-net, 0)
        folds.append(
            {
                "candidate": COMBINED_FALLBACK_CANDIDATE,
                "validation_fold": fold,
                "rows": 11_880,
                "incumbent_accuracy": 0.82,
                "candidate_accuracy": 0.82 + accuracy_delta,
                "accuracy_delta": accuracy_delta,
                "incumbent_repair_recall": incumbent_repair,
                "candidate_repair_recall": incumbent_repair,
                "repair_recall_delta": 0.0,
                "prediction_disagreements": gained + lost,
                "gained_correct": gained,
                "lost_correct": lost,
                "net_additional_correct": net,
            }
        )
    row = {
        "candidate": COMBINED_FALLBACK_CANDIDATE,
        "selected_candidates": selected,
        "selected_families": families,
        "selection_count": 2,
        "incumbent_mean_accuracy": incumbent_accuracy,
        "candidate_mean_accuracy": incumbent_accuracy + delta,
        "mean_accuracy_delta": delta,
        "fold_wins": 3,
        "fold_losses": 2,
        "worst_fold_delta": min(deltas),
        "best_fold_delta": max(deltas),
        "incumbent_repair_recall": incumbent_repair,
        "candidate_repair_recall": incumbent_repair + repair_delta,
        "repair_recall_delta": repair_delta,
        "prediction_disagreements": sum(
            fold["prediction_disagreements"] for fold in folds
        ),
        "gained_correct": sum(fold["gained_correct"] for fold in folds),
        "lost_correct": sum(fold["lost_correct"] for fold in folds),
        "net_additional_correct": sum(fold_nets),
        "component_evidence_eligible": True,
        "passes_formal_gate": False,
        "passes_exploratory_guard": True,
    }
    return {
        "candidate": COMBINED_FALLBACK_CANDIDATE,
        "selected_candidates": selected,
        "selected_families": families,
        "result": row,
        "fold_deltas": folds,
    }


def _ten_seed_result() -> dict[str, object]:
    candidate = "deep_ten_seed_equal_bag_at_0_33"
    incumbent_accuracy = 0.8203030303030303
    fold_delta = -1 / 11_880
    mean_delta = -5 / 59_400
    incumbent_repair = 0.3349547766190292
    folds = [
        {
            "candidate": candidate,
            "validation_fold": fold,
            "rows": 11_880,
            "incumbent_accuracy": 0.82,
            "candidate_accuracy": 0.82 + fold_delta,
            "accuracy_delta": fold_delta,
            "incumbent_repair_recall": incumbent_repair,
            "candidate_repair_recall": incumbent_repair,
            "repair_recall_delta": 0.0,
            "prediction_disagreements": 1,
            "gained_correct": 0,
            "lost_correct": 1,
            "net_additional_correct": -1,
        }
        for fold in range(1, 6)
    ]
    return {
        "candidate": candidate,
        "component_evidence_eligible": False,
        "exploratory_consideration_permitted": False,
        "result": {
            "candidate": candidate,
            "replaced_slot": "deep_xgboost",
            "slot_weight": 0.33,
            "replacement_mode": "direct validated equal ten-seed bag",
            "component_evidence_eligible": False,
            "incumbent_mean_accuracy": incumbent_accuracy,
            "candidate_mean_accuracy": incumbent_accuracy + mean_delta,
            "mean_accuracy_delta": mean_delta,
            "fold_wins": 0,
            "fold_losses": 5,
            "worst_fold_delta": fold_delta,
            "best_fold_delta": fold_delta,
            "incumbent_repair_recall": incumbent_repair,
            "candidate_repair_recall": incumbent_repair,
            "repair_recall_delta": 0.0,
            "prediction_disagreements": 5,
            "gained_correct": 0,
            "lost_correct": 5,
            "net_additional_correct": -5,
            "passes_formal_gate": False,
            "passes_exploratory_guard": False,
        },
        "fold_deltas": folds,
    }


def _component_evidence() -> dict[str, object]:
    return {
        "catboost_identity_grid_bag": [
            {
                "candidate": "complete_identity_spatial_grid_50_50_bag",
                "mean_accuracy_delta": 0.0018,
                "fold_wins": 3,
                "worst_fold_delta": -0.0008,
                "repair_recall_delta": -0.004,
                "passes_gate": True,
            }
        ],
        "rf_probability_bags": [
            {
                "candidate": "Random Forest features 0.3 leaf 2",
                "mean_accuracy_delta": 0.004,
                "fold_wins_vs_incumbent": 5,
                "worst_fold_delta": 0.003,
                "repair_recall_delta": -0.012,
                "passes_gate": True,
            },
            {
                "candidate": "Random Forest features 0.3",
                "mean_accuracy_delta": 0.0009,
                "fold_wins_vs_incumbent": 5,
                "worst_fold_delta": 0.00008,
                "repair_recall_delta": 0.002,
                "passes_gate": False,
            },
            {
                "candidate": "Current Random Forest",
                "mean_accuracy_delta": 0.0,
                "fold_wins_vs_incumbent": 0,
                "worst_fold_delta": 0.0,
                "repair_recall_delta": 0.0,
                "passes_gate": False,
            },
        ],
        "deep_two_seed_bag": {
            "passes_gate_against_both": False,
            "stable_positive_near_miss": True,
            "verdict": "stable_positive_near_miss",
            "comparisons": [
                {"comparison": "average_vs_seed_20260824"},
                {"comparison": "average_vs_seed_20260905"},
            ],
        },
        "eligibility": {
            CATBOOST_IDENTITY_GRID_BAG: True,
            RF_CURRENT_LEAF_1_BAG: True,
            RF_CURRENT_LEAF_2_BAG: True,
            DEEP_SEED_BAG: True,
        },
    }


def _incumbent_components() -> dict[str, np.ndarray]:
    return {
        "deep_xgboost": _probabilities(0.70, 0.10, 0.20),
        "spatial_xgboost": _probabilities(0.65, 0.15, 0.20),
        "random_forest": _probabilities(0.55, 0.25, 0.20),
        "frequency_random_forest": _probabilities(0.50, 0.20, 0.30),
        "spatial_random_forest": _probabilities(0.45, 0.20, 0.35),
        "identity_catboost": _probabilities(0.40, 0.20, 0.40),
    }


def _probabilities(functional: float, repair: float, non_functional: float):
    return np.asarray([[functional, repair, non_functional]], dtype="float64")


def _competition_ids() -> pd.Series:
    return pd.Series(np.arange(COMPETITION_ROWS, dtype="int64"), name="id")


def _full_probabilities() -> np.ndarray:
    return np.tile(np.asarray([[0.7, 0.1, 0.2]]), (COMPETITION_ROWS, 1))


if __name__ == "__main__":
    unittest.main()

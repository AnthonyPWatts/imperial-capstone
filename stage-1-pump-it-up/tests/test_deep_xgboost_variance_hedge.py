"""Tests for the optional deep-XGBoost two-seed variance hedge."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_xgboost_variance_hedge import build_variance_hedge_probabilities
from deep_xgboost_variance_hedge import canonical_deep_xgboost_recipe
from deep_xgboost_variance_hedge import canonical_probability_average_recipe
from deep_xgboost_variance_hedge import EXPECTED_GATE
from deep_xgboost_variance_hedge import make_deep_component_cache_metadata
from deep_xgboost_variance_hedge import OPT_IN_TOKEN
from deep_xgboost_variance_hedge import require_explicit_opt_in
from deep_xgboost_variance_hedge import validate_near_miss_evidence
from deep_xgboost_variance_hedge import validate_submission_frame
from deep_xgboost_variance_hedge import validate_versioned_deep_component_cache
from deep_xgboost_variance_hedge import write_submission_without_overwrite


class DeepXgboostVarianceHedgeTests(unittest.TestCase):
    def test_exact_average_replaces_only_the_unchanged_point_33_slot(self) -> None:
        incumbent = _incumbent_components()
        seed_20260824 = incumbent["deep_xgboost"]
        seed_20260905 = _probabilities(0.50, 0.30, 0.20)

        result = build_variance_hedge_probabilities(
            incumbent,
            seed_20260824,
            seed_20260905,
        )

        expected_average = 0.5 * (seed_20260824 + seed_20260905)
        np.testing.assert_allclose(result.deep_seed_average, expected_average)
        np.testing.assert_allclose(
            result.candidate_ensemble,
            result.incumbent_ensemble
            + 0.33 * (expected_average - incumbent["deep_xgboost"]),
        )
        self.assertIs(incumbent["deep_xgboost"], seed_20260824)

    def test_component_contract_rejects_missing_or_unexpected_inputs(self) -> None:
        incumbent = _incumbent_components()
        del incumbent["spatial_xgboost"]

        with self.assertRaisesRegex(ValueError, "component contract"):
            build_variance_hedge_probabilities(
                incumbent,
                _probabilities(0.70, 0.10, 0.20),
                _probabilities(0.50, 0.30, 0.20),
            )

    def test_near_miss_evidence_accepts_only_the_locked_no_promotion_result(
        self,
    ) -> None:
        result = _near_miss_evidence()
        validate_near_miss_evidence(result)

        weakened = deepcopy(result)
        weakened["promotion_decision"] = "promote"
        with self.assertRaisesRegex(ValueError, "no-promotion"):
            validate_near_miss_evidence(weakened)

        weakened = deepcopy(result)
        weakened["comparison_summary"][0]["mean_accuracy_delta"] = 0.001
        with self.assertRaisesRegex(ValueError, "near-miss"):
            validate_near_miss_evidence(weakened)

    def test_versioned_cache_requires_exact_metadata_ids_and_probabilities(
        self,
    ) -> None:
        expected_ids = pd.Series([10, 20], name="id")
        metadata = make_deep_component_cache_metadata(
            seed=20260905,
            training_rows=59_400,
            competition_rows=14_850,
            data_sha256={"data": "abc"},
            source_sha256={"source": "def"},
            evidence_sha256={"evidence": "ghi"},
        )
        payload = {
            "cache_version": 1,
            "metadata": metadata,
            "competition_ids": expected_ids.copy(),
            "probabilities": np.asarray(
                [[0.70, 0.10, 0.20], [0.20, 0.30, 0.50]]
            ),
            "fit_and_predict_seconds": 1.25,
        }
        validated = validate_versioned_deep_component_cache(
            payload,
            metadata,
            expected_ids,
        )
        self.assertIs(validated, payload)

        stale = {**payload, "metadata": {**metadata, "cache_version": 2}}
        with self.assertRaisesRegex(ValueError, "metadata is stale"):
            validate_versioned_deep_component_cache(
                stale,
                metadata,
                expected_ids,
            )
        misordered = {**payload, "competition_ids": expected_ids.iloc[::-1]}
        with self.assertRaisesRegex(ValueError, "misordered"):
            validate_versioned_deep_component_cache(
                misordered,
                metadata,
                expected_ids,
            )
        wrong_shape = {**payload, "probabilities": np.asarray([[0.7, 0.3]])}
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_versioned_deep_component_cache(
                wrong_shape,
                metadata,
                expected_ids,
            )

    def test_submission_validation_rejects_columns_and_misordered_ids(self) -> None:
        expected_ids = pd.Series([10, 20], name="id")
        with self.assertRaisesRegex(ValueError, "columns"):
            validate_submission_frame(
                pd.DataFrame({"status_group": ["functional", "functional"]}),
                expected_ids,
            )
        with self.assertRaisesRegex(ValueError, "misordered"):
            validate_submission_frame(
                pd.DataFrame(
                    {
                        "id": [20, 10],
                        "status_group": ["functional", "non functional"],
                    }
                ),
                expected_ids,
            )

    def test_submission_writer_refuses_even_an_identical_existing_file(self) -> None:
        identifiers = pd.Series([10], name="id")
        submission = pd.DataFrame(
            {"id": identifiers, "status_group": ["functional"]}
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "candidate.csv"
            write_submission_without_overwrite(
                submission,
                identifiers,
                destination,
            )

            with self.assertRaisesRegex(FileExistsError, "Refusing"):
                write_submission_without_overwrite(
                    submission,
                    identifiers,
                    destination,
                )

    def test_exact_near_miss_opt_in_token_is_required(self) -> None:
        require_explicit_opt_in(OPT_IN_TOKEN)
        for value in (None, "promote", "variance-hedge"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(PermissionError, "failed"):
                    require_explicit_opt_in(value)


def _near_miss_evidence() -> dict[str, object]:
    recipes = {}
    for key, seed in (
        ("seed_20260824", 20260824),
        ("seed_20260905", 20260905),
    ):
        recipe = canonical_deep_xgboost_recipe(seed)
        recipe.pop("training_scope")
        recipes[key] = recipe
    recipes["probability_average"] = canonical_probability_average_recipe()
    comparisons = [
        {
            "comparison": "average_vs_seed_20260824",
            "mean_accuracy_delta": 0.00052,
            "fold_wins": 3,
            "worst_fold_delta": -0.0001,
            "repair_recall_delta": 0.0,
            "passes_gate": False,
        },
        {
            "comparison": "average_vs_seed_20260905",
            "mean_accuracy_delta": 0.00039,
            "fold_wins": 4,
            "worst_fold_delta": -0.00042,
            "repair_recall_delta": -0.00046,
            "passes_gate": False,
        },
    ]
    return {
        "protocol": {
            "cache_only": True,
            "models_fitted": 0,
            "labelled_rows": 59_400,
            "folds": 5,
            "fold_seed": 20260905,
            "comparison": "only the fixed 50:50 probability average",
            "competition_predictions_generated": False,
        },
        "gate": dict(EXPECTED_GATE),
        "recipes": recipes,
        "comparison_summary": comparisons,
        "gate_failures": {
            "average_vs_seed_20260824": ["mean_accuracy_gain"],
            "average_vs_seed_20260905": ["mean_accuracy_gain"],
        },
        "passes_gate": False,
        "stable_positive_near_miss": True,
        "verdict": "stable_positive_near_miss",
        "selected_candidate": None,
        "promotion_decision": "do_not_promote",
        "competition_predictions_generated": False,
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
    return np.asarray([[functional, repair, non_functional]])


if __name__ == "__main__":
    unittest.main()

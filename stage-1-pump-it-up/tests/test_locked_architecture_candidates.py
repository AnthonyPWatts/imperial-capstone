"""Tests for gated substitutions into the locked deep archive."""

from __future__ import annotations

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

from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from locked_architecture_candidates import COMBINED_SUBSTITUTION
from locked_architecture_candidates import RF_SUBSTITUTION
from locked_architecture_candidates import SPATIAL_SUBSTITUTION
from locked_architecture_candidates import build_locked_substitution_probabilities
from locked_architecture_candidates import decide_locked_replacements
from locked_architecture_candidates import LockedReplacementDecision
from locked_architecture_candidates import validate_competition_ids
from locked_architecture_candidates import validate_replacement_cache
from locked_architecture_candidates import validate_screen_contract
from locked_architecture_candidates import write_json_without_overwrite


class LockedArchitectureCandidatesTests(unittest.TestCase):
    def test_each_substitution_preserves_the_incumbent_component_weight(self) -> None:
        incumbent = _incumbent_components()
        spatial = _probabilities(0.15, 0.20, 0.65)
        forest = _probabilities(0.25, 0.25, 0.50)

        candidates = build_locked_substitution_probabilities(
            incumbent,
            LockedReplacementDecision(True, "Extra Trees leaf 2"),
            spatial_grid_catboost=spatial,
            rf_family=forest,
        )

        baseline = sum(
            DEEP_ARCHIVE_WEIGHTS[key] * incumbent[key]
            for key in DEEP_ARCHIVE_WEIGHTS
        )
        np.testing.assert_allclose(
            candidates[SPATIAL_SUBSTITUTION],
            baseline
            + 0.20 * (spatial - incumbent["identity_catboost"]),
        )
        np.testing.assert_allclose(
            candidates[RF_SUBSTITUTION],
            baseline + 0.18 * (forest - incumbent["random_forest"]),
        )
        np.testing.assert_allclose(
            candidates[COMBINED_SUBSTITUTION],
            baseline
            + 0.20 * (spatial - incumbent["identity_catboost"])
            + 0.18 * (forest - incumbent["random_forest"]),
        )

    def test_only_supplied_passing_replacement_produces_a_candidate(self) -> None:
        candidates = build_locked_substitution_probabilities(
            _incumbent_components(),
            LockedReplacementDecision(False, "Extra Trees leaf 2"),
            rf_family=_probabilities(0.25, 0.25, 0.50),
        )

        self.assertEqual(set(candidates), {RF_SUBSTITUTION})

    def test_blend_rejects_replacements_that_disagree_with_gate_decision(self) -> None:
        with self.assertRaisesRegex(ValueError, "Spatial replacement"):
            build_locked_substitution_probabilities(
                _incumbent_components(),
                LockedReplacementDecision(False, None),
                spatial_grid_catboost=_probabilities(0.25, 0.25, 0.50),
            )

    def test_screen_decisions_admit_only_explicit_passing_challengers(self) -> None:
        decision = decide_locked_replacements(
            {
                "passes_gate": True,
                "selected_candidate": "spatial_grid_catboost",
            },
            {
                "passes_gate": True,
                "selected_candidate": "Extra Trees leaf 2",
            },
            allowed_rf_variants=(
                "Current Random Forest",
                "Extra Trees leaf 2",
            ),
        )

        self.assertTrue(decision.use_spatial_grid_catboost)
        self.assertEqual(decision.rf_variant, "Extra Trees leaf 2")

    def test_screen_decisions_reject_missing_or_inconsistent_gates(self) -> None:
        allowed = ("Current Random Forest", "Extra Trees leaf 2")
        cases = (
            (
                {"selected_candidate": "spatial_grid_catboost"},
                {
                    "passes_gate": False,
                    "selected_candidate": "Current Random Forest",
                },
            ),
            (
                {
                    "passes_gate": False,
                    "selected_candidate": "spatial_grid_catboost",
                },
                {
                    "passes_gate": False,
                    "selected_candidate": "Current Random Forest",
                },
            ),
            (
                {
                    "passes_gate": False,
                    "selected_candidate": "complete_identity_catboost",
                },
                {
                    "passes_gate": True,
                    "selected_candidate": "Current Random Forest",
                },
            ),
        )
        for spatial, rf in cases:
            with self.subTest(spatial=spatial, rf=rf):
                with self.assertRaises(ValueError):
                    decide_locked_replacements(
                        spatial,
                        rf,
                        allowed_rf_variants=allowed,
                    )

    def test_stale_screen_recipe_is_rejected(self) -> None:
        expected = {
            "protocol": {"seed": 20260905},
            "model_recipe": {"variant": "d8", "seed": 20260821},
        }
        actual = {
            "protocol": {"seed": 20260905},
            "model_recipe": {"variant": "d8", "seed": 7},
        }

        with self.assertRaisesRegex(ValueError, "seed.*changed"):
            validate_screen_contract(actual, expected, screen="spatial")

    def test_competition_id_misalignment_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "misordered"):
            validate_competition_ids(
                {"competition_ids": pd.Series([20, 10])},
                pd.Series([10, 20]),
                name="replacement",
            )

    def test_stale_replacement_cache_metadata_is_rejected(self) -> None:
        payload = {
            "metadata": {"iterations": 99},
            "competition_ids": pd.Series([10]),
            "probabilities": _probabilities(0.25, 0.25, 0.50),
            "fit_and_predict_seconds": 1.0,
        }

        with self.assertRaisesRegex(ValueError, "metadata is stale"):
            validate_replacement_cache(
                payload,
                {"iterations": 100},
                pd.Series([10]),
                expected_rows=1,
                name="replacement",
            )

    def test_different_existing_manifest_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json_without_overwrite(path, {"candidate": "first"})

            with self.assertRaisesRegex(FileExistsError, "Refusing"):
                write_json_without_overwrite(path, {"candidate": "second"})

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{\n  "candidate": "first"\n}\n',
            )


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

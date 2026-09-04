"""Tests for selective, repair-preserving archive decisions."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_archive_gate import ARCHIVE_GATE_COMPONENTS
from deep_archive_gate import apply_repair_preserving_archive_gate
from deep_archive_gate import archive_choice_target
from deep_archive_gate import build_archive_and_deep_probabilities
from deep_archive_gate import build_archive_gate_features


class DeepArchiveGateTests(unittest.TestCase):
    def test_feature_rows_are_only_hard_disagreements(self) -> None:
        components = _components()
        archive, deep = build_archive_and_deep_probabilities(components)

        features = build_archive_gate_features(components, archive, deep)

        np.testing.assert_array_equal(features.row_positions, [0, 1])
        np.testing.assert_array_equal(features.archive_positions, [0, 2])
        np.testing.assert_array_equal(features.deep_positions, [1, 1])
        self.assertEqual(features.values.shape, (2, 8))
        self.assertTrue(
            np.isfinite(features.values.drop(columns="transition")).all().all()
        )

    def test_gate_never_removes_a_deep_repair_decision(self) -> None:
        components = _components()
        archive, deep = build_archive_and_deep_probabilities(components)
        features = build_archive_gate_features(components, archive, deep)

        gated, selected = apply_repair_preserving_archive_gate(
            archive,
            deep,
            features,
            np.asarray([0.90, 0.90]),
        )

        np.testing.assert_array_equal(selected, [False, False, False])
        np.testing.assert_array_equal(gated, deep)

    def test_choice_target_excludes_third_class_outcomes(self) -> None:
        components = _components()
        archive, deep = build_archive_and_deep_probabilities(components)
        features = build_archive_gate_features(components, archive, deep)

        relevant, target = archive_choice_target(
            np.asarray(["functional", "functional", "functional"]),
            features,
        )

        np.testing.assert_array_equal(relevant, [True, False])
        np.testing.assert_array_equal(target, [1, 0])

    def test_missing_component_is_rejected(self) -> None:
        components = _components()
        components.pop("identity_catboost")

        with self.assertRaisesRegex(KeyError, "identity_catboost"):
            build_archive_and_deep_probabilities(components)


def _components() -> dict[str, np.ndarray]:
    accepted = np.asarray(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.05, 0.90],
            [0.80, 0.10, 0.10],
        ]
    )
    deep = np.asarray(
        [
            [0.05, 0.90, 0.05],
            [0.05, 0.90, 0.05],
            [0.80, 0.10, 0.10],
        ]
    )
    neutral = np.asarray(
        [
            [0.45, 0.45, 0.10],
            [0.10, 0.45, 0.45],
            [0.80, 0.10, 0.10],
        ]
    )
    components = {
        name: neutral.copy() for name in ARCHIVE_GATE_COMPONENTS
    }
    components["accepted_xgboost"] = accepted
    components["deep_xgboost"] = deep
    return components


if __name__ == "__main__":
    unittest.main()

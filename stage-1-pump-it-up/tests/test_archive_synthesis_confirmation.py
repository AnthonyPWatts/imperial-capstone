"""Tests for the fixed archive-synthesis local confirmation recipe."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archive_synthesis_confirmation import ARCHIVE_SYNTHESIS_WEIGHTS
from archive_synthesis_confirmation import blend_archive_synthesis


class ArchiveSynthesisConfirmationTests(unittest.TestCase):
    def test_six_component_recipe_preserves_algebraic_weights(self) -> None:
        components = {
            key: _probabilities(position / 10.0)
            for position, key in enumerate(ARCHIVE_SYNTHESIS_WEIGHTS, start=1)
        }

        actual = blend_archive_synthesis(components)
        expected = sum(
            ARCHIVE_SYNTHESIS_WEIGHTS[key] * components[key]
            for key in ARCHIVE_SYNTHESIS_WEIGHTS
        )

        np.testing.assert_allclose(actual, expected)
        self.assertAlmostEqual(sum(ARCHIVE_SYNTHESIS_WEIGHTS.values()), 1.0)

    def test_missing_component_is_rejected(self) -> None:
        components = {
            key: _probabilities(position / 10.0)
            for position, key in enumerate(ARCHIVE_SYNTHESIS_WEIGHTS, start=1)
            if key != "spatial_xgboost"
        }

        with self.assertRaisesRegex(KeyError, "spatial_xgboost"):
            blend_archive_synthesis(components)


def _probabilities(offset: float) -> np.ndarray:
    values = np.asarray([[0.50 + offset / 10, 0.30, 0.20 - offset / 10]])
    return values / values.sum(axis=1, keepdims=True)


if __name__ == "__main__":
    unittest.main()

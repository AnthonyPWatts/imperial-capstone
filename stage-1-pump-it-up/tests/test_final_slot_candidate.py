"""Tests for the final-slot consensus repair rule."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


STAGE_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = STAGE_DIR / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from prepare_final_slot_candidate import _consensus_repair_mask
from repair_residual_specialist import REPAIR_META_COMPONENTS


class FinalSlotCandidateTests(unittest.TestCase):
    def test_consensus_requires_votes_ratio_and_functional_base(self) -> None:
        base = np.asarray(
            [
                [0.505, 0.490, 0.005],
                [0.510, 0.480, 0.010],
                [0.450, 0.540, 0.010],
            ]
        )
        repair_vote = np.asarray(
            [
                [0.45, 0.54, 0.01],
                [0.45, 0.54, 0.01],
                [0.45, 0.54, 0.01],
            ]
        )
        functional_vote = np.asarray(
            [
                [0.54, 0.45, 0.01],
                [0.54, 0.45, 0.01],
                [0.54, 0.45, 0.01],
            ]
        )
        components = {
            name: (repair_vote if position < 4 else functional_vote)
            for position, name in enumerate(REPAIR_META_COMPONENTS)
        }

        selected = _consensus_repair_mask(base, components)

        np.testing.assert_array_equal(selected, [True, False, False])

    def test_consensus_rejects_an_incomplete_component_set(self) -> None:
        base = np.asarray([[0.505, 0.490, 0.005]])

        with self.assertRaisesRegex(KeyError, "incomplete"):
            _consensus_repair_mask(base, {})


if __name__ == "__main__":
    unittest.main()

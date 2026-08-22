"""Tests for nested confident-disagreement label filtering."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from confident_label_filtering import select_confident_label_disagreements


class ConfidentLabelFilteringTests(unittest.TestCase):
    def test_only_strong_low_observed_probability_disagreements_are_selected(
        self,
    ) -> None:
        target = pd.Series(
            [
                "functional",
                "functional",
                "functional needs repair",
                "non functional",
            ]
        )
        probabilities = np.asarray(
            [
                [0.04, 0.01, 0.95],
                [0.06, 0.01, 0.93],
                [0.02, 0.94, 0.04],
                [0.91, 0.04, 0.05],
            ]
        )

        selected = select_confident_label_disagreements(target, probabilities)

        np.testing.assert_array_equal(selected, [True, False, False, True])

    def test_unknown_observed_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected observed label"):
            select_confident_label_disagreements(
                pd.Series(["unknown"]),
                np.asarray([[0.95, 0.03, 0.02]]),
            )


if __name__ == "__main__":
    unittest.main()

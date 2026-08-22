"""Focused tests for probability reliability and edge-case diagnostics."""

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

from probability_diagnostics import CLASS_LABELS
from probability_diagnostics import class_membership_summary
from probability_diagnostics import class_membership_table
from probability_diagnostics import classwise_probability_scores
from probability_diagnostics import component_disagreement_summary
from probability_diagnostics import edge_case_table
from probability_diagnostics import membership_ambiguity_summary
from probability_diagnostics import probability_quality_summary
from probability_diagnostics import top_two_membership_summary
from probability_diagnostics import top_label_reliability
from probability_diagnostics import write_class_membership_table


class ProbabilityDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = pd.Series(CLASS_LABELS * 2)
        self.probabilities = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.1, 0.8],
                [0.4, 0.3, 0.3],
                [0.5, 0.4, 0.1],
                [0.1, 0.2, 0.7],
            ]
        )

    def test_quality_summary_reports_confidence_errors_and_brier(self) -> None:
        result = probability_quality_summary(self.target, self.probabilities)

        self.assertAlmostEqual(result["accuracy"], 5 / 6)
        self.assertGreater(result["multiclass_brier"], 0)
        self.assertEqual(result["high_confidence_error_rows"], 0)

    def test_reliability_bins_cover_every_row(self) -> None:
        result = top_label_reliability(self.target, self.probabilities)

        self.assertEqual(int(result["rows"].sum()), len(self.target))

    def test_classwise_scores_preserve_all_classes(self) -> None:
        result = classwise_probability_scores(self.target, self.probabilities)

        self.assertEqual(result.index.tolist(), list(CLASS_LABELS))

    def test_component_disagreement_detects_changed_winner(self) -> None:
        changed = self.probabilities.copy()
        changed[0] = [0.1, 0.8, 0.1]

        result = component_disagreement_summary(
            {"first": self.probabilities, "second": changed}
        )

        self.assertEqual(result["non_unanimous_rows"], 1)

    def test_edge_table_separates_repair_false_alarms_and_misses(self) -> None:
        result = edge_case_table(self.target, self.probabilities)

        self.assertEqual(result.loc["repair false alarm", "rows"], 0)
        self.assertEqual(result.loc["repair missed as functional", "rows"], 1)

    def test_membership_table_retains_all_probabilities_and_ambiguity(self) -> None:
        table = class_membership_table(
            pd.Series(range(100, 106)),
            self.probabilities,
            actual=self.target,
        )
        summary = class_membership_summary(table)
        ambiguity = membership_ambiguity_summary(table)
        boundaries = top_two_membership_summary(table)

        probability_columns = [
            "membership_functional",
            "membership_functional_needs_repair",
            "membership_non_functional",
        ]
        np.testing.assert_allclose(table[probability_columns].sum(axis=1), 1.0)
        self.assertEqual(table.loc[1, "predicted_class"], CLASS_LABELS[1])
        self.assertEqual(table.loc[1, "runner_up_class"], CLASS_LABELS[0])
        self.assertEqual(summary.index.tolist(), list(CLASS_LABELS))
        self.assertEqual(ambiguity["rows"], len(table))
        self.assertEqual(ambiguity["no_majority_rows"], 1)
        self.assertEqual(int(boundaries["rows"].sum()), len(table))

    def test_membership_writer_verifies_existing_csv(self) -> None:
        table = class_membership_table(
            pd.Series(range(100, 106)),
            self.probabilities,
            actual=self.target,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "memberships.csv"

            first = write_class_membership_table(table, destination)
            second = write_class_membership_table(table, destination)

            self.assertEqual(first, second)
            self.assertEqual(len(pd.read_csv(destination)), len(table))
            changed = table.copy()
            changed.loc[0, "confidence"] = 0.7
            with self.assertRaises(FileExistsError):
                write_class_membership_table(changed, destination)


if __name__ == "__main__":
    unittest.main()

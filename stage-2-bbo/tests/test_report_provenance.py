"""Reject stale source/results before publishing current-source proposal records."""

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / "src"))
from report_provenance import (notebook_path, report_inputs, runtime_versions,
                               verify_report_provenance, write_report_provenance)


class ReportProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.stage = self.root / "stage-2-bbo"
        for name in ("src/calculation.py", "scripts/run_initial_analysis.py"):
            path = self.stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("original source", encoding="utf-8")
        (self.root / "requirements.txt").write_text("numpy\n", encoding="utf-8")

    def make_report(self, function=2, suffix="risk"):
        source = notebook_path(self.stage, function, suffix)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("notebook source", encoding="utf-8")
        data = self.stage / "data/initial_data" / f"function_{function}"
        data.mkdir(parents=True, exist_ok=True)
        for name in ("initial_inputs.npy", "initial_outputs.npy"):
            (data / name).write_bytes(b"original data")
        output = self.root / f".runtime/function-{function}-{suffix}"
        output.mkdir(parents=True, exist_ok=True)
        for name in ("analysis.html", "analysis.ipynb", "risk-study.json", "selection-study.json"):
            (output / name).write_text("original artefact", encoding="utf-8")
        write_report_provenance(self.stage, function, suffix,
                                report_inputs(self.stage, function, suffix))
        return output

    def test_matching_report_and_numerical_results_are_accepted(self):
        for function in (2, 3):
            for suffix in ("initial-analysis", "risk"):
                self.make_report(function, suffix)
                record = verify_report_provenance(self.stage, function, suffix)
                self.assertEqual(record["function"], function)

    def test_source_data_notebook_and_artefact_changes_are_rejected(self):
        output = self.make_report()
        paths = [self.stage / "src/calculation.py",
                 notebook_path(self.stage, 2, "risk"),
                 self.stage / "data/initial_data/function_2/initial_outputs.npy",
                 output / "risk-study.json", output / "analysis.html",
                 output / "analysis.ipynb", self.root / "requirements.txt"]
        for path in paths:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "stale.*--function 2 --decision-risk"):
                    verify_report_provenance(self.stage, 2, "risk")
                path.write_bytes(original)

    def test_selection_results_cannot_be_replaced_after_rendering(self):
        output = self.make_report(2, "initial-analysis")
        (output / "selection-study.json").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_report_provenance(self.stage, 2, "initial-analysis")

    def test_missing_manifest_and_changed_runtime_are_rejected(self):
        output = self.make_report()
        changed = {**runtime_versions(), "numpy": "different version"}
        with patch("report_provenance.runtime_versions", return_value=changed):
            with self.assertRaises(ValueError):
                verify_report_provenance(self.stage, 2, "risk")
        (output / "provenance.json").unlink()
        with self.assertRaises(ValueError):
            verify_report_provenance(self.stage, 2, "risk")

    def test_changes_during_execution_are_not_attested(self):
        output = self.make_report()
        original_manifest = (output / "provenance.json").read_bytes()
        before = report_inputs(self.stage, 2, "risk")
        (self.stage / "src/calculation.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed during execution"):
            write_report_provenance(self.stage, 2, "risk", before)
        self.assertEqual((output / "provenance.json").read_bytes(), original_manifest)

    def test_builder_checks_all_reports_before_modifying_existing_pack(self):
        for function in range(1, 9):
            for suffix in ("initial-analysis", "risk"):
                self.make_report(function, suffix)
        (self.root / ".runtime/function-8-risk/risk-study.json").write_text("changed", encoding="utf-8")
        targets = [self.root / ".runtime/first-query-review/index.html",
                   self.root / ".runtime/capstone-first-query-pack.zip",
                   self.stage / "submissions/round-01-proposals.json"]
        for path in targets:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"existing pack")
        spec = importlib.util.spec_from_file_location("review_builder", STAGE / "scripts/build_first_query_review.py")
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with patch.object(builder, "stage", self.stage), patch.object(builder, "root", self.root):
            with self.assertRaisesRegex(ValueError, "Function 8 risk"):
                builder.main()
        for path in targets:
            self.assertEqual(path.read_bytes(), b"existing pack")


if __name__ == "__main__":
    unittest.main()

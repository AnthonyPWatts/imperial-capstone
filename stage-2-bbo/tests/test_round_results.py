"""Protect exact source observations and repeatable result ingestion."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from import_round_results import DIMENSIONS, import_round


class RoundResultsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.stage = Path(self.temporary.name)
        self.source = self.stage / "data/rounds/round-01"
        self.source.mkdir(parents=True)
        points = [[0.5] * dimensions for dimensions in DIMENSIONS]
        self.source.joinpath("inputs.txt").write_text("[" + ", ".join(f"array({p!r})" for p in points) + "]")
        self.source.joinpath("outputs.txt").write_text(repr([-0.005] * 8))
        self.source.joinpath("email-body.txt").write_text("\n".join(
            [f"Function {i}: {p!r}" for i, p in enumerate(points, 1)] +
            [f"Function {i}: -0.005" for i in range(1, 9)]))
        self.source.joinpath("source.json").write_text(json.dumps({
            "round": 1, "received_utc": "2026-10-04T23:38:04+00:00",
            "received_local": "2026-10-05T00:38:04+01:00", "source_commit": "test"}))
        submissions = self.stage / "submissions"
        submissions.mkdir()
        submissions.joinpath("round-01-proposals.json").write_text(json.dumps({
            "round": 1, "proposals": [{"function": i, "portal_input": "-".join(["0.500000"] * d)}
                                      for i, d in enumerate(DIMENSIONS, 1)]}))
        for i, dimensions in enumerate(DIMENSIONS, 1):
            initial = self.stage / "data/initial_data" / f"function_{i}"
            initial.mkdir(parents=True)
            np.save(initial / "initial_inputs.npy", [[0.1] * dimensions, [0.9] * dimensions])
            np.save(initial / "initial_outputs.npy", [1e-124, 7e-16])

    def test_import_preserves_tiny_outputs_and_originals_and_is_idempotent(self):
        original = {p: p.read_bytes() for p in (self.stage / "data/initial_data").rglob("*.npy")}
        record = import_round(self.stage, 1)
        first = {p: p.read_bytes() for p in (self.stage / "data/accumulated").rglob("*") if p.is_file()}
        self.assertEqual(record, import_round(self.stage, 1))
        self.assertTrue(all(p.read_bytes() == value for p, value in first.items()))
        self.assertTrue(all(p.read_bytes() == value for p, value in original.items()))
        y = np.load(self.stage / "data/accumulated/function_1/outputs.npy")
        np.testing.assert_array_equal(y, [1e-124, 7e-16, -0.005])
        self.assertFalse(record["results"][0]["improved_incumbent"])

    def test_bad_last_function_is_rejected_before_any_dataset_write(self):
        self.source.joinpath("outputs.txt").write_text(repr([-0.005] * 7 + [9]))
        with self.assertRaisesRegex(ValueError, "disagree"):
            import_round(self.stage, 1)
        self.assertFalse((self.stage / "data/accumulated").exists())

    def test_unexpected_proposal_is_rejected(self):
        path = self.stage / "submissions/round-01-proposals.json"
        proposal = json.loads(path.read_text())
        proposal["proposals"][-1]["portal_input"] = "unexpected"
        path.write_text(json.dumps(proposal))
        with self.assertRaisesRegex(ValueError, "does not match"):
            import_round(self.stage, 1)
        self.assertFalse((self.stage / "data/accumulated").exists())

    def test_non_literal_attachment_is_rejected_without_execution(self):
        self.source.joinpath("inputs.txt").write_text("[__import__('os')] * 8")
        with self.assertRaises(ValueError):
            import_round(self.stage, 1)

    def test_out_of_bounds_coordinate_is_rejected(self):
        path = self.source / "inputs.txt"
        path.write_text(path.read_text().replace("0.5", "1.0", 1))
        with self.assertRaisesRegex(ValueError, "bounds"):
            import_round(self.stage, 1)


if __name__ == "__main__":
    unittest.main()

"""Validate an emailed round and rebuild accumulated observations from sources."""

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np


DIMENSIONS = (2, 2, 3, 4, 4, 5, 6, 8)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_return(directory):
    """Accept only the attachment's list of array literals; never execute it."""
    expression = ast.parse((directory / "inputs.txt").read_text(), mode="eval").body
    if not isinstance(expression, ast.List) or len(expression.elts) != 8:
        raise ValueError("Expected eight input arrays.")
    points = []
    for item, dimensions in zip(expression.elts, DIMENSIONS):
        if (not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name)
                or item.func.id != "array" or len(item.args) != 1 or item.keywords):
            raise ValueError("Expected array([...]) literals only.")
        point = np.asarray(ast.literal_eval(item.args[0]), dtype=float)
        if (point.shape != (dimensions,) or not np.isfinite(point).all()
                or np.any(point < 0) or np.any(point >= 1)):
            raise ValueError("Invalid input dimensions, finiteness or [0, 1) bounds.")
        if not np.array_equal(point, np.round(point, 6)):
            raise ValueError("Returned coordinates exceed portal precision.")
        points.append(point)
    outputs = np.asarray(ast.literal_eval((directory / "outputs.txt").read_text()), dtype=float)
    if outputs.shape != (8,) or not np.isfinite(outputs).all():
        raise ValueError("Expected eight finite scalar outputs.")
    body = (directory / "email-body.txt").read_text(encoding="utf-8")
    input_lines = re.findall(r"^Function (\d+): (\[.*\])$", body, re.MULTILINE)
    output_lines = re.findall(r"^Function (\d+): ([-+0-9.eE]+)$", body, re.MULTILINE)
    if [int(n) for n, _ in input_lines] != list(range(1, 9)) or [int(n) for n, _ in output_lines] != list(range(1, 9)):
        raise ValueError("Email body must identify each function once for inputs and outputs.")
    for index, ((_, point), (_, value)) in enumerate(zip(input_lines, output_lines)):
        if not np.array_equal(points[index], ast.literal_eval(point)) or outputs[index] != float(value):
            raise ValueError("Email body and attachments disagree.")
    return points, outputs


def import_round(stage, round_number):
    if round_number < 1:
        raise ValueError("Round number must be positive.")
    rounds_root = stage / "data" / "rounds"
    directories = sorted(rounds_root.glob("round-[0-9][0-9]"))
    if [p.name for p in directories] != [f"round-{n:02d}" for n in range(1, round_number + 1)]:
        raise ValueError("Import the latest round with a complete consecutive source history.")
    rounds = [read_return(directory) for directory in directories]
    source = json.loads((directories[-1] / "source.json").read_text(encoding="utf-8"))
    if source["round"] != round_number:
        raise ValueError("Source metadata round does not match.")
    proposals_path = stage / "submissions" / f"round-{round_number:02d}-proposals.json"
    proposals = json.loads(proposals_path.read_text(encoding="utf-8"))
    if proposals["round"] != round_number or sorted(p["function"] for p in proposals["proposals"]) != list(range(1, 9)):
        raise ValueError("Expected one proposal per function for this round.")
    proposed = {p["function"]: p for p in proposals["proposals"]}
    prepared, results, source_hashes = [], [], {}
    # Validate every function before writing any derived dataset.
    for function, dimensions in enumerate(DIMENSIONS, 1):
        initial = stage / "data" / "initial_data" / f"function_{function}"
        xp, yp = initial / "initial_inputs.npy", initial / "initial_outputs.npy"
        x, y = np.load(xp, allow_pickle=False), np.load(yp, allow_pickle=False)
        if (x.ndim != 2 or x.shape[1] != dimensions or len(x) == 0 or y.shape != (len(x),)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(x < 0) or np.any(x >= 1) or len(np.unique(x, axis=0)) != len(x)):
            raise ValueError(f"Invalid initial observations for function {function}.")
        source_hashes.update({str(p.relative_to(stage)): sha256(p) for p in (xp, yp)})
        initial_count = len(x)
        for number, (points, outputs) in enumerate(rounds, 1):
            point, value = points[function - 1], outputs[function - 1]
            duplicate = bool(np.any(np.all(x == point, axis=1)))
            # A confirmed repeat is an observation, not a duplicate ingestion.
            # Rebuilding from unique round directories prevents repeated ingestion.
            before = float(y.max())
            if number == round_number:
                portal = "-".join(f"{coordinate:.6f}" for coordinate in point)
                if portal != proposed[function]["portal_input"]:
                    raise ValueError(f"Function {function} return does not match its proposal.")
            x, y = np.vstack([x, point]), np.append(y, value)
        results.append({"function": function, "point": point.tolist(), "portal_input": portal,
                        "output": float(value), "matches_proposal": True,
                        "repeated_input": duplicate, "initial_observations": initial_count,
                        "observations": len(x), "best_before": before,
                        "best_after": float(y.max()), "improved_incumbent": bool(value > before)})
        text = io.StringIO(newline="")
        writer = csv.writer(text)
        writer.writerow(["observation_id", "source_round", *[f"x{i + 1}" for i in range(dimensions)], "y"])
        for index, (point, value) in enumerate(zip(x, y)):
            number = 0 if index < initial_count else index - initial_count + 1
            identity = f"initial-{index + 1:03d}" if number == 0 else f"round-{number:02d}"
            writer.writerow([identity, number, *[repr(float(v)) for v in point], repr(float(value))])
        prepared.append((function, x, y, text.getvalue()))
    for directory in directories:
        for filename in ("inputs.txt", "outputs.txt", "email-body.txt", "source.json"):
            path = directory / filename
            source_hashes[str(path.relative_to(stage))] = sha256(path)
    record = {"round": round_number, "status": "evaluation_confirmed_by_results_email",
              "received_utc": source["received_utc"], "received_local": source["received_local"],
              "source_commit_at_intake": source["source_commit"],
              "proposals_sha256": sha256(proposals_path), "source_sha256": source_hashes,
              "results": results}
    for function, x, y, text in prepared:
        destination = stage / "data" / "accumulated" / f"function_{function}"
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "inputs.npy", x, allow_pickle=False)
        np.save(destination / "outputs.npy", y, allow_pickle=False)
        (destination / "observations.csv").write_bytes(text.encode("utf-8"))
    target = stage / "submissions" / f"round-{round_number:02d}-results.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    args = parser.parse_args()
    record = import_round(Path(__file__).resolve().parents[1], args.round)
    print(f"Imported round {args.round}: {sum(r['observations'] for r in record['results'])} accumulated observations.")

"""Build a local pooled snapshot from pinned, published evaluated histories.

Run with the capstone environment from the repository root. Source downloads
and the full cohort inventory are retained under .runtime/cohort-side-quest/.
Only literal data are read from peer Python and notebook sources; none execute.
"""

import ast
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import nbformat
import numpy as np
import pandas as pd


STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[2]
NOTEBOOKS = ROOT / "stage-2-bbo/notebooks"
CACHE = ROOT / ".runtime/cohort-side-quest"
SOURCES = CACHE / "sources"
DIMENSIONS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}


def same_reported_output(existing, reported):
    """Match reported rounding without equating zero or opposite signs."""
    value = float(reported)
    if value == existing:
        return True
    if value == 0 or existing == 0 or np.signbit(value) != np.signbit(existing):
        return False
    tolerance = float(Decimal(1).scaleb(Decimal(str(reported)).as_tuple().exponent) / 2)
    return abs(value - existing) <= tolerance


def literal_assignment(source, name):
    """Extract a named literal assignment without running untrusted code."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal assignment: {name}")


def workbook_rows(path, sheet):
    """Read literal/cached values from the small source tracker, without Excel."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(item.itertext()) for item in
                       ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", ns)]
        for row in ET.fromstring(archive.read(sheet)).findall(".//s:row", ns):
            cells = {}
            for cell in row.findall("s:c", ns):
                value, inline = cell.find("s:v", ns), cell.find("s:is", ns)
                text = value.text if value is not None else "".join(inline.itertext()) if inline is not None else ""
                if cell.get("t") == "s":
                    text = strings[int(text)]
                cells[re.sub(r"\d", "", cell.get("r"))] = text
            yield int(row.get("r")), cells


def build():
    downloads = json.loads((CACHE / "downloads.json").read_text(encoding="utf-8"))
    source_index = {(item["repository"], item["path"]): item for item in downloads if "error" not in item}
    observations, provenance, exclusions, duplicates = {}, [], [], []
    base_data = {}
    for number, dimension in DIMENSIONS.items():
        data = pd.read_csv(NOTEBOOKS / f"Function_{number}/Week_04/observations.csv", float_precision="round_trip")
        base_data[number] = data
        observations[number] = data.to_dict("records")
        for row in observations[number]:
            provenance.append({"function": number, "observation_id": row["observation_id"],
                               "repository": "AnthonyPWatts/imperial-capstone",
                               "source_round": int(row["source_round"]),
                               "path": f"notebooks/Function_{number}/Week_04/observations.csv",
                               "source": "local authorised initial data" if row["source_round"] == 0 else "confirmed portal return",
                               "commit": "c0c23aebdcf6c2aae50c90c250fa704309b1727e"})

    def add(number, point, output, round_number, repository, path, location):
        record = {"function": number, "inputs": point, "output": output, "source_round": round_number,
                  "repository": repository, "path": path, "location": location}
        source = source_index[(repository, path)]
        record.update({key: source[key] for key in ("url", "commit", "sha256")})
        if output is None or str(output).strip().lower() in ("", "pending", "none", "nan"):
            exclusions.append({**record, "reason": "No returned output"})
            return
        values = [float(value) for value in point]
        value = float(output)
        if len(values) != DIMENSIONS[number] or not np.isfinite([*values, value]).all():
            exclusions.append({**record, "reason": "Invalid dimension or non-finite value"})
            return
        if not all(0 <= coordinate < 1 for coordinate in values):
            exclusions.append({**record, "reason": "Reported input outside [0, 1); no clipping or guessing"})
            return
        inputs = [f"x{index + 1}" for index in range(len(values))]
        key = tuple(round(coordinate, 6) for coordinate in values)
        # A source rounded to six decimals is not a second measurement merely
        # because the original source retained more digits. Match the reported
        # output's rounding interval, never a broad absolute tolerance near zero.
        for existing in observations[number]:
            if tuple(round(float(existing[column]), 6) for column in inputs) == key:
                if same_reported_output(existing["y"], output):
                    duplicates.append({**record, "matched_observation_id": existing["observation_id"]})
                    return
        identity = f"peer-{repository.split('/')[0]}-round-{round_number:02d}"
        if any(row["observation_id"] == identity for row in observations[number]):
            raise ValueError(f"More than one source row has identity {identity} for Function {number}")
        observations[number].append({"observation_id": identity, "source_round": round_number,
                                     **dict(zip(inputs, values)), "y": value})
        provenance.append({**record, "observation_id": identity})

    # Verify that published initial arrays describe exactly the same problems.
    checked_arrays = []
    for item in downloads:
        match = re.search(r"function_([1-8])/initial_inputs\.npy$", item.get("path", ""))
        if not match or "error" in item:
            continue
        number = int(match[1])
        directory = SOURCES / item["repository"] / Path(item["path"]).parent
        x = np.load(directory / "initial_inputs.npy", allow_pickle=False)
        y = np.load(directory / "initial_outputs.npy", allow_pickle=False).reshape(-1)
        initial = base_data[number].loc[base_data[number].source_round.eq(0)]
        inputs = [f"x{i + 1}" for i in range(DIMENSIONS[number])]
        if not np.array_equal(x, initial[inputs].to_numpy()) or not np.array_equal(y, initial.y.to_numpy()):
            raise ValueError(f"Different initial problem data: {item['repository']}, Function {number}")
        checked_arrays.append({"repository": item["repository"], "function": number,
                               "rows": len(y), "status": "Exact copy of shared initial observations; included once"})

    repository, path = "kosina/bbocapstone", "capstone_bbo.ipynb"
    notebook = nbformat.read(SOURCES / repository / path, as_version=4)
    cell = next(cell for cell in notebook.cells if cell.cell_type == "code" and "HISTORY = {" in cell.source)
    history = literal_assignment(cell.source, "HISTORY")
    for week, results in history.items():
        for number, (query, output) in results.items():
            add(number, query.split("-"), output, week, repository, path, f"HISTORY[{week}][{number}]")

    repository, path = "mkamar17/bbocapstone", "project/stage-two/query_history.py"
    history = literal_assignment((SOURCES / repository / path).read_text(encoding="utf-8"), "query_history")
    for number, records in history.items():
        for index, (point, output) in enumerate(records, 1):
            add(number, point, output, index, repository, path, f"query_history[{number}][{index - 1}] (ordered query history)")

    repository, path = "tomxandrews/bbocapstone-s1", "capstone_bbo_master_tracker.xlsx"
    for row_number, row in workbook_rows(SOURCES / repository / path, "xl/worksheets/sheet2.xml"):
        if not row.get("A", "").startswith("Week ") or not row.get("B", "").startswith("Function "):
            continue
        add(int(row["B"].split()[-1]), row["E"].split("-"), row.get("F"),
            int(row["A"].split()[-1]), repository, path, f"Weekly Submissions Log!E{row_number}:F{row_number}")

    repository = "arwelbrown/black-box-capstone"
    for number in DIMENSIONS:
        path = f"history/function_{number}_history.csv"
        frame = pd.read_csv(SOURCES / repository / path, dtype=str, keep_default_na=False)
        inputs = [f"x{i + 1}" for i in range(DIMENSIONS[number])]
        for index, row in frame.iterrows():
            add(number, [row[column] for column in inputs], row["y"], int(row.iteration),
                repository, path, f"CSV row {index + 2}")

    summary = []
    for number, rows in observations.items():
        frame = pd.DataFrame(rows).sort_values(["source_round", "observation_id"], kind="stable").reset_index(drop=True)
        directory = STUDY / f"notebooks/Function_{number}/Week_04"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "observations.csv"
        frame.to_csv(path, index=False)
        restored = pd.read_csv(path, float_precision="round_trip")
        pd.testing.assert_frame_equal(frame, restored)
        initial_best = float(frame.loc[frame.source_round.eq(0), "y"].max())
        summary.append({"function": number, "initial": int(frame.source_round.eq(0).sum()),
                        "own_returned": len(base_data[number]) - int(frame.source_round.eq(0).sum()),
                        "peer_added": len(frame) - len(base_data[number]), "total": len(frame),
                        "own_best": float(base_data[number].y.max()), "pooled_best": float(frame.y.max()),
                        "pooled_best_id": str(frame.loc[frame.y.idxmax(), "observation_id"]),
                        "initial_best": initial_best,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    audit = {"collected_on": "2026-10-20", "purpose": "Exploratory side quest; official Week 4 submission is unchanged",
             "initial_array_checks": checked_arrays, "summary": summary,
             "duplicate_records": duplicates, "excluded_records": exclusions,
             "provenance": provenance,
             "sources": [source_index[key] for key in sorted({(row["repository"], row["path"])
                         for row in [*provenance, *duplicates, *exclusions] if "url" in row})]}
    (STUDY / "dataset-audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary)[["function", "initial", "own_returned", "peer_added", "total", "own_best", "pooled_best"]].to_string(index=False))
    print(f"Shared initial arrays checked: {len(checked_arrays)}; duplicate records: {len(duplicates)}; excluded/pending records: {len(exclusions)}")
if __name__ == "__main__":
    build()

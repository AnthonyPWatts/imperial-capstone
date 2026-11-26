"""Reconcile the public 2024 competition snapshots as inert query and score data."""

import argparse
import ast
from collections import defaultdict
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

STAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STAGE / "scripts"))
from import_round_results import DIMENSIONS, read_input_list

COMMIT = "27e7ed89734766f89ae23af1871deebb34f3bca4"
REPOSITORY = "https://github.com/wrm65/Capstone-Competition-2024"
DESTINATION = STAGE / "data/historical-cohort/2024"
FILENAME = re.compile(r"(data|results)/week_(\d+)/(\d+)[-_](queries|observations)\.txt")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_snapshot(queries, observations):
    inputs = ast.parse(queries.decode("utf-8-sig")).body
    outputs = ast.parse(observations.decode("utf-8-sig")).body
    if not inputs or len(inputs) != len(outputs):
        raise ValueError("The query and observation lists must have the same nonzero length")
    for input_row, output_row in zip(inputs, outputs, strict=True):
        if not isinstance(input_row, ast.Expr) or not isinstance(output_row, ast.Expr):
            raise ValueError("Expected only expressions containing literal arrays and scores")
        points = read_input_list(input_row.value)
        values = np.asarray(ast.literal_eval(output_row.value), dtype=float)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("Expected eight finite scores")
        yield tuple((tuple(point.tolist()), float(value)) for point, value in zip(points, values, strict=True))


def build(archive):
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(archive), *arguments])

    def blob(path):
        return git("show", f"{COMMIT}:{path}")

    starts = []
    starter_pool = {}
    for function, dimensions in enumerate(DIMENSIONS, 1):
        values = {}
        for kind in ("inputs", "outputs"):
            file = f"function_{function}/initial_{kind}.npy"
            shared = blob(f"initial_data1/{file}")
            if shared != (STAGE / "data/initial_data" / file).read_bytes():
                raise ValueError(f"The shared starter array differs: {file}")
            extra = blob(f"initial_data2/{file}")
            merged = blob(f"initial_data/{file}")
            first, second, combined = (np.load(BytesIO(data), allow_pickle=False)
                                       for data in (shared, extra, merged))
            if not np.array_equal(np.concatenate([first, second]), combined):
                raise ValueError(f"The merged archive starter array differs: {file}")
            starts.append({"path": f"initial_data1/{file}", "sha256": digest(shared),
                           "additional_path": f"initial_data2/{file}", "additional_sha256": digest(extra)})
            values[kind] = second
        x, y = values["inputs"], values["outputs"]
        if (x.shape != (len(y), dimensions) or y.ndim != 1 or not np.isfinite(x).all()
                or not np.isfinite(y).all() or (x < 0).any() or (x >= 1).any()):
            raise ValueError(f"Invalid additional starter arrays for Function {function}")
        starter_pool[function] = [
            {"observation_id": f"archive-2024-starter-{position:03d}", "participant": "", "position": 0,
             "point": point, "y": result, "source_copies": 1,
             "query_path": f"initial_data2/function_{function}/initial_inputs.npy",
             "output_path": f"initial_data2/function_{function}/initial_outputs.npy"}
            for position, (point, result) in enumerate(zip(x, y, strict=True), 1)
        ]

    paths = git("ls-tree", "-r", "--name-only", "-z", COMMIT).decode("utf-8").split("\0")
    pairs = defaultdict(dict)
    for path in paths:
        match = FILENAME.fullmatch(path)
        if match:
            folder, week, participant, role = match.groups()
            pair = pairs[(folder, int(week), participant)]
            if role in pair:
                raise ValueError(f"Duplicate {role} file for {folder}, week {week}, participant {participant}")
            pair[role] = path

    candidates = defaultdict(lambda: defaultdict(list))
    rejected = []
    files = {}
    for (folder, week, participant), pair in sorted(pairs.items()):
        if pair.keys() != {"queries", "observations"}:
            rejected.append({"paths": pair, "reason": "Unpaired query or observation file"})
            continue
        query, output = (blob(pair[role]) for role in ("queries", "observations"))
        files.update({pair["queries"]: digest(query), pair["observations"]: digest(output)})
        try:
            snapshot = list(read_snapshot(query, output))
        except (SyntaxError, ValueError) as error:
            rejected.append({"paths": pair, "reason": str(error)})
            continue
        for position, batch in enumerate(snapshot, 1):
            for function, measurement in enumerate(batch, 1):
                candidates[participant, position, function][measurement].append(
                    {"folder": folder, "snapshot_week": week, **pair})

    observations = defaultdict(list)
    conflicts = []
    redundant = 0
    for (participant, position, function), versions in sorted(candidates.items()):
        if len(versions) != 1:
            conflicts.append({"participant": participant, "position": position, "function": function,
                              "versions": [{"point": point, "output": result, "sources": locations}
                                           for (point, result), locations in versions.items()]})
            continue
        ((point, result), locations), = versions.items()
        source = min(locations, key=lambda row: (row["snapshot_week"], row["folder"] != "data"))
        observations[function].append({
            "observation_id": f"archive-2024-{participant}-{position:02d}",
            "participant": participant, "position": position, "point": point, "y": result,
            "query_path": source["queries"], "output_path": source["observations"],
            "source_copies": len(locations),
        })
        redundant += len(locations) - 1

    if sorted(observations) != list(range(1, 9)):
        raise ValueError("No consistent measured observations for one or more functions")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    dataset = {}
    for function, dimensions in enumerate(DIMENSIONS, 1):
        target = DESTINATION / f"function_{function}.csv"
        with target.open("w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(["observation_id", "participant", "position",
                             *(f"x{i}" for i in range(1, dimensions + 1)), "y",
                             "query_path", "output_path", "source_copies"])
            for row in [*starter_pool[function], *observations[function]]:
                writer.writerow([row["observation_id"], row["participant"], row["position"],
                                 *row["point"], row["y"], row["query_path"],
                                 row["output_path"], row["source_copies"]])
        dataset[function] = {"starter_rows": len(starter_pool[function]),
                             "query_rows": len(observations[function]),
                             "rows": len(starter_pool[function]) + len(observations[function]),
                             "sha256": digest(target.read_bytes())}
    audit = {
        "repository": REPOSITORY, "commit": COMMIT, "starter_matches": starts, "source_files": files,
        "file_pairs": len(pairs), "rejected_pairs": rejected, "redundant_cumulative_rows": redundant,
        "conflicts": conflicts, "participants": sorted({row["participant"] for rows in observations.values() for row in rows}),
        "functions": dataset,
    }
    (DESTINATION / "intake-audit.json").write_bytes((json.dumps(audit, indent=2) + "\n").encode())
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Local Git repository with the pinned commit")
    audit = build(parser.parse_args().archive.resolve())
    print(f"{sum(row['rows'] for row in audit['functions'].values())} observations across "
          f"{len(audit['participants'])} reporting IDs. {len(audit['conflicts'])} conflicting rows and "
          f"{len(audit['rejected_pairs'])} malformed or incomplete file pairs excluded.")

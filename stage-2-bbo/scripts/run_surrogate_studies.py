"""Execute the eight separate Week 5 discussion-evidence notebooks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functions", type=int, nargs="+", default=list(range(1, 9)), choices=range(1, 9))
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    expected = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError(f"Use the Capstone Python interpreter: {expected}")
    submitted = root / "stage-2-bbo/submissions/Week_05/submissions.txt"
    fingerprint = hashlib.sha256(submitted.read_bytes()).hexdigest()
    for number in arguments.functions:
        folder = root / f"stage-2-bbo/notebooks/Function_{number}/Week_05"
        if number <= 4:
            folder /= "official"
        path = folder / "surrogate_comparison.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        start = time.monotonic()
        print(f"Running Function {number}: {path.relative_to(root)}", flush=True)
        try:
            NotebookClient(notebook, timeout=900, kernel_name="python3",
                resources={"metadata": {"path": str(folder)}}).execute()
        finally:
            nbformat.write(notebook, path)
        assert hashlib.sha256(submitted.read_bytes()).hexdigest() == fingerprint, "Submitted inputs changed"
        summary = json.loads((folder / "surrogate-study/results.json").read_text(encoding="utf-8"))
        print(json.dumps({"function": number, "seconds": round(time.monotonic()-start, 1),
            "best_regressor": summary["regression_comparison"][0]["Model"],
            "best_classifier": summary["classification_comparison"][0]["Model"],
            "nn_regression_architecture": summary["selected_regression_architecture"],
            "nn_classification_architecture": summary["selected_classification_architecture"]}), flush=True)
    sys.path.insert(0, str(root / "stage-2-bbo/notebooks/Shared"))
    from surrogate_study_report import summarise_studies
    index = summarise_studies(root)
    if index is not None:
        print(f"Saved cross-function index: {index.relative_to(root)}", flush=True)


if __name__ == "__main__":
    main()

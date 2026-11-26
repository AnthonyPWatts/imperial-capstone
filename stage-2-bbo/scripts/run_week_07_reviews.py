"""Execute Week 7 reviews, including Function 1's activity and source studies."""

import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main(*, week=7, forecast_sources=None, full_cohort=False, decision_reference_sources=None):
    parser = argparse.ArgumentParser(description=f"Execute Week {week} observation and model reviews.")
    parser.add_argument("--functions", type=int, nargs="+", default=list(range(1, 9)), choices=range(1, 9))
    study_kinds = ("activity", "source") if week == 7 else ()
    if decision_reference_sources is not None:
        study_kinds = (*study_kinds, "decision")
    parser.add_argument("--kind", choices=("observations", "models", *study_kinds, "all"), default="all")
    arguments = parser.parse_args()
    if arguments.kind in ("activity", "source") and arguments.functions != [1]:
        parser.error("The activity/source comparisons are for Function 1; supply --functions 1.")
    root = Path(__file__).resolve().parents[2]
    expected = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError(f"Use the Capstone Python interpreter: {expected}")
    stage = root / "stage-2-bbo"
    previous = stage / f"submissions/Week_{week-1:02d}"
    sources = [previous / "submissions.txt"]
    if forecast_sources is None:
        sources.append(previous / "proposals.json")
    else:
        sources.extend(forecast_sources(stage))
    sources.extend(stage.glob(f"notebooks/Function_*/Week_{week:02d}/**/observations.csv"))
    if arguments.kind == "decision":
        for function in arguments.functions:
            current = stage / f"notebooks/Function_{function}/Week_{week:02d}"
            official = current / "official" if function <= 4 else current
            sources.extend(official / name for name in ("model-review.json", "observation-review.json"))
        sources.extend(decision_reference_sources(stage))
    fingerprints = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    observation_names = ("exploration.ipynb", "exploration-review.ipynb", "exploration-review.ipynb",
                         *("observation_review.ipynb",) * 5)
    for function in arguments.functions:
        folder = stage / f"notebooks/Function_{function}/Week_{week:02d}"
        if full_cohort:
            folder /= "full-cohort"
        elif function <= 4 and arguments.kind != "decision":
            folder /= "official"
        names = []
        if arguments.kind == "decision":
            names.append("weekly_review.ipynb")
        if arguments.kind in ("observations", "all"):
            names.append("observation_review.ipynb" if full_cohort else observation_names[function-1])
        if arguments.kind in ("models", "all"):
            names.append("model_predictions.ipynb")
        if week == 7 and function == 1 and arguments.kind in ("activity", "all"):
            names.append("activity_hypothesis.ipynb")
        if week == 7 and function == 1 and arguments.kind in ("source", "all"):
            names.append("source_shape_comparison.ipynb")
        for name in names:
            path = folder / name
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            start = time.monotonic()
            print(f"Running F{function}: {path.relative_to(root)}", flush=True)
            manager = KernelManager(kernel_name="python3")
            manager.kernel_spec.argv = [str(expected), "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            try:
                NotebookClient(notebook, timeout=900, km=manager,
                               resources={"metadata": {"path": str(folder)}}).execute()
            finally:
                nbformat.write(notebook, path)
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            errors = [output for cell in notebook.cells for output in cell.get("outputs", [])
                      if output.output_type == "error"]
            if errors:
                raise RuntimeError(f"Saved errors in {path}")
            for source, fingerprint in fingerprints.items():
                if hashlib.sha256(source.read_bytes()).hexdigest() != fingerprint:
                    raise RuntimeError(f"An input source changed: {source}")
            print(f"Completed F{function} {name}: {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

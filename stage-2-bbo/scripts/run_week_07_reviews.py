"""Execute the Week 7 observation reviews and model comparisons with the local kernel."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functions", type=int, nargs="+", default=list(range(1, 9)), choices=range(1, 9))
    parser.add_argument("--kind", choices=("observations", "models", "all"), default="all")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    expected = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError(f"Use the Capstone Python interpreter: {expected}")
    stage = root / "stage-2-bbo"
    sources = [stage / "submissions/Week_06/proposals.json", stage / "submissions/Week_06/submissions.txt"]
    sources.extend(stage.glob("notebooks/Function_*/Week_07/**/observations.csv"))
    fingerprints = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    observation_names = ("exploration.ipynb", "exploration-review.ipynb", "exploration-review.ipynb",
                         *("observation_review.ipynb",) * 5)
    for function in arguments.functions:
        folder = stage / f"notebooks/Function_{function}/Week_07"
        if function <= 4:
            folder /= "official"
        names = []
        if arguments.kind in ("observations", "all"):
            names.append(observation_names[function-1])
        if arguments.kind in ("models", "all"):
            names.append("model_predictions.ipynb")
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

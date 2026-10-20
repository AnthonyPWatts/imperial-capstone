"""Execute the independent copies; never submit queries."""

import argparse
import json
from pathlib import Path
import sys
import time

from jupyter_client import KernelManager
from nbclient import NotebookClient
import nbformat


STUDY = Path(__file__).resolve().parent


def run(numbers):
    progress = []
    for number in numbers:
        week = STUDY / f"notebooks/Function_{number}/Week_04"
        names = ("exploration.ipynb" if number <= 3 else "observation_review.ipynb", "model_predictions.ipynb")
        for name in names:
            path = week / name
            notebook = nbformat.read(path, as_version=4)
            kernel = KernelManager(kernel_name="python3")
            kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            client = NotebookClient(notebook, km=kernel, timeout=180,
                                    resources={"metadata": {"path": str(week)}})
            started = time.monotonic()
            print(f"Running Function {number}: {name}", flush=True)
            with client.setup_kernel(cleanup_kc=True):
                client.execute()
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            codes = [cell for cell in notebook.cells if cell.cell_type == "code"]
            assert all(cell.execution_count is not None for cell in codes)
            assert not any(output.output_type == "error" for cell in codes for output in cell.outputs)
            result = {"function": number, "notebook": name, "executed_cells": len(codes),
                      "seconds": round(time.monotonic() - started, 1)}
            progress.append(result)
            (STUDY / "run-results.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
            print(f"Passed: {len(codes)} code cells in {result['seconds']} seconds", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functions", type=int, nargs="+", default=list(range(1, 9)))
    run(parser.parse_args().functions)

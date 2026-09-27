"""Bind an executed reading copy to its data, source and numerical artefacts."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform


def notebook_path(stage, function, suffix):
    if suffix == "batch-coverage":
        name = "01b-function-1-batch-coverage.ipynb"
    elif suffix == "risk":
        name = ("01d-function-2-decision-risk.ipynb" if function == 2
                else f"02-function-{function}-decision-risk.ipynb")
    else:
        name = {1: "01-data-validation-and-exploration.ipynb",
                2: "01c-function-2-initial-exploration.ipynb",
                3: "01e-function-3-initial-exploration.ipynb"}.get(
                    function, f"01-function-{function}-initial-exploration.ipynb")
    return stage / "notebooks" / name


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_inputs(stage, function, suffix):
    data = stage / "data" / "initial_data" / f"function_{function}"
    sources = [*sorted((stage / "src").glob("*.py")),
               stage / "scripts" / "run_initial_analysis.py",
               stage.parent / "requirements.txt",
               notebook_path(stage, function, suffix),
               data / "initial_inputs.npy", data / "initial_outputs.npy"]
    return {path.relative_to(stage.parent).as_posix(): file_hash(path) for path in sources}


def runtime_versions():
    return {"python": platform.python_version(), **{
        name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn",
                                         "matplotlib", "nbconvert", "nbformat", "nbclient",
                                         "ipykernel")}}


def report_artefacts(output, function, suffix):
    names = ["analysis.html", "analysis.ipynb"]
    if suffix == "risk":
        names.append("risk-study.json")
    elif function == 2 and suffix == "initial-analysis":
        names.append("selection-study.json")
    return {name: file_hash(output / name) for name in names}


def write_report_provenance(stage, function, suffix, inputs_before):
    """Call only after successful execution/export, using the pre-run snapshot."""
    if report_inputs(stage, function, suffix) != inputs_before:
        raise ValueError("Report source or data changed during execution; rerun the report.")
    output = stage.parent / f".runtime/function-{function}-{suffix}"
    record = {"version": 1, "function": function, "suffix": suffix,
              "inputs": inputs_before,
              "runtime": runtime_versions(),
              "artefacts": report_artefacts(output, function, suffix)}
    (output / "provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def verify_report_provenance(stage, function, suffix):
    output = stage.parent / f".runtime/function-{function}-{suffix}"
    command = f"stage-2-bbo/scripts/run_initial_analysis.py --function {function}"
    if suffix == "risk":
        command += " --decision-risk"
    elif suffix == "batch-coverage":
        command += " --batch-coverage"
    try:
        record = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        if (record["version"] != 1 or record["function"] != function
                or record["suffix"] != suffix
                or record["runtime"] != runtime_versions()
                or record["inputs"] != report_inputs(stage, function, suffix)
                or record["artefacts"] != report_artefacts(output, function, suffix)):
            raise ValueError("source, data or generated files do not match")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"Function {function} {suffix} is stale or lacks verified provenance. "
                         f"Rerun: python {command}") from error
    return record

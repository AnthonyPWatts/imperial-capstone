"""Execute a function exploration and export a reading view without code."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import sys

from jupyter_client import KernelManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter
import nbformat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", type=int, choices=range(1, 9), default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--batch-coverage", action="store_true",
                        help="Run the follow-on analysis of planning two or three points.")
    mode.add_argument("--decision-risk", action="store_true",
                      help="Run the follow-on decision-risk study for the selected function.")
    args = parser.parse_args()
    if args.batch_coverage and args.function != 1:
        parser.error("The batch-coverage study currently covers Function 1 only.")

    stage = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(stage / "src"))
    from report_provenance import notebook_path, report_inputs, write_report_provenance

    suffix = ("batch-coverage" if args.batch_coverage else "risk" if args.decision_risk
              else "initial-analysis")
    source = notebook_path(stage, args.function, suffix)
    inputs_before = report_inputs(stage, args.function, suffix)
    output = stage.parent / ".runtime" / f"function-{args.function}-{suffix}"
    output.mkdir(parents=True, exist_ok=True)

    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)

    # Use this script's environment, irrespective of the user's default kernel.
    kernel = KernelManager(kernel_name="python3")
    kernel.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
    ]
    client = NotebookClient(
        notebook, km=kernel, timeout=int(notebook.metadata.get("execution_timeout", 120)),
        resources={"metadata": {"path": str(stage)}},
    )
    with client.setup_kernel(cleanup_kc=True):
        client.execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, output / "analysis.ipynb")
    print(f"Executed Function {args.function}: {output / 'analysis.ipynb'}")
    exporter = HTMLExporter(template_name="basic")
    exporter.exclude_input = True
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _ = exporter.from_notebook_node(notebook)
    title = notebook.metadata.get("reading_title", "Function 1 — from observations to a first query")
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>""" + escape(title) + """</title>
<style>
body { max-width: 1040px; margin: auto; padding: 28px; color: #24343f;
       background: white; font: 17px/1.55 system-ui, sans-serif; }
h1 { font-size: 2em; line-height: 1.2; } h2 { margin-top: 2em; font-size: 1.4em; }
p { max-width: 900px; } img { max-width: 100%; height: auto; }
.output_png { text-align: center; } .output_area { overflow-x: auto; }
table { border-collapse: collapse; margin: 1em 0; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 18px; border-bottom: 1px solid #d9e1e5; text-align: right; }
th { background: #edf3f6; } .anchor-link { display: none; }
@media (max-width: 600px) { body { padding: 16px; font-size: 16px; } }
</style></head><body>
""" + body + "\n</body></html>"
    (output / "analysis.html").write_text(html, encoding="utf-8")
    write_report_provenance(stage, args.function, suffix, inputs_before)
    print(f"HTML preview: {output / 'analysis.html'}")


if __name__ == "__main__":
    main()

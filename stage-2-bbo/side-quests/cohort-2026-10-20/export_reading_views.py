"""Export executed notebooks, including interactive Plotly figures, without fits."""

import copy
from html import escape
import json
from pathlib import Path
import re

from nbconvert import HTMLExporter
import nbformat
import plotly.io as pio
from plotly.offline import get_plotlyjs


STUDY = Path(__file__).resolve().parent


def export(only=None):
    (STUDY / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")
    audit = json.loads((STUDY / "dataset-audit.json").read_text(encoding="utf-8"))
    links, summaries = [], []
    for row in audit["summary"]:
        number = row["function"]
        folder = f"notebooks/Function_{number}/Week_04"
        names = ("exploration" if number <= 3 else "observation_review", "model_predictions")
        for name in names:
            if only is not None and (number, name) not in only:
                continue
            path = STUDY / folder / f"{name}.ipynb"
            notebook = nbformat.read(path, as_version=4)
            for cell in notebook.cells[1:]:
                if cell.cell_type == "markdown":
                    cell.source = re.sub(
                        rf"\]\([^)]*/Function_{number}/Week_04/(model_predictions|exploration|observation_review)\.ipynb\)",
                        lambda match: "](" + match[1] + ".ipynb)", cell.source)
                    cell.source = cell.source.replace(
                        "Week 4 and the remaining\npure exploration queries follow the fixed exploration plan above.",
                        "The original submitted Week 4 input and remaining exploration plan are retained above as historical references.")
                for output in cell.get("outputs", []):
                    plot = output.get("data", {}).get("application/vnd.plotly.v1+json")
                    if plot and any(trace.get("name") == "Measured target heights" for trace in plot.get("data", [])):
                        # Rendering only: move the marker key clear of the colour bar.
                        plot["layout"]["legend"] = dict(orientation="h", x=0, y=1.02,
                                                          xanchor="left", yanchor="bottom")
                        filenames = re.findall(r'[\"\']([^\"\']+\.html)[\"\']', cell.source)
                        if len(filenames) == 1:
                            pio.write_html(plot, path.parent / "figures" / filenames[0],
                                           include_plotlyjs="../../../../plotly.min.js", auto_play=False)
            nbformat.write(notebook, path)
            reading = copy.deepcopy(notebook)
            for cell in reading.cells:
                if cell.cell_type == "markdown":
                    cell.source = cell.source.replace("](model_predictions.ipynb)", "](model_predictions.html)")
                    cell.source = cell.source.replace("](exploration.ipynb)", "](exploration.html)")
                    cell.source = cell.source.replace("](observation_review.ipynb)", "](observation_review.html)")
                for output in cell.get("outputs", []):
                    plot = output.get("data", {}).get("application/vnd.plotly.v1+json")
                    if plot is not None:
                        output["data"]["text/html"] = pio.to_html(
                            plot, full_html=False, include_plotlyjs="../../../plotly.min.js",
                            auto_play=False, config={"responsive": True})
                        del output["data"]["application/vnd.plotly.v1+json"]
            exporter = HTMLExporter(template_name="basic", exclude_input=True,
                                    exclude_input_prompt=True, exclude_output_prompt=True)
            body, _ = exporter.from_notebook_node(reading)
            title = f"Pooled cohort study — Function {number}: {name.replace('_', ' ')}"
            page = "<!doctype html><html lang='en'><meta charset='utf-8'><title>" + escape(title) + "</title>"
            page += "<style>body{max-width:1150px;margin:30px auto;padding:0 24px;font:17px/1.5 system-ui;color:#172d40}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:6px 10px;border-bottom:1px solid #cbd5e1}a{color:#075985}</style>"
            page += "<p><a href='../../../index.html'>All eight functions</a></p>" + body + "</html>"
            path.with_suffix(".html").write_text(page, encoding="utf-8")
        observed = f"{folder}/{names[0]}.html"
        predicted = f"{folder}/model_predictions.html"
        links.append(f"<tr><th>Function {number}</th><td>{row['initial']}</td><td>{row['own_returned']}</td><td>{row['peer_added']}</td><td>{row['total']}</td><td><a href='{observed}'>Observed data</a> · <a href='{predicted}'>Models and predictions</a></td></tr>")
        gain = row["pooled_best"] - row["own_best"]
        summaries.append(f"<tr><th>{number}</th><td>{row['own_best']:.8g}</td><td>{row['pooled_best']:.8g}</td><td>{gain:.8g}</td><td>{escape(row['pooled_best_id'])}</td></tr>")
    page = """<!doctype html><html lang='en'><meta charset='utf-8'>
<title>Pooled cohort side quest</title><style>
body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#172d40;background:#f8fafc}
table{border-collapse:collapse;width:100%;background:white;margin:20px 0 35px}td,th{text-align:left;padding:12px;border-bottom:1px solid #cbd5e1}
a{color:#075985}h1{font-size:32px}.note{background:#e0f2fe;padding:18px;border-left:4px solid #0369a1}small{color:#475569}
</style><h1>A larger map of the same eight functions</h1>
<p>20 October 2026 · A separate copy of all sixteen Week 4 notebooks.</p>
<p class='note'><b>242 usable observations; 43 additional peer records.</b> Four peer repositories contain returned evaluations.
The common 175 starting observations are included once. The official Week 4 submissions remain unchanged.</p>
<p>Start with the observed-data views, then open the models. All fitting and figures were rerun on the pooled snapshot.
The reading views hide code; the corresponding notebooks sit beside them.</p>
<p>Each observed-data view begins with the course description and cohort clues.
Its prediction companion adds model interpretations, qualifications and a link back to that evidence.</p>
<table><tr><th>Function</th><th>Initial</th><th>Own returns</th><th>Peer additions</th><th>Total</th><th>Open the results</th></tr>"""
    page += "".join(links) + "</table><h2>What does the published pool add?</h2>"
    page += "<table><tr><th>Function</th><th>Own best</th><th>Pooled best</th><th>Difference</th><th>Best record</th></tr>" + "".join(summaries) + "</table>"
    page += """<p>Functions 1, 4 and 8 have a higher published best. These are peer-reported observations, not independently verified portal results.
Seven duplicate histories were matched. Twenty-four pending records and six inputs reported as exactly 1 were excluded; the latter were not clipped to fit our [0, 1) domain.
Peer logs have different precision and incomplete coverage. Equal values at different inputs are retained as reported.</p>
<p>The pooled progress charts show the best published result per reported round. They measure the available shared record,
not the performance of one strategy. Latest-result validation uses this collected snapshot and is not a historical forecast.</p>
<p><a href='README.md'>Methods and limitations</a> · <a href='dataset-audit.json'>Local source audit</a></p></html>"""
    (STUDY / "index.html").write_text(page, encoding="utf-8")
    print("Exported reading views and index.html; Plotly figures use one local JavaScript asset.")


if __name__ == "__main__":
    export()

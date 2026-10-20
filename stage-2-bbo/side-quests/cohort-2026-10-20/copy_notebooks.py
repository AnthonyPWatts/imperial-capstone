"""Copy the Week 4 methods and adapt evidence labels to the pooled snapshot."""

import json
import os
from pathlib import Path
import re
import shutil

import nbformat
import pandas as pd

from magnitude_prediction_cells import add_magnitude_predictions
from function_briefs import add_function_brief


STUDY = Path(__file__).resolve().parent
ORIGINAL = STUDY.parents[1] / "notebooks"
TARGET = STUDY / "notebooks"


def add_function_1_magnitude_view(notebook):
    """Keep the optional magnitude comparison when rebuilding the side quest."""
    notebook.cells = [cell for cell in notebook.cells if not cell.id.startswith("cohort-f1-magnitude-")]
    section = [
        nbformat.v4.new_markdown_cell("""## Absolute magnitude: do both signs point to the same active region?

Keep the signed views above, then compare **|y|**, the distance of an output from
zero. A large negative reading becomes high in this view. That makes it useful
for examining possible activity, while the optimisation objective still rewards
the highest **signed y**.

The map uses one logarithmic magnitude colour scale for positive circles and
negative squares. It shows only observed locations. Exact zeros, if present,
are grey crosses outside the log colour scale; tiny non-zero values stay non-zero.
"""),
        nbformat.v4.new_code_cell("""from cohort_magnitude import plot_magnitude_map, plot_magnitude_terrain

plot_magnitude_map(data, save_to=figures / "absolute-magnitude-map.png")
"""),
        nbformat.v4.new_code_cell("""# Compare magnitude rank with the existing signed-output rank; leave data unchanged.
magnitude_review = data.assign(
    abs_y=data['y'].abs(),
    magnitude_rank=data['y'].abs().rank(ascending=False, method='min').astype(int),
    signed_rank=data['y'].rank(ascending=False, method='min').astype(int),
)
display(magnitude_review.nlargest(7, 'abs_y')[
    ['observation_id', 'x1', 'x2', 'y', 'abs_y', 'magnitude_rank', 'signed_rank']
].set_index('observation_id').style.format(
    {'x1': '{:.6f}', 'x2': '{:.6f}', 'y': '{:.6e}', 'abs_y': '{:.6e}'},
))
strongest_positive = magnitude_review.loc[magnitude_review['y'].gt(0)].nlargest(1, 'abs_y')
strongest_negative = magnitude_review.loc[magnitude_review['y'].lt(0)].nlargest(1, 'abs_y')
if len(strongest_positive) and len(strongest_negative):
    positive, negative = strongest_positive.iloc[0], strongest_negative.iloc[0]
    separation = float(np.linalg.norm(
        positive[input_columns].to_numpy(dtype=float) - negative[input_columns].to_numpy(dtype=float)
    ))
    print(f"Strongest positive: ({positive['x1']:.6f}, {positive['x2']:.6f}); y = {positive['y']:.6e}")
    print(f"Strongest negative: ({negative['x1']:.6f}, {negative['x2']:.6f}); y = {negative['y']:.6e}")
    print(f"Their full input-space separation is {separation:.6f} on the unit square.")
    print(f"Their absolute magnitudes differ by {abs(np.log10(negative['abs_y']) - np.log10(positive['abs_y'])):.1f} orders of magnitude.")
"""),
        nbformat.v4.new_markdown_cell("""The absolute-magnitude terrain folds the signed terrain upwards using the
same display scale, camera and vertical relief. Marker colour and shape retain
the sign; hover over a marker to read its original signed output and magnitude.
The gold diamond now marks the **largest |y|**, which can have a poor signed rank.

The strongest positive and negative readings both lie in the broad central part
of this sample, but at different locations. This supports examining that region
from both signs. It does not establish a shared source, an ECG-like response or
a relationship between magnitude and distance to a source. The joined faces
illustrate gaps between observations; they do not measure a ridge through them.
"""),
        nbformat.v4.new_code_cell("""magnitude_terrain = plot_magnitude_terrain(
    data, scale=terrain_scale, save_to=figures / "absolute-magnitude-terrain.html",
)
magnitude_terrain.show(config={**PLOT_CONFIG, 'responsive': False})
"""),
    ]
    for index, cell in enumerate(section):
        cell.id = f"cohort-f1-magnitude-{index}"
    insertion = next(index for index, cell in enumerate(notebook.cells)
                     if cell.cell_type == "markdown" and cell.source.startswith("## Next query"))
    notebook.cells[insertion:insertion] = section


def repair_links(text, source, destination):
    def replace(match):
        link = match[1]
        if "://" in link or link.startswith("#") or link in (
            "model_predictions.ipynb", "exploration.ipynb", "observation_review.ipynb"
        ):
            return match[0]
        base, separator, fragment = link.partition("#")
        if not (destination.parent / base).exists() and (source.parent / base).exists():
            link = Path(os.path.relpath(source.parent / base, destination.parent)).as_posix()
            if separator:
                link += "#" + fragment
        return "](" + link + ")"
    return re.sub(r"\]\(([^)]+)\)", replace, text)


def build():
    audit = json.loads((STUDY / "dataset-audit.json").read_text(encoding="utf-8"))
    (TARGET / "Shared").mkdir(parents=True, exist_ok=True)
    for helper in (ORIGINAL / "Shared").glob("*.py"):
        shutil.copy2(helper, TARGET / "Shared" / helper.name)
    helper = TARGET / "Shared/observation_review_helpers.py"
    helper.write_text(helper.read_text(encoding="utf-8").replace(
        'label="Returned query output"', 'label="Best published return in round"'), encoding="utf-8")

    for summary in audit["summary"]:
        number, count = summary["function"], summary["total"]
        week = TARGET / f"Function_{number}/Week_04"
        data = pd.read_csv(week / "observations.csv", float_precision="round_trip")
        function_helpers = ORIGINAL / f"Function_{number}/FunctionSpecificShared"
        if function_helpers.is_dir():
            destination = week.parent / "FunctionSpecificShared"
            destination.mkdir(parents=True, exist_ok=True)
            for file in function_helpers.glob("*.py"):
                shutil.copy2(file, destination / file.name)
        for name in ("exploration.ipynb" if number <= 3 else "observation_review.ipynb", "model_predictions.ipynb"):
            source, target = ORIGINAL / f"Function_{number}/Week_04" / name, week / name
            notebook = nbformat.read(source, as_version=4)
            for index, cell in enumerate(notebook.cells):
                cell.id = f"cohort-f{number}-{Path(name).stem}-{index}"
                cell.metadata = {}
                if cell.cell_type == "code":
                    cell.execution_count, cell.outputs = None, []
                    cell.source = re.sub(r"\b(?:13|18|23|33|43) (real )?observations\b", f"{count} observations", cell.source)
                    cell.source = cell.source.replace("Next query remains the first saved coverage point:", "Original submitted Week 4 reference input:")
                    cell.source = cell.source.replace("Actual coverage-led next query:", "Original submitted Week 4 reference input:")
                    cell.source = cell.source.replace(f"Function {number} - Week 4:", f"Function {number} - cohort study:")
                else:
                    cell.source = re.sub(r"\b(?:13|18|23|33|43) (real )?observations\b", f"{count} observations", cell.source)
                    cell.source = cell.source.replace("actual Week 4 query plan", "original Week 4 reference plan")
                    cell.source = cell.source.replace("actual query decision", "original query decision")
                    if cell.source.startswith("## Decision:"):
                        cell.source = "## Side-quest proposal\n\nThe calculated input is an exploratory proposal from the pooled data. The official Week 4 input has already been submitted and is unchanged."

            kind = "exploration" if name == "exploration.ipynb" else "observation review" if name == "observation_review.ipynb" else "model predictions"
            reference = Path(os.path.relpath(source, week)).as_posix()
            notebook.cells[0].source = f"""# Function {number}: pooled cohort {kind}

**Side quest — 20 October 2026.** A copy of the [Week 4 notebook]({reference}),
using **{count} observations**: {summary['initial']} shared initial points,
{summary['own_returned']} confirmed own returns and **{summary['peer_added']} additional peer records**.
The methods, target treatments and search settings are retained. Peer data are
published reports, not independently verified portal returns. See the
[dataset audit](../../../dataset-audit.json) for exact sources and exclusions.

Week 4 has already been submitted. Any original coverage plan is a historical
reference; any newly calculated suggestion here is hypothetical. This pooled
snapshot includes records collected later, so its validation is not a forecast
made before the original Week 3 return. Query-budget numbers refer to the
original start of Week 4 and confer no additional evaluation allowance.
"""
            if name == "exploration.ipynb":
                for cell in notebook.cells:
                    if cell.cell_type == "code" and "data, planned_queries," in cell.source:
                        cell.source = """# Geometric comparison: peer points are a fixed background in every panel.
# This is not a reconstruction of when those points became available.
progression_data = data.copy()
progression_data.loc[progression_data['observation_id'].str.startswith('peer-'), 'source_round'] = 0
""" + cell.source.replace("data, planned_queries,", "progression_data, planned_queries,")
                notebook.cells.append(nbformat.v4.new_markdown_cell(
                    "The fixed-plan progression treats all published peer inputs as a background and adds the original planned points in order. Other exploration plots use the full pooled data. This geometric comparison does not reconstruct historical data availability."
                ))

            if name == "observation_review.ipynb":
                counts = data.source_round.value_counts().to_dict()
                notebook.cells[3].source = re.sub(
                    r"if data\[\"source_round\"\]\.value_counts\(\)\.to_dict\(\) != \{[^}]+\}:",
                    f'if data["source_round"].value_counts().to_dict() != {counts!r}:', notebook.cells[3].source)
                notebook.cells[3].source = notebook.cells[3].source.replace(
                    'print(f"Weekly queries used: {len(returned)}/{TOTAL_QUERY_WEEKS}")',
                    'print(f"Published completed rounds: {returned.source_round.nunique()}; pooled returned queries: {len(returned)}")')
                notebook.cells[2].source = """## 1. Establish the pooled evidence

The common supplied observations form one initial batch. Later rows retain
the round reported in each peer history; an ordered history without explicit
round numbers uses its recorded query order. Coverage across repositories is
incomplete. Several returned queries can belong to the same round."""
                latest = data.loc[data.source_round.eq(data.source_round.max())].iloc[0]
                rank = int(data.y.rank(ascending=False, method="min").loc[latest.name])
                best = data.loc[data.y.idxmax()]
                notebook.cells[4].source = f"""## 2. Place the latest recorded round among the pooled outputs

The own round 3 return is **{latest.y:.6g}**, ranked **{rank}/{count} (1 = highest)**
in the pooled data. The best published output is **{best.y:.6g}** from
`{best.observation_id}`. These rankings describe the available records, with
the precision and reporting limitations documented in the audit."""
                notebook.cells[7].source = f"""The full number line retains all {count} measured/reported outputs. The
close-up labels the highest five, retaining their actual spacing. Moving right
means a higher output. This selected sample does not estimate the distribution
of outputs across the domain. A missing peer return is not a zero."""
                notebook.cells[8].source = """## 3. Track the best published output across rounds

For each reported round, show its highest returned output and the best seen
through that round. Several peers can contribute; this is a pooled record,
not a measure of one person's strategy. Repository coverage is incomplete."""
                notebook.cells[9].source = """running_best = float(initial['y'].max())
progress_rows = [{'source_round': 0, 'stage': 'Initial batch', 'returned_y': np.nan,
                  'best_y': running_best, 'change_vs_prior_best': np.nan}]
for round_number, group in returned.groupby('source_round', sort=True):
    value = float(group['y'].max())
    change = value - running_best
    running_best = max(running_best, value)
    progress_rows.append({'source_round': int(round_number), 'stage': f'Round {round_number}',
                          'returned_y': value, 'best_y': running_best,
                          'change_vs_prior_best': change})
progress = pd.DataFrame(progress_rows)
display(progress.set_index('stage').style.format(precision=6, na_rep='—'))
"""
                notebook.cells[11].source = f"""The pooled incumbent changes from **{summary['initial_best']:.6g}** in the
shared initial data to **{summary['pooled_best']:.6g}**. The own-data incumbent
is **{summary['own_best']:.6g}**. Access to other people's experiments changes
the evidence budget, so this comparison cannot demonstrate that our strategy
outperformed theirs, or vice versa."""
                notebook.cells[15].source = """Read each row as one complete measured/reported combination. The input
colours describe settings, while the row label reports its output. Rows can
have very different performance even when neighbouring in rank. Compare all
coordinates before suggesting a pattern; these are not controlled tests of
one input at a time."""
                notebook.cells[18].source = """The full-input distance and output difference above compare the own round 3
query with the best published record from earlier reported rounds. Distances
do not identify causal effects or account for differing input relevance.
Published precision and potential reporting mistakes remain limitations."""
                notebook.cells[19].source = """## 5. Continue with the pooled model study

Open [model predictions](model_predictions.ipynb) to compare the same methods
on this larger dataset, inspect held-out errors and uncertainty, and calculate
a hypothetical query. The official submission is unchanged."""
                notebook.cells[20].source = """**Provenance:** the audit retains commit-pinned GitHub URLs, source locations,
duplicate matches and excluded records. Initial arrays from three peer
repositories matched the authorised starting data exactly and were included once."""

            for cell in notebook.cells:
                if cell.cell_type == "markdown":
                    cell.source = repair_links(cell.source, source, target)
            if number == 1 and name == "exploration.ipynb":
                add_function_1_magnitude_view(notebook)
            if number == 1 and name == "model_predictions.ipynb":
                add_magnitude_predictions(notebook)
            add_function_brief(notebook, number, prediction=name == "model_predictions.ipynb")
            nbformat.validate(notebook)
            nbformat.write(notebook, target)
        print(f"Copied Function {number}: {count} rows, two notebooks.")


if __name__ == "__main__":
    build()

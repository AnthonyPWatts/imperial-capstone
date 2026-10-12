# Reproducing the analyses

This guide covers the current weekly notebooks and the historical first-query
reading copies. For the reasoning behind them, see [methodology](methodology.md).

## Environment and local data

Follow the [repository environment setup](../../README.md#modelling-environment)
and provide the authorised course data described in [the data notes](../data/README.md).
Week 1 analyses use the preserved initial arrays. Week 2 and Week 3 notebooks load
`observations.csv` beside each notebook, with that function/week directory as
the kernel's working directory. These weekly CSV files are Git-ignored; populate
them with the matching historical round before running a fresh checkout:
Week 2 uses observations through round 1; Week 3 uses observations through round 2.
See the [recovery instructions](../data/README.md#recovering-earlier-snapshots).
Snapshots are not updated automatically when later results arrive.

## Historical first-query reading copies

From the repository root, these commands render Function 1's exploration and
risk reports:

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py --function 1
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py --function 1 --decision-risk
```

Use `--function N` to select another function. The pack needs all sixteen
exploration/risk reports; rendering only Function 1 is not sufficient for a
fresh checkout. The [notebook guide](../notebooks/README.md) lists the other
studies and the command for computing the remaining risk studies.

Once all sixteen reports have been rendered and their provenance is current,
build the combined reading pack:

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/build_first_query_review.py
```

The builder writes `.runtime/first-query-review/index.html` and the portable
`.runtime/capstone-first-query-pack.zip`. The HTML embeds figures and hides code.

Install the repository's pinned dependencies with
`.venv\Scripts\python.exe -m pip install -r requirements.txt` from the repository
root. Matplotlib and nbconvert are required for the reading copies and pack.

## Provenance and stale reports

Successful report runs write `provenance.json` beside each HTML file, recording
source, notebook and data hashes, runtime versions, and hashes of the generated
report and numerical results. The pack builder checks all sixteen manifests
before changing any pack or proposal files. Missing or stale manifests produce
an error naming the report command to rerun. Older reading copies remain useful
historical snapshots, but must be regenerated before inclusion in a new pack;
do not relabel their results with current source hashes.

## Repository structure

| Path | Contents |
| --- | --- |
| `data/` | Locally supplied input and output arrays; ignored by Git |
| `notebooks/` | Exploration, surrogate modelling and acquisition experiments |
| `notebooks/Shared/` | Weekly plotting, coverage-search, cache and query-budget helpers |
| `index.html`, `styles.css` | Static Stage 2 web workspace and its styling |
| `src/` | Reusable BBO code |
| `results/` | Observation and experiment logs safe to publish |
| `submissions/` | Proposed points and portal-submission notes |
| `docs/` | Methodology and reproduction guides, intake notes and unfinished final-deliverable templates |

## Constraints to enforce

- Work on all eight functions.
- Keep every coordinate in the interval `[0, 1)`.
- Format portal submissions to six decimal places as hyphen-separated
  coordinates with no spaces.
- Check dimensions, duplicates and bounds before proposing a point.
- Preserve an auditable link between every submitted point, returned value,
  model configuration and source commit.

Follow the portal and Module 12 instructions if their submission rules differ
from this scaffold.

## Documentation figures

The introductory diagrams and progress chart are generated from a small
illustrative example and the published round 1/2 result records. No local course
arrays are needed. With the repository's plotting dependencies installed, run:

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/build_documentation_figures.py
```

This refreshes the PNGs under `stage-2-bbo/docs/figures/` and
`stage-2-bbo/results/figures/`. The exploration diagram is schematic; the progress
chart uses real best-observed values. When a later round is recorded, extend the
chart inputs and update the surrounding status text together.

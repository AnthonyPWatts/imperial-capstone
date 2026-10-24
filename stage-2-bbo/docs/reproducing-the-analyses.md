# Reproducing the analyses

This guide covers the current weekly notebooks and the historical first-query
reading copies. For the reasoning behind them, see [methodology](methodology.md).

## Environment and local data

Run `.\setup-venv.ps1` from the repository root to create or refresh the local
Python environment. The [repository environment setup](../../README.md#modelling-environment)
also covers selecting an interpreter, rebuilding and activation. Provide the
authorised course data described in [the data notes](../data/README.md).
Week 1 analyses use the preserved initial arrays. Week 2–5 notebooks load
`observations.csv` beside each notebook, with that notebook's directory as
the kernel's working directory. These weekly CSV files are Git-ignored; populate
them with the matching historical round before running a fresh checkout:
Week 2 uses observations through round 1; Week 3 uses observations through round 2;
Week 4 uses observations through round 3 (199 in total); Week 5 uses observations
through round 4 (207 in total).
See the [recovery instructions](../data/README.md#recovering-earlier-snapshots).
Snapshots are not updated automatically when later results arrive.

Select this repository's `.venv` kernel and use the individual `Week_05`
directory as the working directory for the current observation reviews,
except Functions 1–4, which use `Week_05/official/` for the official analysis
and `Week_05/cohort/` for the pooled comparison.
Function 1 uses `exploration.ipynb`; Functions 2 and 3 use
`exploration-review.ipynb`; Functions 4–8 use `observation_review.ipynb`.
All eight have saved executions and regenerated figures. Their sources follow
the current Week 4 pattern, with updated data checks, result commentary and
historical forecast comparisons. For Function 2, continue in
`Function_2/Week_05/official/model_predictions.ipynb`. It has a saved run using the same
14-observation snapshot and compares the higher-dimensional model pattern with
the established x1-only and separate-length-scale GP hypotheses. Its proposed
input remains unsubmitted. Function 3 continues in
`Function_3/Week_05/official/model_predictions.ipynb`, using its unchanged 19-observation
snapshot. It compares seven declared regressors, then fitted GP candidates with
shared or separate input length scales, and includes conditional three-input
curves, hull-masked response planes, interactive terrains and acquisition views.
Its proposed input also remains unsubmitted. Function 4 continues in
`Function_4/Week_05/official/model_predictions.ipynb`, using 34 official rows
and its five established model families. It also has a separate 40-row cohort
comparison. Function 1 has a signed-score and magnitude model study described
below, retaining its saved coverage plan. Functions 5–8 have no new Week 5 model studies; their
Week 4 model links are explicitly historical.

For the previous model studies, use the individual `Week_04`
directory as the working directory. Functions 1–3 start with `exploration.ipynb`;
Functions 4–8 start with `observation_review.ipynb`. Continue in
`model_predictions.ipynb` and review the validation and sensitivity results
before recording a query decision. The observation reviews for Functions 4–8
have saved runs and figures, as do all eight model notebooks. The Week 4
strategies were retained and submission was confirmed on 20 October 2026. The model notebooks
may perform coverage searches and fits; earlier weeks' caches and outputs
were not copied into the new week.

## Function 1: Week 5 official and cohort runs

Run `exploration.ipynb` then `model_predictions.ipynb` from
`notebooks/Function_1/Week_05/official/`, followed by the matching notebooks
in `cohort/`. Each folder uses its own local CSV and saves its own figures.
Restore the 14-row official and 20-row pooled snapshots, plus the detailed
local audit, using the [data notes](../data/README.md#function-1-week-5-cohort-comparison).
No new coverage search is required: one fixed exploration point remains.

The settings declare the five existing model families, fixed target scale,
GP length scales, MAE selection rule, seeded candidates and uncertainty weights.
Signed-log score and log absolute magnitude are refitted independently.
Predictions and errors remain in the declared transformed units. Magnitude
does not predict sign or replace the competition objective.

The cohort model reads the sibling official CSV and saved target-review
summary, checks its settings, and reproduces the official fit before
comparing identical held-out official targets and common candidate inputs.
Both versions retain the saved portal input `0.912140-0.861783`; model-led
choices are hypothetical. The week-level `report.md` contains interpretation
and the course lifecycle. The brief does not state that Function 1 is noisy,
so no noise analysis is included.

## Functions 2 and 3: Week 5 cohort copies

For Function 2, run `exploration-review.ipynb` then `model_predictions.ipynb`
from `notebooks/Function_2/Week_05/cohort/`, using its local `observations.csv`
and `data-audit.json`. The comparison reads the official CSV, saved proposal
and coverage cache from the sibling `official/` folder. Each version writes
to its own `figures/` and `.coverage-cache/`. Start with the week-level
`report.md` to read the results without running code.

For Function 3, run `exploration-review.ipynb` then `model_predictions.ipynb`
from `notebooks/Function_3/Week_05/cohort/`, using its local `observations.csv`
and `data-audit.json`. Its sibling `official/` folder has the same notebook
names and its own observations, figures and cache. The week-level `report.md`
contains the discussion and cohort comparison; notebook settings are at the
top, with calculations and plotting in shared helpers. The Function 3 brief
does not state that it is noisy, so no noise analysis is included.

Use the capstone-local kernel and restore the data as described in the
[Function 2 data notes](../data/README.md#function-2-week-5-cohort-comparison) or
[Function 3 data notes](../data/README.md#function-3-week-5-cohort-comparison). These notebooks
also read the sibling official `observations.csv`; each model comparison reads
the saved official notebook's proposal summary. Both versions write figures
under their own `figures/` and use their own `.coverage-cache/`. A cohort run
does not overwrite official notebooks, CSVs or figures. The paired comparison refits the original model families
on the matching 14 or 19 official rows to reproduce the recorded baseline and
proposal model. Function 3 retains all three coordinates in conditional views
and in the proposal comparison.

## Function 4: Week 5 official and cohort runs

Run `observation_review.ipynb` then `model_predictions.ipynb` from
`notebooks/Function_4/Week_05/official/`, followed by the matching notebooks
in `cohort/`. Each folder has its own local `observations.csv`, `figures/`
and `.coverage-cache/`. Restore the snapshots and `cohort/data-audit.json`
as described in the [data notes](../data/README.md#function-4-week-5-cohort-comparison).
The cohort model reads the saved official proposal and refits the same five
families on the matching official targets.

Use `Week_05/report.md` for discussion and the cohort comparison. Notebook
settings are at the top; shared helpers prepare official progress, verify
the saved Week 4 forecast, fit models and retain every coordinate in conditional
curves and proposal profiles. The original Week 4 model must have its saved
printed forecast for the actually submitted input. No historical forecast is
recreated from later data. Function 4's course description does not state that
it is noisy, so these runs include no noise analysis.

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

The setup helper installs the repository's pinned dependencies, including
Matplotlib and nbconvert, which are required for the reading copies and pack.

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

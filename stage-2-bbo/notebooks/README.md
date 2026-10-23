# Notebooks

## Shared implementation

Use [`Shared/`](Shared/README.md) for helpers reused by several functions and
`Function_N/FunctionSpecificShared/` for helpers tied to one function across
weeks. The shared-helper guide lists the modelling and plotting modules.
Weekly notebooks retain their data, assumptions, teaching steps and decisions.

## Module 14 reporting cutoff

Use each function's `Week_03/` notebooks to review the decisions made for
Module 14. Their independent CSV snapshots contain **191 observations through
round 2**: 12, 12, 17, 32, 22, 22, 32 and 42 rows for Functions 1–8.
All eight Week 3 inputs were submitted; confirmation was recorded on
12 October 2026. Functions 1–3 used the next points in their saved coverage
shortlists, and Functions 4–8 used updated GP comparisons and UCB selection.
The [Stage 2 overview](../README.md) records that module-end position.

The third-query return and Week 4 runs below are later workspace history.
They must not be used to explain the earlier Week 3 decisions or reflection.

## Current work: Week 5 observation reviews

All eight `Function_N/Week_05/` folders contain independent local CSV snapshots
through round 4 and saved executions of the current Week 4 observation pattern:

| Function | Week 5 notebook | Observations |
| --- | --- | ---: |
| 1 | [Exploration](Function_1/Week_05/exploration.ipynb) | 14 |
| 2 | [Results report, including cohort comparison](Function_2/Week_05/report.md); [official exploration review](Function_2/Week_05/official/exploration-review.ipynb), then [official model predictions](Function_2/Week_05/official/model_predictions.ipynb) | 14 |
| 3 | [Results report, including cohort comparison](Function_3/Week_05/report.md); [official exploration review](Function_3/Week_05/official/exploration-review.ipynb), then [official model predictions](Function_3/Week_05/official/model_predictions.ipynb) | 19 |
| 4 | [Results report, including cohort comparison](Function_4/Week_05/report.md); [official observation review](Function_4/Week_05/official/observation_review.ipynb), then [official model predictions](Function_4/Week_05/official/model_predictions.ipynb) | 34 |
| 5 | [Observation review](Function_5/Week_05/observation_review.ipynb) | 24 |
| 6 | [Observation review](Function_6/Week_05/observation_review.ipynb) | 24 |
| 7 | [Observation review](Function_7/Week_05/observation_review.ipynb) | 34 |
| 8 | [Observation review](Function_8/Week_05/observation_review.ipynb) | 44 |

- Function 1: `exploration.ipynb`, retaining output number lines, sampling-gap
  views, interactive terrain/cloud figures and the fixed exploration progression.
- Function 2: `exploration-review.ipynb` closes the saved coverage plan. Its
  `model_predictions.ipynb` continues with held-out comparisons, fitted surfaces,
  uncertainty, candidate selection and sensitivity checks using all 14 observations.
  These notebooks and their figures are in `Week_05/official/`; the week-level
  `report.md` presents their results alongside the separate `cohort/` comparison.
- Function 3: `exploration-review.ipynb` closes its fixed plan, followed by an
  executed `model_predictions.ipynb` using all 19 observations. Conditional curves,
  planes and interactive terrains retain all three inputs and full 3D hull checks.
  Official and pooled notebooks live in `Week_05/official/` and `Week_05/cohort/`;
  their discussion and comparison are in the week-level `report.md`.
- Function 4: `observation_review.ipynb` and a new `model_predictions.ipynb`
  live in `Week_05/official/`, with matching notebooks under `cohort/` and a
  week-level `report.md`. The pooled snapshot adds six retained peer returns.
  Calculations use shared helpers, with settings at the top and all four
  inputs retained. The problem description does not state that it is noisy,
  so no noise analysis is included.
- Functions 5–8: `observation_review.ipynb`, retaining measured-output ranks,
  round progress, strongest full input profiles, distance comparisons and a
  check against the saved Week 4 forecast for the actual submitted input.

The snapshots contain **207 observations**: 14, 14, 19, 34, 24, 24, 34 and 44
rows respectively. They include the Week 4 return received on **23 October 2026**;
all eight inputs match the saved submissions. Only Function 7 improves its
incumbent, to `2.6189682154023832`. Source data and earlier weekly files are
unchanged. The raw feedback and CSV snapshots remain Git-ignored; see the
[data notes](../data/README.md#week-5-notebook-snapshots).

Use the capstone-local `.venv` with each notebook's folder as the working
directory. For Functions 2–4, use `Week_05/official/` or `Week_05/cohort/`;
the other functions still use their `Week_05/` folder. All eight observation
notebooks were executed from that interpreter and their
figures regenerated. Functions 2–4 also have saved Week 5 model runs and
unsubmitted proposals; the other functions' Week 5 model studies remain separate
work. No Week 5 submission is recorded. Nine query opportunities
remain per function, including Week 5. Function 1 has one saved exploration point
left; Functions 2 and 3 have completed their saved shortlists.

### Function 2: retained cohort comparison

In `Function_2/Week_05/cohort/`, [cohort exploration review](Function_2/Week_05/cohort/exploration-review.ipynb)
and [cohort model predictions](Function_2/Week_05/cohort/model_predictions.ipynb) are
separate, executed copies of the official notebooks. They use 19 observations:
the 14 official rows plus five accepted peer returns from the pinned 20 October
cohort snapshot. No peer repository refresh is implied. The official notebooks,
data and provisional proposal remain unchanged.

The model copy compares the same seven model families, scores the same 14 official
held-out targets with and without peer training rows, and compares both fitted
policies on one common candidate pool. Its final discussion distinguishes the
new exploratory alternative from the official proposal. Figures live under
`cohort/figures/`, alongside the official version's `official/figures/`, with
each version's own local coverage cache. The pooled CSV and detailed
source audit remain local and Git-ignored; see the
[data notes](../data/README.md#function-2-week-5-cohort-comparison).

### Function 3: retained cohort comparison

In `Function_3/Week_05/cohort/`, [cohort exploration review](Function_3/Week_05/cohort/exploration-review.ipynb)
and [cohort model predictions](Function_3/Week_05/cohort/model_predictions.ipynb) are
executed copies using **24 observations**: the 19 official rows and five accepted
peer returns from the same pinned 20 October snapshot. Official data, model
results and proposal are preserved. Each version has its own `figures/` and
local coverage cache alongside its notebooks.

The comparison retains all three inputs, the same seven model families, and
conditional 3D plots. It scores the same 19 official targets with and without
peer training rows, and compares both policies on one common candidate pool.
The shared-length-scale GP becomes preferred, while the separate-length-scale
GP's paired error increases. The week-level [report](Function_3/Week_05/report.md)
records the low-x3 exploratory alternative and its limitations. Settings remain
at the top of each notebook, with calculations in shared helpers. Function 3's
course description does not state that it is noisy, so no noise analysis is
included. See the
[data notes](../data/README.md#function-3-week-5-cohort-comparison) for the local
pooled CSV and pinned source audit.

### Function 4: retained cohort comparison

In `Function_4/Week_05/cohort/`, [observation review](Function_4/Week_05/cohort/observation_review.ipynb)
and [model predictions](Function_4/Week_05/cohort/model_predictions.ipynb) use
**40 observations**: 34 official rows plus six accepted peer returns from the
pinned 20 October snapshot. The peer-reported best is 0.712146, compared with
the official best of -0.129918. Official progress and the saved Week 4 forecast
remain distinct from that pooled evidence.

The same five established model families are compared on identical official
targets. Peer training rows improve kernel ridge's RMSE but increase the GP's;
the GP remains the strongest of those methods. The [report](Function_4/Week_05/report.md)
contains the results, four-coordinate proposal profiles and the common-candidate
comparison, including its search sensitivity. See the
[data notes](../data/README.md#function-4-week-5-cohort-comparison) for the local
snapshot and checksum verification.

The course lifecycle for the original observation-only rollover is listed below.
The model notebooks for Functions 2–4 follow the same ten steps through
candidate comparison and proposed next inputs; neither performs a portal submission.

1. **Define the goal and scope** — review the fourth return and prepare the next week's evidence.
2. **Gather the data** — retain the results email and original cumulative attachments.
3. **Explore the data** — regenerate output, coverage, progress and input-profile views.
4. **Clean and preprocess the data** — validate dimensions, bounds, precision and cumulative history; add each return once.
5. **Select and engineer features** — retain the existing ordered input coordinates; no new features are required.
6. **Define the machine-learning task** — not applicable to new modelling here; the existing regression/optimisation task is unchanged.
7. **Partition the data** — not applicable to descriptive observation reviews; no model is evaluated on these snapshots.
8. **Select and train candidate methods** — not applicable; Week 5 fitting is separate work.
9. **Evaluate and interpret the results** — compare measured returns, incumbents, ranks and saved pre-return forecasts.
10. **Deploy and iterate** — no deployment or portal submission applies; carry this evidence into the next query review.

## Previous workspace: Week 4 observation reviews and model studies

Each `Function_N/Week_04/` notebook expects a local `observations.csv`
with all initial observations and the three confirmed weekly returns. Expected
row counts for Functions 1–8 are 13, 13, 18, 33, 23, 23, 33 and 43
(199 total). The schema remains `observation_id,source_round,x1,...,y`.

These are independent snapshots of the accumulated data as of 14 October 2026.
They are Git-ignored and are not supplied by a fresh checkout. Follow the
[recovery instructions](../data/README.md#recovering-earlier-snapshots) before
running the notebooks on a new machine.
See the [data notes](../data/README.md#week-4-notebook-snapshots) for provenance
and verification. Earlier weekly snapshots and notebooks remain unchanged.

Functions 1–3 have `exploration.ipynb` and `model_predictions.ipynb`, with saved
executions and figures. Functions 4–8 use `observation_review.ipynb` followed by
`model_predictions.ipynb`. Their observation reviews retain the output number
line, distinguish initial observations from returned queries, track the best
observed output across rounds, and compare the strongest full input profiles.
All five observation reviews and all eight model notebooks have saved executions
and figures. Week 4 submission was confirmed on 20 October 2026.

Functions 1–3 keep measured-output and coverage reviews in exploration, with
fitted surfaces, validation, uncertainty and hypothetical model-led queries in
model predictions. The two-input functions share contour and terrain plots;
Function 3 uses coordinate curves and conditional planes through its three-input
space. Their actual next queries still follow the saved coverage sequence.

Functions 1–3 skip the first two evaluated points of the saved Week 2
shortlists. Function 1 has two planned points left; Functions 2 and 3 each
have one. Functions 4–8 retain their modelling and candidate-search settings.
The project owner reviewed the Week 4 runs and retained the existing strategies.
Function 7's new best is recorded in its observation review. The eight selected
inputs are retained in the [Week 4 submission file](../submissions/Week_04/submissions.txt).

Use the capstone-local `.venv` kernel and each notebook's own week directory
as the working directory. For Functions 1–3, run exploration first; for
Functions 4–8, run observation review first. Continue in model predictions.
The CSVs retain full float64 precision and the new notebooks
load them with `float_precision="round_trip"`. Ten weekly query opportunities
were available per function at the start of Week 4. All eight Week 4 inputs
were submitted; their outputs are now recorded in the round 4 results and
reviewed in the Week 5 notebooks above.

## Weekly output review convention

Use `Shared/output_helpers.py` for the printed review and reuse the same report
in the observed-output chart caption. State the returned value and its actual
rank together, for example `rank 5/17 (1 = highest output)`, then compare it with
the earlier output range. Keep the rank definition beside the actual rank;
do not present `rank 1` as a standalone summary of the latest result.

When discussing percentages, distinguish earlier observations strictly below
the new value from the chart's share of all observations at or below it,
including the new result. Keep the next-query commentary brief and grounded in
the result; reuse the saved shortlist unless the evidence justifies changing it.

## Previous work: Week 3

`Function_N/Week_03/` retains the analyses that selected the eight
[Week 3 inputs](../submissions/Week_03/submissions.txt). Their fixed snapshots
contain 191 observations through round 2. Submission was confirmed on
12 October 2026; the return arrived on 14 October. All eight inputs matched,
and Function 7 improved its incumbent. The saved pre-submission wording and
outputs remain historical. See the [submission log](../submissions/README.md).

## Previous work: Week 2

`Function_N/Week_02/exploration.ipynb` exists for all eight functions. Each
reviews the round 1 return and compares geometric coverage batches using shared
helpers in `Shared/`. Functions 1–3 selected the first points of four-, three-
and three-point batches respectively for Week 2.

Functions 4–8 also contain `Week_02/model_predictions.ipynb`. These compare
regression models with leave-one-out validation and select one next input using
a Gaussian-process upper confidence bound. The saved outputs document model
errors, candidate comparisons and limitations. The
[methodology guide](../docs/methodology.md#week-2-approach-and-submitted-inputs) summarises the
Week 2 methods; the [submission log](../submissions/README.md) now records
its confirmed results.

Run each Week 2 notebook with its own directory as the working directory and
its local `observations.csv` beside it. These CSV files are ignored by Git;
populate them with observations through round 1 only, as described in
[the data notes](../data/README.md#recovering-earlier-snapshots). The expected
row counts are 11, 11, 16, 31, 21, 21, 31 and 41 (183 total).
Coverage caches are local and keyed to their data, search code and settings.
The superseded practice notebooks are retained in Git history; the
function/week folders and shared helpers are the maintained workflow.

The sections below describe the historical Week 1 studies.

## Function 1 exploration

[`01-data-validation-and-exploration.ipynb`](Function_1/Week_01/01-data-validation-and-exploration.ipynb)
is a finished descriptive first look at Function 1: all ten observations, ranked
locations, sampling gaps and output magnitudes, with a second scale comparison
excluding the largest-magnitude output from the charts only. It then uses
sequential maximin sampling to propose the first new point, with a comparison
against the central gap and a map of the selected location. A boundary trade-off
then compares average coverage from the corner, an inset point and the centre.
Finally it optimises that average directly and proposes `(0.468429, 0.435440)`.
The taper strategy explains why this first query prioritises coverage and how
later observations can justify concentrating on a promising region. A closing
comparison identifies what changes when coordinate-pair plots become projections.
No portal submission is made. It contains brief plot calls;
loading, basic checks and plotting live in
[`initial_exploration.py`](../src/initial_exploration.py).

Run from the capstone repository root using its existing environment:

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py
```

The runner saves an executed notebook, six PNG figures and a self-contained
`analysis.html` under the ignored `.runtime/function-1-initial-analysis/`
directory. The HTML hides code and embeds the charts, so it can be opened on
another computer without Python or the repository. The source notebook has no
saved outputs; the full supplied data remain local.

This notebook's prose describes Function 1 specifically. The other functions
have their own descriptive notebooks and reuse the plotting helpers. With
eight inputs there are 28 distinct pairs. Pairwise gap maps describe projected
coverage, not coverage of the full eight-dimensional space.

[`coverage_optimisation.py`](../src/coverage_optimisation.py) contains the
two-dimensional area integral and search. Its focused checks use standard
`unittest`: `.\run-python.ps1 -m unittest discover -s stage-2-bbo/tests` from the repository
root, using the same environment.

## Planning several coverage points

[`01b-function-1-batch-coverage.ipynb`](Function_1/Week_01/01b-function-1-batch-coverage.ipynb) explores
jointly selecting N = 2 and N = 3 points, with N = 1 as the reference. It compares
joint and sequential selection, plots remaining gaps on a common colour scale,
and reports measured computation times. Run:

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py --batch-coverage
```

The self-contained HTML, executed notebook, figure and `study.json` are saved to
`.runtime/function-1-batch-coverage/`. Computation and presentation helpers live in
[`coverage_optimisation.py`](../src/coverage_optimisation.py) and
[`batch_coverage_analysis.py`](../src/batch_coverage_analysis.py).

## Function 2 exploration

[`01c-function-2-initial-exploration.ipynb`](Function_2/Week_01/01c-function-2-initial-exploration.ipynb)
shows all ten supplied observations, ranked locations, sampling gaps and ordered
outputs on their original and logarithmic-magnitude scales. Its observations
distinguish the course's noisy log-likelihood description from what the small
sample establishes. Its taper discussion connects the upper-left coverage gap
with the hypothesis of a promising region near `x1 = 0.7`, and frames the move
from learning towards fine tuning. It also establishes what to compare as
dimensionality increases. The selection section fits an x1-only explanation and
a local two-input explanation, maps coverage benefit and standardised predictive
disagreement, and compares mixtures of the two scores. Noise, predictive checks
and a less restrictive two-input fit expose sensitivity to the modelling choices.
All observations are retained; no portal query is submitted.

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py --function 2
```

The executed notebook, five PNG figures, `selection-study.json` and self-contained HTML are saved under
`.runtime/function-2-initial-analysis/`. Loading and plotting reuse
[`initial_exploration.py`](../src/initial_exploration.py); the notebook contains
short calls and descriptive prose. Fitting, scoring, search and additional plots
live in [`function_2_selection.py`](../src/function_2_selection.py).

## Function 2 decision risk

[`01d-function-2-decision-risk.ipynb`](Function_2/Week_01/01d-function-2-decision-risk.ipynb) replaces
the coverage/hypothesis weighting question with expected shortfall of the final
recommendation. It calculates finite-set knowledge-gradient values under three
models and compares eight first queries followed by twelve simulated queries
under a declared taper policy. Model, noise, grid and simulation limitations are
explicit; the final recommendation may be an unqueried point in the finite pool.

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py --function 2 --decision-risk
```

The executed notebook, two figures, `risk-study.json` and standalone HTML are
saved to `.runtime/function-2-risk/`. The helper is
[`function_2_risk.py`](../src/function_2_risk.py). No portal evaluations are used.

## Function 3 exploration

[`01e-function-3-initial-exploration.ipynb`](Function_3/Week_01/01e-function-3-initial-exploration.ipynb)
shows all fifteen observations, all three ranked coordinate-pair projections,
projected sampling gaps and three slices measuring distance in the full cube.
Ordered raw/log-magnitude charts include a second view without the worst output.
Input/output plots and a small correlation sensitivity comparison show how that
observation affects the apparent x3 trend. The taper discussion identifies
coverage gaps and competing explanations without selecting a query or fitting a
model.

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py --function 3
```

The executed notebook, six PNG figures, `exploration-summary.json` and standalone
HTML are saved under `.runtime/function-3-initial-analysis/`. The source notebook
remains unexecuted. Helpers live in
[`multidimensional_exploration.py`](../src/multidimensional_exploration.py) and
[`initial_exploration.py`](../src/initial_exploration.py).

## Remaining explorations and knowledge-gradient studies

`01-function-N-initial-exploration.ipynb` covers Functions 4–8. Each displays all
supplied rows, every coordinate-pair projection in readable groups, projected
gaps on common scales, a full-space coverage summary and output-scale comparisons.
The prose identifies the function's observed patterns and their limitations.

`02-function-N-decision-risk.ipynb` covers Functions 1 and 3–8. These compare
additive effects with local and separate-length interactions, checking held-out
prediction, noise, candidate pools and a thirteen-query taper policy. A larger
pool recomputes the conservative decision criterion; the selected point also
appears in the bounded-pool rollout. Function 1 deliberately retains coverage
after an inconclusive raw-output KG comparison. Function 2 uses its existing
`01d` study.

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py --function 8
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py --function 8 --decision-risk
.venv\Scripts\python.exe stage-2-bbo/scripts/build_first_query_review.py
```

To compute all remaining risk studies before rendering, use:

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/run_risk_studies.py
```

Numerical results are reused only when the supplied data and numerical source
fingerprints match. Changed report prose or plotting code does not repeat the
expensive calculations. Reading copies and executed notebooks remain under
`.runtime/`; these Week 1 source notebooks retain no outputs. The final local
review page links all sixteen exploration/risk reports and includes the eight
portal strings.

## Originally planned later notebook sequence

These filenames are a plan, not existing notebooks. Current weekly analyses
use the `Function_N/Week_XX/` layout described above.

2. `02-gaussian-process-baseline.ipynb`
3. `03-acquisition-comparison.ipynb`
4. `04-iterative-optimisation.ipynb`
5. `05-final-analysis.ipynb`

The iterative notebook should be rerunnable from the observation log rather
than depending on hidden notebook state.

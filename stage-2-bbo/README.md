# Stage 2: black-box optimisation

This is the assessed capstone problem: maximise eight unknown functions using
only their observed inputs and outputs. The functions range from two to eight
dimensions. Each course round adds a portal observation to the local dataset.

The practical problem is deciding which experiment to run next when evaluations
are scarce. Applications include model hyperparameter tuning, process settings
and product experiments. As a software developer, I am using the project to
develop my judgement about applying ML: comparing methods, checking assumptions
and explaining decisions through reproducible code and evidence.

Browse the [Stage 2 workspace](https://anthonypwatts.github.io/imperial-capstone/stage-2-bbo/)
or return to the [Capstone Hub](https://anthonypwatts.github.io/imperial-capstone/).

## Current status: 8 October 2026

Round 1 is the latest confirmed evaluation. Week 2 inputs have been submitted
for all eight functions and their results are pending. No third round has been
submitted. Week 2 submission was confirmed on 8 October; this is the confirmation
date, not a recorded portal timestamp. The [submission log](submissions/README.md)
distinguishes submitted inputs from evaluated results.

### Round 1 results

The first results email arrived on **5 October 2026 at 00:38 BST**. All eight
returned inputs match the saved proposals. The recorded dataset now contains
**183 observations**, retained in the versioned [working snapshots](Week2_Notebooks/).
The round 1 import record documents preservation of the original 175 observations.

Functions **4, 5, 6 and 7** improved their best observed values. The other four
incumbents are unchanged. Values below are rounded for readability; the
[results record](submissions/round-01-results.json) retains full precision.

| Function | Returned output | Best observed after round 1 | Improved incumbent? |
| --- | ---: | ---: | --- |
| 1 | -0.00550935 | 7.710875e-16 | No |
| 2 | 0.283444 | 0.611205 | No |
| 3 | -0.0564312 | -0.0348353 | No |
| 4 | -2.738157 | -2.738157 | Yes |
| 5 | 3933.145724 | 3933.145724 | Yes |
| 6 | -0.378691 | -0.378691 | Yes |
| 7 | 1.405759 | 1.405759 | Yes |
| 8 | 9.498300 | 9.598482 | No |

Start with the [Function 1 result analysis](results/function-1-round-01-analysis.md).
Its returned output, `-0.00550934555381189`, is a new minimum; the best observed
value is unchanged. Average sampling distance fell by 7.53776%, while the three
existing GP families still failed to beat the mean baseline on held-out RMSE.
The [Function 2 result analysis](results/function-2-round-01-analysis.md) records
its fifth-ranked return, unchanged incumbent and stronger held-out support for
the x1-only and flexible models. Its next-query choices remain sensitive to
model and noise assumptions. All eight functions now have Week 2 exploration
notebooks reviewing the returned observation, output scale and sampling gaps.
Functions 4–8 also have Week 2 model-comparison and query-selection notebooks.

See the [round results record](submissions/round-01-results.json) and
[accumulated data layout](data/README.md#accumulated-observations-round-1).
The first-query material below describes historical pre-evaluation work;
its original proposal JSON and source arrays remain unchanged.

## Module 12 intake

The initial data were collected and verified on **24 September 2026**: 16 NumPy
arrays containing 175 observations across all eight functions. The original ZIP
and extracted arrays belong under `data/` and remain Git-ignored; provide these
separately when setting up a fresh checkout.

The course allows **one evaluated query per function per week**, over 13 rounds
in Modules 12–24. Pending inputs can be revised before the week ends or processing
occurs. See the [Module 12 intake notes](docs/module-12-intake.md) for sources,
deadlines, verified data shapes and the first reflection requirements.

## Inputs, outputs and objective

Each function accepts a numerical vector with coordinates in **`[0, 1)`** and
returns one scalar output. Every function is maximised, including those with
negative outputs: a less negative value is better. The functions have different
output scales, so raw values are compared within each function.

| Function | Input dimensions | Initial observations | Observations after round 1 |
| --- | ---: | ---: | ---: |
| 1 | 2 | 10 | 11 |
| 2 | 2 | 10 | 11 |
| 3 | 3 | 15 | 16 |
| 4 | 4 | 30 | 31 |
| 5 | 4 | 20 | 21 |
| 6 | 5 | 20 | 21 |
| 7 | 6 | 30 | 31 |
| 8 | 8 | 40 | 41 |

For a function with `n` observations and `d` coordinates, the input array has
shape `(n, d)` and the output array has shape `(n,)`. Portal inputs use exactly
six decimal places, hyphens between coordinates and no spaces. For example,
Function 2's evaluated input `0.774999-0.949999` returned
`0.2834440433397594`. A surrogate regression model learns from these input/output
pairs and predicts responses; a Gaussian process also estimates uncertainty.
The query-selection rule uses that information to choose the next input.

The budget is 13 additional evaluations per function, or 104 across all eight.
After round 1, 12 evaluations remain per function: the submitted Week 2 query
awaiting evaluation and 11 subsequent query opportunities.
Weekly feedback limits adaptation. Sparse observations, unknown function
structure and possible noise limit what can be inferred; the true global
maxima are unavailable for checking. Progress is measured by best observed
outputs, supported by model diagnostics and a record of why each query was chosen.

## Week 2 approach and submitted inputs

The [Week 2 portal inputs](submissions/Week_02/submissions.txt) contain one
submitted input per function, in function-number order. They draw on two routes:

- **Functions 1–3: geometric exploration.** The exploration notebooks optimise
  batches jointly to reduce the largest sampled distance to an observation.
  An elbow in the coverage-versus-query-count curve suggests batches of four,
  three and three points respectively. Week 2 uses the first point from each
  batch; the remaining points are a provisional exploration plan. This minimax
  criterion differs from Function 1's round 1 average-distance criterion.
- **Functions 4–8: model-guided selection.** Leave-one-out comparisons assess
  a mean baseline, linear regression, ridge regression, kernel ridge and a
  Gaussian process. Function 5 also tests a log-target GP. The raw-output GP has
  the lowest RMSE among the tested models for each of these five functions.
  Each model notebook proposes one query, to be reassessed after feedback.

| Function | Week 2 selection | Evidence |
| --- | --- | --- |
| 1 | First point of a four-point coverage batch | [Exploration](notebooks/Function_1/Week_02/exploration.ipynb) |
| 2 | First point of a three-point coverage batch | [Exploration](notebooks/Function_2/Week_02/exploration.ipynb) |
| 3 | First point of a three-point coverage batch | [Exploration](notebooks/Function_3/Week_02/exploration.ipynb) |
| 4 | GP upper confidence bound (UCB), near the promising region | [Model predictions](notebooks/Function_4/Week_02/model_predictions.ipynb) |
| 5 | GP UCB, all coordinates at `0.999999` | [Model predictions](notebooks/Function_5/Week_02/model_predictions.ipynb) |
| 6 | GP UCB, probing the low-x5 boundary | [Model predictions](notebooks/Function_6/Week_02/model_predictions.ipynb) |
| 7 | GP UCB, testing near the best observation | [Model predictions](notebooks/Function_7/Week_02/model_predictions.ipynb) |
| 8 | GP UCB, testing a different corner | [Model predictions](notebooks/Function_8/Week_02/model_predictions.ipynb) |

The Week 2 UCB score is **predicted mean + 1.96 × posterior standard deviation**.
It rewards both high predicted responses and uncertainty worth investigating.
The weight is a selection setting, not a calibrated 95% confidence guarantee.
Functions 6 and 7 illustrate the distinction: their selected points have
predicted means below the incumbent but enough uncertainty to warrant a query.
Candidate searches include broad sampling and corners, with additional boundary
or local searches where appropriate. Functions 5–8 also compare a single
pure-coverage alternative.

Saved Week 2 GP RMSE reductions against the mean baseline are **81.4%, 21.1%,
38.9%, 41.4% and 69.4%** for Functions 4–8 respectively. These are validation
diagnostics on the observed samples, not measured optimisation gains. Function
5's GP substantially underpredicted the latest result when it was held out;
Function 7's neighbouring high-value observations help its validation score.
Neither result establishes reliable prediction throughout the input space.

Coverage is assessed on a finite sample and ignores response values. Its elbow
is a geometric heuristic, not an optimal allocation of the query budget.
Likewise, GP uncertainty and acquisition scores depend on the fitted kernel,
assumed noise and searched candidates. The notebooks examine these assumptions;
no proposal guarantees improvement or identifies a proven global maximum.

## Working strategy: the taper

**Taper-waper: start by reducing ignorance, finish by exploiting what we have learned.**

Across the 13 rounds, early queries address consequential blind spots and test
competing explanations; later queries increasingly refine promising regions.
The balance responds to evidence and can differ between functions. Coverage,
model disagreement and noise affect which uncertainty is worth spending a query
on. Geometric coverage alone is not a measured probability of finding the optimum.

The Function 1 and Function 2 explorations establish the two-dimensional baseline.
As dimensionality increases, track how the purpose of queries changes and what
pairwise plots conceal about the full input space. Differences in function
behaviour and starting sample size also matter, so dimension is not the only
explanation for changes in approach.

## Machine-learning workflow

1. **Define the goal and scope** — maximise each function within the course query budget.
2. **Gather the data** — preserve the supplied arrays and record each later observation.
3. **Explore the data** — inspect integrity, output scale, coverage and function-specific evidence.
4. **Clean and preprocess the data** — retain valid extremes and justify any scaling without changing the objective.
5. **Select and engineer features** — retain the supplied coordinates unless evidence supports a change.
6. **Define the machine-learning task** — distinguish sequential maximisation from a surrogate's regression task.
7. **Partition the data** — choose validation appropriate to the small, adaptively growing sample before evaluating models.
8. **Select and train candidate methods** — compare a bounded set of surrogate and acquisition assumptions.
9. **Evaluate and interpret the results** — compare against the best observed value and inspect uncertainty and failure cases.
10. **Deploy and iterate** — validate and submit a chosen point, record the result and revisit earlier decisions.

This keeps the method aligned with the course material without turning the
exercise into a research project. The useful business analogue is sequential
decision-making where experiments are expensive: marketing tests, process
settings, pricing trials or product configurations.

Start with the [Function 1 exploration notebook](notebooks/Function_1/Week_01/01-data-validation-and-exploration.ipynb)
and [Function 1 initial findings](results/function-1-initial-analysis.md). The
notebook focuses on the supplied data, ranked locations, sampling gaps and output
scales, then compares sequential maximin with average coverage and optimises the
latter to propose the first query. That notebook fits no predictive model;
its proposed input has since been evaluated in round 1.

The [follow-on coverage study](notebooks/Function_1/Week_01/01b-function-1-batch-coverage.ipynb)
plans two and three points jointly, compares that with sequential selection,
and records the measured computation cost.

The [Function 2 exploration](notebooks/Function_2/Week_01/01c-function-2-initial-exploration.ipynb)
provides a descriptive first look at its ten observations, spatial coverage and
output scales, with the course's noisy log-likelihood description as context.
It then combines coverage and discrimination between explicit fitted explanations
to compare possible next queries, including sensitivity to the blend weight,
noise and model assumptions.

The [Function 2 decision-risk study](notebooks/Function_2/Week_01/01d-function-2-decision-risk.ipynb)
then measures expected shortfall of a final recommendation, estimates the value
of one answer, and tests first-query choices with twelve simulated queries still
to come. Its risk values are conditional on the fitted models and finite pool;
the thirteen-query comparison uses a declared taper policy.

The [Function 3 exploration](notebooks/Function_3/Week_01/01e-function-3-initial-exploration.ipynb)
extends the descriptive analysis to fifteen observations in three dimensions.
It separates pairwise projections from full-cube sampling distances and examines
how one extreme output influences the apparent relationship with the third input.
No predictive model or next-query selection is included in this first look.

## Historical round 1 approach across all eight functions

The [first-query proposals](results/first-query-proposals.md) collect the eight
suggested inputs and their rationale. The [structured proposal record](submissions/round-01-proposals.json)
includes portal strings, numerical evidence, runtime versions and source hashes.
These files preserve the original proposed inputs; the separate round 1
results record now confirms their evaluation.

The Week 1 descriptive notebooks for Functions 4–8 show every supplied observation,
all coordinate-pair projections, full-space coverage summaries and output scales.
Functions 1 and 3–8 have knowledge-gradient follow-ons with additive, local and
separate-length models, held-out prediction checks, noise sensitivity, a larger
candidate-pool check and paired thirteen-query simulations. Function 2 retains
its earlier function-specific analysis.

Function 1 retained the original coverage query after its raw-output KG models
failed to improve held-out prediction. The other round 1 recommendations
reflected their different data and model evidence; a common decision criterion
does not imply equal confidence across functions.

## Reproducing the analyses

Follow the [repository environment setup](../README.md#modelling-environment)
and provide the authorised course data described in [the data notes](data/README.md).
Week 1 analyses use the preserved initial arrays. Week 2 notebooks load
`observations.csv` beside each notebook, with that function/week directory as
the kernel's working directory. These weekly CSV files are Git-ignored; populate
them from the corresponding accumulated dataset before running a fresh checkout.
For the round 1 state, the versioned `Week2_Notebooks/function_N/observations.csv`
snapshots also provide these rows. Copy the matching function's CSV into its
`notebooks/Function_N/Week_02/` folder. These snapshots are not updated
automatically; check their recorded rounds before using them for later weeks.

The commands below describe the historical first-query reports. From the
repository root, use the full `stage-2-bbo/scripts/` path to each script.

Generate an individual reading copy with `--function N`, then its risk follow-on
with `--function N --decision-risk`, using `scripts/run_initial_analysis.py` in
the existing Python environment. Run `scripts/build_first_query_review.py` to
assemble `.runtime/first-query-review/index.html` and the portable
`.runtime/capstone-first-query-pack.zip`. The HTML embeds figures and hides code.

Install the repository's pinned dependencies with
`.venv\Scripts\python.exe -m pip install -r requirements.txt` from the repository
root. Matplotlib and nbconvert are required for the reading copies and pack.

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
| [`Week2_Notebooks/`](Week2_Notebooks/) | Function-by-function practice notebooks and independent working data snapshots |
| `index.html`, `styles.css` | Static Stage 2 web workspace and its styling |
| `src/` | Reusable BBO code |
| `results/` | Observation and experiment logs safe to publish |
| `submissions/` | Proposed points and portal-submission notes |
| `docs/` | Intake notes and unfinished datasheet, model-card and final-summary templates |

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

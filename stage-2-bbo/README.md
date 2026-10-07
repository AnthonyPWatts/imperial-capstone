# Stage 2: black-box optimisation

This is the assessed capstone problem: maximise eight unknown functions using
only their observed inputs and outputs. The functions range from two to eight
dimensions. Each course round adds a portal observation to the local dataset.

Browse the [Stage 2 workspace](https://anthonypwatts.github.io/imperial-capstone/stage-2-bbo/)
or return to the [Capstone Hub](https://anthonypwatts.github.io/imperial-capstone/).

## Latest results: round 1

The first results email arrived on **5 October 2026 at 00:38 BST**. All eight
returned inputs match the saved proposals. The accumulated local datasets now
contain **183 observations**, with the initial arrays preserved unchanged.

Start with the [Function 1 result analysis](results/function-1-round-01-analysis.md).
Its returned output, `-0.00550934555381189`, is a new minimum; the best observed
value is unchanged. Average sampling distance fell by 7.53776%, while the three
existing GP families still failed to beat the mean baseline on held-out RMSE.
The [Function 2 result analysis](results/function-2-round-01-analysis.md) records
its fifth-ranked return, unchanged incumbent and stronger held-out support for
the x1-only and flexible models. Its next-query choices remain sensitive to
model and noise assumptions. Functions 3–8 have been ingested but not yet
reviewed in detail.

See the [round results record](submissions/round-01-results.json) and
[accumulated data layout](data/README.md#accumulated-observations-round-1).
The first-query material below describes historical pre-evaluation work;
its original proposal JSON and source arrays remain unchanged.

## Module 12 intake

The initial data were collected and verified on **24 September 2026**: 16 NumPy
arrays containing 175 observations across all eight functions. The original ZIP
and extracted arrays are available under `data/` and remain Git-ignored.

The course allows **one evaluated query per function per week**, over 13 rounds
in Modules 12–24. Pending inputs can be revised before the week ends or processing
occurs. See the [Module 12 intake notes](docs/module-12-intake.md) for sources,
deadlines, verified data shapes and the first reflection requirements.

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

## Intended approach

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

Start with the [Function 1 exploration notebook](notebooks/01-data-validation-and-exploration.ipynb)
and [Function 1 initial findings](results/function-1-initial-analysis.md). The
notebook focuses on the supplied data, ranked locations, sampling gaps and output
scales, then compares sequential maximin with average coverage and optimises the
latter to propose the first query. That notebook fits no predictive model;
its proposed input has since been evaluated in round 1.

The [follow-on coverage study](notebooks/01b-function-1-batch-coverage.ipynb)
plans two and three points jointly, compares that with sequential selection,
and records the measured computation cost.

The [Function 2 exploration](notebooks/01c-function-2-initial-exploration.ipynb)
provides a descriptive first look at its ten observations, spatial coverage and
output scales, with the course's noisy log-likelihood description as context.
It then combines coverage and discrimination between explicit fitted explanations
to compare possible next queries, including sensitivity to the blend weight,
noise and model assumptions.

The [Function 2 decision-risk study](notebooks/01d-function-2-decision-risk.ipynb)
then measures expected shortfall of a final recommendation, estimates the value
of one answer, and tests first-query choices with twelve simulated queries still
to come. Its risk values are conditional on the fitted models and finite pool;
the thirteen-query comparison uses a declared taper policy.

The [Function 3 exploration](notebooks/01e-function-3-initial-exploration.ipynb)
extends the descriptive analysis to fifteen observations in three dimensions.
It separates pairwise projections from full-cube sampling distances and examines
how one extreme output influences the apparent relationship with the third input.
No predictive model or next-query selection is included in this first look.

## First-query review across all eight functions

The [first-query proposals](results/first-query-proposals.md) collect the eight
suggested inputs and their rationale. The [structured proposal record](submissions/round-01-proposals.json)
includes portal strings, numerical evidence, runtime versions and source hashes.
These files preserve the original proposed inputs; the separate round 1
results record now confirms their evaluation.

Functions 4–8 now have descriptive notebooks showing every supplied observation,
all coordinate-pair projections, full-space coverage summaries and output scales.
Functions 1 and 3–8 have knowledge-gradient follow-ons with additive, local and
separate-length models, held-out prediction checks, noise sensitivity, a larger
candidate-pool check and paired thirteen-query simulations. Function 2 retains
its earlier function-specific analysis.

Function 1 retains the original coverage query after its raw-output KG models
fail to improve held-out prediction. The other recommendations reflect their
different data and model evidence; a common decision criterion does not imply
equal confidence across functions.

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
| [`Week2_Notebooks/`](Week2_Notebooks/) | Function-by-function practice notebooks and independent working data snapshots |
| `index.html`, `styles.css` | Static Stage 2 web workspace and its styling |
| `src/` | Reusable BBO code |
| `results/` | Observation and experiment logs safe to publish |
| `submissions/` | Proposed points and portal-submission notes |
| `docs/` | Datasheet, model card and non-technical final summary |

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

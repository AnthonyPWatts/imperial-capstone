# How the experiments are chosen

[Start with the introduction](../README.md) for the plain-language story. This
guide explains the decision rules, their evidence and their limits. It records
the approach through the evaluated Week 3 inputs, as of 14 October 2026. Week 4
notebooks are prepared, with initial runs saved pending review.

## The decision in one sentence

Choose one input per function that is worth spending the next scarce evaluation
on. A predicted high score is one reason; resolving an important uncertainty is
another. The objective is sequential maximisation. Any regression model is a
helper for that decision, rather than the final product.

## Terms used below

An **incumbent** is the best output observed so far. A **surrogate** is a fitted
model standing in for the unknown function. An **acquisition rule** scores
possible inputs so that one can be selected for evaluation. These are separate
pieces: predicting accurately on held-out observations and choosing a useful
next experiment are related, but one does not establish the other.

## Inputs, outputs and objective

Each function accepts a numerical vector with coordinates in **`[0, 1)`** and
returns one scalar output. Every function is maximised, including those with
negative outputs: a less negative value is better. The functions have different
output scales, so raw values are compared within each function.

| Function | Input dimensions | Initial observations | Observations after round 3 |
| --- | ---: | ---: | ---: |
| 1 | 2 | 10 | 13 |
| 2 | 2 | 10 | 13 |
| 3 | 3 | 15 | 18 |
| 4 | 4 | 30 | 33 |
| 5 | 4 | 20 | 23 |
| 6 | 5 | 20 | 23 |
| 7 | 6 | 30 | 33 |
| 8 | 8 | 40 | 43 |

For a function with `n` observations and `d` coordinates, the input array has
shape `(n, d)` and the output array has shape `(n,)`. Portal inputs use exactly
six decimal places, hyphens between coordinates and no spaces. For example,
Function 2's evaluated input `0.774999-0.949999` returned
`0.2834440433397594`. A surrogate regression model learns from these input/output
pairs and predicts responses; a Gaussian process also estimates uncertainty.
The query-selection rule uses that information to choose the next input.

The budget is 13 additional evaluations per function, or 104 across all eight.
After round 3, 10 query opportunities remain per function, including Week 4.
Weekly feedback limits adaptation. Sparse observations, unknown function
structure and possible noise limit what can be inferred; the true global
maxima are unavailable for checking. Progress is measured by best observed
outputs, supported by model diagnostics and a record of why each query was chosen.

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

## Week 2 approach and submitted inputs

The [Week 2 portal inputs](../submissions/Week_02/submissions.txt) contain one
evaluated input per function, in function-number order. Their selection used two routes:

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

A mean baseline predicts the average observed output regardless of the input.
In leave-one-out validation, each observation takes a turn being left out of
the fit so that its prediction can be compared with its real output.
**RMSE** (root mean square error) summarises those prediction errors in the
output's units; lower is better. This checks prediction on the available
observations, rather than directly testing how well the next-query rule works.

| Function | Week 2 selection | Evidence |
| --- | --- | --- |
| 1 | First point of a four-point coverage batch | [Exploration](../notebooks/Function_1/Week_02/exploration.ipynb) |
| 2 | First point of a three-point coverage batch | [Exploration](../notebooks/Function_2/Week_02/exploration.ipynb) |
| 3 | First point of a three-point coverage batch | [Exploration](../notebooks/Function_3/Week_02/exploration.ipynb) |
| 4 | GP upper confidence bound (UCB), near the promising region | [Model predictions](../notebooks/Function_4/Week_02/model_predictions.ipynb) |
| 5 | GP UCB, all coordinates at `0.999999` | [Model predictions](../notebooks/Function_5/Week_02/model_predictions.ipynb) |
| 6 | GP UCB, probing the low-x5 boundary | [Model predictions](../notebooks/Function_6/Week_02/model_predictions.ipynb) |
| 7 | GP UCB, testing near the best observation | [Model predictions](../notebooks/Function_7/Week_02/model_predictions.ipynb) |
| 8 | GP UCB, testing a different corner | [Model predictions](../notebooks/Function_8/Week_02/model_predictions.ipynb) |

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

## Week 3 submitted inputs

The [Week 3 portal inputs](../submissions/Week_03/submissions.txt) follow the
approved notebooks. Functions 1–3 continue their saved exploration batches
with the next shortlisted point. Functions 4–8 use their updated model
predictions and GP UCB recommendations. The round 2 results did not justify
changing these approaches. Submission is now confirmed in the
[submission log](../submissions/README.md); the saved notebooks retain their
pre-submission wording as the record of how the inputs were selected.

The Week 3 return arrived on 14 October 2026. All eight inputs match these
submissions; Function 7 improved its incumbent to `2.55241765269229`.
The Week 4 copies use the 199 confirmed observations. Functions 1–3 advance
past the two evaluated points in their saved Week 2 shortlists; Functions 4–8
retain their model and candidate-search settings for the next comparison.
Some initial runs are saved; their Week 4 validation, forecasts and query
decisions remain to be reviewed. See the [notebook guide](../notebooks/README.md).

## How the first-round approach developed

Start with the [Function 1 exploration notebook](../notebooks/Function_1/Week_01/01-data-validation-and-exploration.ipynb)
and [Function 1 initial findings](../results/function-1-initial-analysis.md). The
notebook focuses on the supplied data, ranked locations, sampling gaps and output
scales, then compares sequential maximin with average coverage and optimises the
latter to propose the first query. That notebook fits no predictive model;
its proposed input has since been evaluated in round 1.

The [follow-on coverage study](../notebooks/Function_1/Week_01/01b-function-1-batch-coverage.ipynb)
plans two and three points jointly, compares that with sequential selection,
and records the measured computation cost.

The [Function 2 exploration](../notebooks/Function_2/Week_01/01c-function-2-initial-exploration.ipynb)
provides a descriptive first look at its ten observations, spatial coverage and
output scales, with the course's noisy log-likelihood description as context.
It then combines coverage and discrimination between explicit fitted explanations
to compare possible next queries, including sensitivity to the blend weight,
noise and model assumptions.

The [Function 2 decision-risk study](../notebooks/Function_2/Week_01/01d-function-2-decision-risk.ipynb)
then measures expected shortfall of a final recommendation, estimates the value
of one answer, and tests first-query choices with twelve simulated queries still
to come. Its risk values are conditional on the fitted models and finite pool;
the thirteen-query comparison uses a declared taper policy.

The [Function 3 exploration](../notebooks/Function_3/Week_01/01e-function-3-initial-exploration.ipynb)
extends the descriptive analysis to fifteen observations in three dimensions.
It separates pairwise projections from full-cube sampling distances and examines
how one extreme output influences the apparent relationship with the third input.
No predictive model or next-query selection is included in this first look.

### Bringing the first-query studies together

The [first-query proposals](../results/first-query-proposals.md) collect the eight
suggested inputs and their rationale. The [structured proposal record](../submissions/round-01-proposals.json)
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

## Reading the evidence without overstating it

Coverage reduces gaps between sampled locations. It does not use outputs and
cannot by itself identify a high-scoring region. Model-guided choices can use
outputs, but their uncertainty is conditional on a fitted model. In either
case, the returned portal score is the evidence for whether that particular
experiment improved the incumbent.

Continue to [results and progress](../results/progress.md) to see those returns,
or [reproducing the analyses](reproducing-the-analyses.md) to inspect the work.

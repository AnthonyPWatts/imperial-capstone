# Function 1: initial findings

Historical pre-evaluation analysis. The proposed average-coverage point was
subsequently evaluated; see the [first returned result](function-1-round-01-analysis.md).

Ten supplied observations, before any new evaluation. The
[exploration notebook](../notebooks/01-data-validation-and-exploration.ipynb)
contains the data preview and figures; loading and plotting are kept in a helper.

## Locations and coverage

The highest output occurs near `(0.731, 0.733)` and the lowest near
`(0.650, 0.682)`. These are the closest observed locations, about 0.096 units
apart. This contrast does not establish the function's shape between them.
Map labels are output ranks: 1 is highest, 10 is lowest. They are not row IDs,
and rank differences do not measure output differences.

A broad central gap lies near `(0.42, 0.46)`; several boundary regions also have
poor coverage. The sampling-gap heatmap measures distance to the nearest
observed input. It is not a predicted response surface or uncertainty estimate.

## Output scale

Seven outputs are positive and three are negative; none is exactly zero.
The lowest output, approximately `-0.003606`, dominates the raw scale. All other
magnitudes are at most `7.71e-16`, which is also the highest observed output.
Non-zero magnitudes span about 121 orders of magnitude.

The ordered log-magnitude plot retains signs through its markers. It reveals
small values without changing the optimisation objective: a large negative
magnitude remains a poor output. The tiny values could contain structure, but
the observations do not establish their usefulness for finding a larger
positive response. A second chart pair omits rank 10 for display only. Its raw
axis expands to show the remaining values; the ranks and logarithmic axis remain
comparable with the full-data view. The largest positive value then dominates
the raw scale, while the two remaining negative outputs are visible in the log
view. All ten observations remain in the data and the table.

## Reproduction

Run `stage-2-bbo/scripts/run_initial_analysis.py` with the capstone Python
environment. It creates a self-contained HTML reading copy, an executed notebook
and six figures under `.runtime/function-1-initial-analysis/`. No predictive model is
fitted or portal query submitted by the analysis.

## First query: sequential maximin

The supplied results do not establish a useful target value or noise level.
The first query therefore prioritises coverage: maximise the distance from the
candidate to its nearest existing observation. This uses all ten input
locations, regardless of their outputs.

Searching the two-dimensional domain, including its edges and corners, selects
`(0.000000, 0.999999)`. Its nearest observation is about `0.397752` units away,
compared with `0.316185` for the central alternative `(0.421062, 0.463562)`.
The portal representation is **`0.000000-0.999999`**. It is the maximin candidate,
not a settled first-query choice, and has not been submitted.

The corner wins this geometric criterion without predicting a high output.
The next decision will use the updated observations once its result is known.
See [sequential maximin design](https://bookdown.org/rbg/surrogates/chap4.html#sequential-maximin-design)
for the selection principle.

## Boundary trade-off

Maximin rewards isolation of the new point, rather than the area helped by it.
A uniform 701-by-701 grid over the usable square gives these approximate
reductions in mean distance to the nearest sample after adding one point:

| Candidate | Mean-distance reduction |
| --- | ---: |
| Maximin corner `(0.000000, 0.999999)` | 3.37% |
| Inset point `(0.07, 0.93)` | 5.64% |
| Central gap `(0.421062, 0.463562)` | 7.38% |

The centre is best among these three for this different criterion; global
optimality has not been established. The figures describe geometric coverage,
not expected function improvement or measured information gain. This comparison
motivates optimising average coverage directly.

## Optimised average coverage: proposed first query

The proposed input is **`(0.468429, 0.435440)`**, with portal representation
**`0.468429-0.435440`**. It reduces mean nearest-sample distance from approximately
`0.152412231` to `0.140923767`, a **7.53776%** improvement. It has not been submitted.

The objective weights the whole unit square uniformly and uses all ten input
locations, without using their outputs. Distance is integrated geometrically
over clipped Voronoi cells. Two seeded differential-evolution searches and
local refinement from other candidate regions agree on the solution. Rounding
to the selected six-decimal input changes the objective by about `3.3e-14`.

Known integral values, independent numerical quadrature and dense-grid checks
support the calculation. Agreement between searches provides numerical evidence,
not a formal certificate of global optimality. This is a coverage decision, not
a prediction that the unknown output will improve.

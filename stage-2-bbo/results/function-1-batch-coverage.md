# Function 1: planning two or three coverage points

The [batch-coverage notebook](../notebooks/Function_1/Week_01/01b-function-1-batch-coverage.ipynb)
extends the one-point analysis. All ten original locations remain fixed. The
objective is mean distance to the nearest original or proposed sample, uniformly
weighted over the unit square. Outputs do not enter the score.

## Numerical plans

| New points | Proposed coordinates | Mean distance | Reduction from the original ten |
| --- | --- | ---: | ---: |
| 1 | `(0.468429, 0.435440)` | 0.140923767 | 7.5378% |
| 2 | `(0.112627, 0.828222)`; `(0.468429, 0.435440)` | 0.130673391 | 14.2632% |
| 3 | `(0.112627, 0.828222)`; `(0.460039, 0.441074)`; `(0.681832, 0.121853)` | 0.123742720 | 18.8105% |

Original mean distance: `0.152412231`. Each row is an alternative unordered plan,
not an instruction to submit every listed point. Nothing has been submitted.

## Joint versus sequential selection

For two points, joint planning agrees with choosing the best point and then the
best next point to the displayed coordinate precision. For three points, the
joint plan shifts the central and lower-right locations. Its advantage over
sequential placement is approximately **0.00566 percentage points** of total
coverage improvement. The distinction is measurable but small for this dataset.

## Computational cost

The executed study on 24 September 2026 measured approximately **5.95 seconds**
for the joint two-point search and **19.65 seconds** for three points. The complete
study, including the one-point reference and sequential comparisons, took about
28.82 seconds before plotting. These timings are machine-dependent.

Three points are practical here. The spatial integral remains two-dimensional,
although the search has six coordinate variables. This does not estimate the
cost of coverage planning for higher-dimensional functions. The search is not
exhaustive enumeration: it combines two seeded differential-evolution searches
with local refinement, including a start from the sequential plan.

## Verification and reproduction

Eight focused tests passed, covering integral values, independent quadrature,
overlap, duplicates, point order and dimensionality. Dense-grid checks against
all three numerical plans agreed with the geometric mean distances within
`2.6e-7`. Independently seeded searches and local refinements agreed; no formal
global-optimality certificate is claimed. Source arrays were unchanged.

Run with the capstone environment:

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/run_initial_analysis.py --batch-coverage
```

The ignored `.runtime/function-1-batch-coverage/` directory contains the executed
notebook, self-contained HTML, comparison figure and machine-readable study.

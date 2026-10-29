# Shared notebook helpers

Keep reusable implementation here. Notebooks declare their settings and run
calculations beside their data. Where a week has a `report.md`, put the
human-readable discussion and interpretation in that report; concise code
comments can explain the calculations. Historical teaching notebooks retain
their existing explanations.

| Module | Responsibility |
| --- | --- |
| `plot_helpers.py` | Observed-output plots, including a configurable linear number line; coverage and sampling-gap plots; fixed-plan progressions for two inputs and selected projections from any input count; observed two-input terrains; common interactive plot settings and colours. |
| `output_helpers.py` | Weekly output summaries and chart captions. |
| `exploration_review_helpers.py` | Validated snapshot loading, ranked plotting data and remaining fixed-plan inputs with portal strings, independent of input dimension. |
| `observation_review_helpers.py` | Dimension-independent measured review preparation, official round progress, saved historical forecast checks, measured-output plots and strongest input profiles. |
| `coverage_helpers.py`, `coverage_cache.py` | Geometric coverage comparisons and their local caches. |
| `query_budget.py` | Remaining query allowance. |
| `model_helpers.py` | Configurable fixed-length-scale model families, leave-one-out comparisons, GP teaching curves, validation plots and hypothetical query scoring, shared across input dimensions. |
| `model_review_helpers.py` | Configurable regression families, optional scaled polynomial ridge, held-out review with RMSE or MAE selection, candidate construction, geometric tie-breaks and sensitivity calculations for any input count. |
| `model_review_plots.py` | Held-out predictions with configurable target units, latest-result checks, conditional coordinate curves, paired cohort errors and proposal profiles, independent of input dimension. |
| `cohort_review_helpers.py` | Validate pooled snapshots against official records, read saved proposal or named analysis summaries, align identical held-out targets by ID and compare fits on common candidates and proposals, for any input count. |
| `noise_review_helpers.py` | Full-input neighbour differences, exact-input repeat candidates, cutoff sensitivity and separate conditional GP noise fits, for any input count. |
| `two_input_model_helpers.py` | Model curves in the input square, prediction grids, contour maps and fitted terrains. |
| `three_input_model_helpers.py` | Three-input measured clouds, conditional coordinate curves, contour planes, terrain sliders, proposal exports and a paired cohort comparison map. |
| `surrogate_study_helpers.py` | Bounded neural/regression/classification comparisons, training-only class thresholds, seed aggregation, checked input backpropagation, actual SVM support vectors and hypothetical surrogate paths. |
| `surrogate_study_plots.py` | Surrogate validation, seed variation, confusion matrices, complete input-gradient profiles, conditional decision boundaries and support-vector profiles. |
| `surrogate_study_report.py` | Generate the seven reflection-evidence sections from numerical results, preserving target units, model limitations and the ten-step course lifecycle. |

Use `Function_N/FunctionSpecificShared/` for code tied to one function, shared
across its weekly notebooks. For example, Function 1's mirrored output number
line, magnitude plots and signed-log target reviews live in
[`Function_1/FunctionSpecificShared/`](../Function_1/FunctionSpecificShared/).
The transform accepts an explicit scale; choosing that scale remains a notebook
decision. New function folders need this directory only when they have helpers
to put in it.

Function 1's Week 5 target-review helpers refit the same configured families
independently to signed and magnitude targets. They score the saved coverage
point separately from hypothetical alternatives, validate the saved official
review, and use generic paired-target and common-candidate comparisons.
Those calculations accept arbitrary input dimensions; exporting the input-square
contours and query maps remains explicitly two-dimensional. Fixed-model defaults
and the shared review's default RMSE selection remain unchanged for other callers.
Both Function 1 versions declare MAE selection and the same target scale at the
top, with interpretation in the week-level report.

Use `plot_helpers.plot_output_number_line` for outputs with a moderate range,
such as Function 2. Function 1's function-specific version uses mirrored log
axes for its much wider range. Both accept an output review and star the latest
submission; number lines work independently of the number of input dimensions.

For a few input pairs, `plot_sampling_gap_projections` combines sampling-gap
maps in one row with a shared colour scale and legend. An optional boolean
`highlight` mask stars the same observations in every projection. Distances
remain local to each two-input plane. The existing pairwise helpers still
return separate figures for notebooks that need them.

Run notebooks using their own directory and the repository's `.venv` kernel.
Add `notebooks/Shared` and, when needed, that function's `FunctionSpecificShared`
directory to `sys.path`, following the Week 4 notebooks. Keep `observations.csv`,
saved plans and generated figures with their week; helpers do not evaluate the
course functions or submit queries.

Function 2's Week 5 official exploration review uses a settings cell for its
input columns, expected round history, full-precision plan and plot options.
The shared preparation and remaining-plan helpers accept any input dimension;
the terrain and fixed-plan figures retain their explicit two-input renderers.
The notebook runs from `Function_2/Week_05/official/` and its discussion lives
in the parent week's `report.md`.

The official Function 2 Week 5 model notebook follows the same pattern. Its
settings cell declares the model families, feature subsets, noise and kernel
assumptions, seeded searches, geometric tie-break and sensitivity cases.
Numerical review helpers accept arbitrary input counts; explicit two-input
renderers export this function's contours, slices, terrain and proposal maps.
The final JSON output preserves the saved-summary format consumed by the
separate cohort notebook. Interpretation and the course lifecycle remain in
the week-level report.

Both Function 2 Week 5 cohort notebooks follow the same settings-and-helpers
pattern. Snapshot validation preserves official records exactly and keeps
peer round labels out of the owner's timeline. Pooled predictions are aligned
to official target IDs before paired scoring. Nearby-pair noise proxies use
all input coordinates; distinct inputs are not treated as repeated experiments.
Separate white-noise GP fits report variance in normalised units and SD in
original output units, with fitting warnings retained. These diagnostic fits
do not change model selection or saved query proposals. Their interpretation
and the cohort discussion live in the week-level report.

Function 3 Week 5 uses the same `official/` and `cohort/` layout, with one
week-level report. Its optional quadratic ridge model is constructed by the
shared regression factory, with polynomial features and scaling fitted inside
each held-out fold. The generic fixed-plan projection renderer accepts a
configured list of input pairs; all calculations still use the full input
vectors. Three-input clouds, conditional planes and terrains use their explicit
three-dimensional renderers. The course description does not label Function 3
noisy, so these notebooks omit noise diagnostics and the earlier
noise-sensitivity table, retaining the existing GP diagonal regularisation and
all model scores, paired comparisons and provisional inputs.

Function 4 Week 5 follows the same folder and report pattern, keeping
`observation_review.ipynb` as its measured-review name. Its new model runs
retain the five established families. Shared review preparation accepts any
input count, keeps official progress separate from peer labels and verifies
the historical printed forecast at the actual submitted coordinates. Pooled
output charts distinguish peers and retain their full source IDs in the
highest-output labels. Conditional model curves hold every other coordinate
fixed, while proposal-profile charts show every input without projecting it
away. The problem description does not label Function 4 noisy, so no noise
analysis is included; its existing GP regularisation setting is retained.

The separate Week 5 `surrogate_comparison.ipynb` studies use only the official
snapshots and write under their own `surrogate-study/` folders. Their MLP input
gradients use fitted public weights and biases, include input and regression
target scaling, and are checked against finite differences. Classifier thresholds
use each fold's training targets, and the full-fit boundary is labelled separately.
SVM scores are not probabilities; unavailable probability metrics are saved as
JSON null. Seed variation is optimisation sensitivity, not GP uncertainty.
The study does not change existing model defaults, saved proposals or submissions.

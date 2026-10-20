# Shared notebook helpers

Keep reusable implementation here and leave each notebook's assumptions,
teaching sequence, interpretation and query decision beside its data.

| Module | Responsibility |
| --- | --- |
| `plot_helpers.py` | Observed-output plots, including a linear number line; coverage and sampling-gap plots; fixed-plan progressions for two inputs; observed two-input terrains; common interactive plot settings and colours. |
| `output_helpers.py` | Weekly output summaries and chart captions. |
| `observation_review_helpers.py` | Measured-output number lines, progress across returned rounds and strongest input profiles for the Week 4 reviews of Functions 4–8. |
| `coverage_helpers.py`, `coverage_cache.py` | Geometric coverage comparisons and their local caches. |
| `query_budget.py` | Remaining query allowance. |
| `model_helpers.py` | Model construction, leave-one-out comparisons, GP teaching curves, validation plots and hypothetical query scoring, shared across input dimensions. |
| `two_input_model_helpers.py` | Model curves in the input square, prediction grids, contour maps and fitted terrains. |
| `three_input_model_helpers.py` | Three-input clouds, conditional coordinate curves, contour planes and terrain sliders. |

Use `Function_N/FunctionSpecificShared/` for code tied to one function, shared
across its weekly notebooks. For example, Function 1's mirrored output number
line, historical magnitude plots and signed-log target transform live in
[`Function_1/FunctionSpecificShared/`](../Function_1/FunctionSpecificShared/).
The transform accepts an explicit scale; choosing that scale remains a notebook
decision. New function folders need this directory only when they have helpers
to put in it.

Use `plot_helpers.plot_output_number_line` for outputs with a moderate range,
such as Function 2. Function 1's function-specific version uses mirrored log
axes for its much wider range. Both accept an output review and star the latest
submission; number lines work independently of the number of input dimensions.

For a few input pairs, `plot_sampling_gap_projections` combines sampling-gap
maps in one row with a shared colour scale and legend. An optional boolean
`highlight` mask stars the same observations in every projection. Distances
remain local to each two-input plane. The existing pairwise helpers still
return separate figures for notebooks that need them.

Run notebooks using their own week directory and the repository's `.venv` kernel.
Add `notebooks/Shared` and, when needed, that function's `FunctionSpecificShared`
directory to `sys.path`, following the Week 4 notebooks. Keep `observations.csv`,
saved plans and generated figures with their week; helpers do not evaluate the
course functions or submit queries.

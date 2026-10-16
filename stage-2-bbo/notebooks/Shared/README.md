# Shared notebook helpers

Keep reusable implementation here and leave each notebook's assumptions,
teaching sequence, interpretation and query decision beside its data.

| Module | Responsibility |
| --- | --- |
| `plot_helpers.py` | Observed-output, coverage and sampling-gap plots; observed two-input terrains; common interactive plot settings and colours. |
| `output_helpers.py` | Weekly output summaries and chart captions. |
| `coverage_helpers.py`, `coverage_cache.py` | Geometric coverage comparisons and their local caches. |
| `query_budget.py` | Remaining query allowance. |
| `model_helpers.py` | Model construction, leave-one-out comparisons, GP teaching curves, validation plots and hypothetical query scoring, shared across input dimensions. |
| `two_input_model_helpers.py` | Model curves in the input square, prediction grids, contour maps and fitted terrains. |
| `three_input_model_helpers.py` | Three-input clouds, conditional coordinate curves, contour planes and terrain sliders. |

Use `Function_N/FunctionSpecificShared/` for code tied to one function, shared
across its weekly notebooks. For example, Function 1's magnitude plots and
signed-log target transform live in
[`Function_1/FunctionSpecificShared/`](../Function_1/FunctionSpecificShared/).
The transform accepts an explicit scale; choosing that scale remains a notebook
decision. New function folders need this directory only when they have helpers
to put in it.

Run notebooks using their own week directory and the repository's `.venv` kernel.
Add `notebooks/Shared` and, when needed, that function's `FunctionSpecificShared`
directory to `sys.path`, following the Week 4 notebooks. Keep `observations.csv`,
saved plans and generated figures with their week; helpers do not evaluate the
course functions or submit queries.

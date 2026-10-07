# Stage 2 side work

Function-by-function practice notebooks and independent working snapshots of
the BBO observations. Each `function_N/` folder keeps its notebook, where
present, beside `inputs.npy`, `outputs.npy` and `observations.csv`.

The snapshots are separate copies and are not updated automatically when new
rounds arrive. See the [data notes](../data/README.md) for the canonical initial
and accumulated datasets and their import process.

Notebooks locate the Stage 2 workspace from the current directory and load
their function's snapshot from this folder. Use the shared Python environment
and dependencies described in the [repository setup](../../README.md#local-setup).

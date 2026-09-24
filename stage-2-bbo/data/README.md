# Local BBO data

The initial input and output `.npy` files were collected on 24 September 2026.
They are ignored by Git because access is course-restricted.

The original `Initial_data_points_starter.zip` is retained unchanged. Extracted
files are under `initial_data/function_1/` through `initial_data/function_8/`,
each containing `initial_inputs.npy` and `initial_outputs.npy`. Keep these
original observations unchanged and store later accumulated observations separately.

Verified starting shapes:

| Function | Input rows | Dimensions |
| --- | ---: | ---: |
| 1 | 10 | 2 |
| 2 | 10 | 2 |
| 3 | 15 | 3 |
| 4 | 30 | 4 |
| 5 | 20 | 4 |
| 6 | 20 | 5 |
| 7 | 30 | 6 |
| 8 | 40 | 8 |

All arrays loaded with `allow_pickle=False`; outputs have shape `(rows,)`.
Values are finite, all input coordinates lie in `[0, 1)`, and input rows are
distinct within each function. There are 175 observations in total.

The local `import-verification-2026-09-24.json` records file hashes and checks.
See the [intake notes](../docs/module-12-intake.md) for the official data link and
submission rules. Do not add the supplied arrays to the repository.

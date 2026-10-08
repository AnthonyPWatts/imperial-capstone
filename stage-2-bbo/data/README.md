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

## Accumulated observations: round 2

The Week 2 return was imported on **8 October 2026**, giving **191 observations**:
12, 12, 17, 32, 22, 22, 32 and 42 rows for Functions 1–8 respectively.
This includes the Week 1 return imported on 5 October and one new observation
per function from Week 2.
The original 175 rows and starter archive remain unchanged.

- `rounds/round-01/` retains complete decoded `inputs.txt` and `outputs.txt`
  attachment text, the plain email body and source metadata. Attachment text
  byte lengths agree with the email metadata (419 and 160 bytes).
- `rounds/round-02/` retains the original downloaded attachments (839 and 330
  bytes), the plain email body and source metadata. These attachments contain
  both rounds; the first round matches the retained return exactly and is
  included only once in the accumulated datasets.
- `accumulated/function_N/inputs.npy` and `outputs.npy` hold the initial rows
  followed by the confirmed round observations, in matching order.
- `accumulated/function_N/observations.csv` provides the same data with stable
  observation IDs and `source_round` (`0` means an initial observation).
- The [results record](../submissions/round-02-results.json) links returned
  values to proposals and source hashes. All eight proposals matched; all
  returned coordinates are new, finite, correctly dimensioned and in `[0, 1)`.

Use `accumulated/` for new analyses. Historical first-query notebooks deliberately
continue to load `initial_data/`, preserving their original evidence. No existing
loader was silently redirected. The initial and accumulated `.npy` arrays use
float64; small non-zero outputs are preserved without decimal-place rounding.

Rebuild the accumulated files with
`.venv/Scripts/python.exe stage-2-bbo/scripts/import_round_results.py --round 2`
from the capstone root. It checks all functions before writing, reconciles the
email and attachments, verifies the latest proposals, and rebuilds from source
rounds rather than appending blindly. It accepts a single-round return or a
complete cumulative history, checking earlier entries against the retained
rounds. Future imports require consecutive source round directories and the
matching proposal JSON or `submissions/Week_NN/submissions.txt` (eight lines in
function order). The results record identifies and hashes the proposal source.
Both raw and accumulated observations remain course-restricted and Git-ignored.

## Week 3 notebook snapshots

`notebooks/Function_1/Week_03/observations.csv` through
`notebooks/Function_8/Week_03/observations.csv` are independent copies of the
round 2 accumulated CSVs, ready for Week 3 notebooks. Each retains the existing
`observation_id,source_round,x1,...,y` schema, with initial observations followed
by `round-01` and `round-02`. Values retain their full float64 precision.

These snapshots are Git-ignored and remain fixed when later rounds are imported.
The local `import-verification-2026-10-08.json` records their hashes, row counts,
exact CSV/NumPy agreement and preservation checks. All eight returned inputs
match the saved Week 2 submission lines; repeated imports produce identical
files. The Week 2 snapshots and existing notebooks remain unchanged.

A historical round 1 working snapshot of all eight datasets is retained in
[`Week2_Notebooks/`](../Week2_Notebooks/), alongside the function-specific practice notebooks.
Those copies are separate files, not links, and are not updated automatically
when new rounds are imported. The canonical source and accumulated directories
described above remain Git-ignored.

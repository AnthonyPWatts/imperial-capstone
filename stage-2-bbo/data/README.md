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

## Accumulated observations: round 1

The Week 1 return was imported on **5 October 2026**, giving **183 observations**:
11, 11, 16, 31, 21, 21, 31 and 41 rows for Functions 1–8 respectively.
The original 175 rows and starter archive remain unchanged.

- `rounds/round-01/` retains complete decoded `inputs.txt` and `outputs.txt`
  attachment text, the plain email body and source metadata. Attachment text
  byte lengths agree with the email metadata (419 and 160 bytes).
- `accumulated/function_N/inputs.npy` and `outputs.npy` hold the initial rows
  followed by the confirmed round observations, in matching order.
- `accumulated/function_N/observations.csv` provides the same data with stable
  observation IDs and `source_round` (`0` means an initial observation).
- The [results record](../submissions/round-01-results.json) links returned
  values to proposals and source hashes. All eight proposals matched; all
  returned coordinates are new, finite, correctly dimensioned and in `[0, 1)`.

Use `accumulated/` for new analyses. Historical first-query notebooks deliberately
continue to load `initial_data/`, preserving their original evidence. No existing
loader was silently redirected. The initial and accumulated `.npy` arrays use
float64; small non-zero outputs are preserved without decimal-place rounding.

Rebuild the accumulated files with
`.venv/Scripts/python.exe stage-2-bbo/scripts/import_round_results.py --round 1`
from the capstone root. It checks all functions before writing, reconciles the
email and attachments, verifies the latest proposals, and rebuilds from source
rounds rather than appending blindly. Future imports require consecutive source
round directories and the matching proposal record. Both raw and accumulated
observations remain course-restricted and Git-ignored.

A versioned working snapshot of all eight accumulated datasets is retained in
[`side-work/`](../../side-work/), alongside the Function 1 practice notebook.
Those copies are separate files, not links, and are not updated automatically
when new rounds are imported. The canonical source and accumulated directories
described above remain Git-ignored.

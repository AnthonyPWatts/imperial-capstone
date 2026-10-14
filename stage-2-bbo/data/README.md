# Local BBO data

## Module 14 reporting cutoff

The [Module 14 overview](../README.md) and strategy reflection use the fixed
[Week 3 notebook snapshots](#week-3-notebook-snapshots): **191 observations
through round 2**, with 12, 12, 17, 32, 22, 22, 32 and 42 rows for Functions 1–8.
The submitted Week 3 queries were selected from that evidence.

The accumulated-data inventory and Week 4 snapshots below include the later
third-query return. They remain intact for subsequent work and must not be
substituted for the Week 3 snapshots when reproducing the Module 14 decisions.

## Source data and recovery

The initial input and output `.npy` files were collected on 24 September 2026.
They are ignored by Git because access is course-restricted.

The layout and verification results below describe the local intake and
analysis records. A fresh checkout does not contain the starter archive,
source feedback, accumulated arrays or weekly CSVs. Restore authorised local
files or use the [historical recovery route](#recovering-earlier-snapshots)
before running the analyses.

The original `Initial_data_points_starter.zip` was retained unchanged at intake.
Restore its extracted files under `initial_data/function_1/` through
`initial_data/function_8/`, each containing `initial_inputs.npy` and
`initial_outputs.npy`. Keep these
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

## Accumulated observations: round 3

The Week 3 return was imported on **14 October 2026**, giving **199 observations**:
13, 13, 18, 33, 23, 23, 33 and 43 rows for Functions 1–8 respectively.
This includes the Week 1 and Week 2 returns imported on 5 and 8 October,
plus one new observation per function from Week 3.
The import preserved the original 175 rows and starter archive.

On a machine holding the complete intake, the local layout is:

- `rounds/round-01/` retains complete decoded `inputs.txt` and `outputs.txt`
  attachment text, the plain email body and source metadata. Attachment text
  byte lengths agree with the email metadata (419 and 160 bytes).
- `rounds/round-02/` retains the original downloaded attachments (839 and 330
  bytes), the plain email body and source metadata. These attachments contain
  both rounds; the first round matches the retained return exactly and is
  included only once in the accumulated datasets.
- `rounds/round-03/` retains the original downloaded attachments (1,259 and 494
  bytes), the plain email body and source metadata. Their three-round cumulative
  history agrees exactly with the earlier returns; each observation is included
  only once. The email arrived at 10:16 BST on 14 October 2026.
- `accumulated/function_N/inputs.npy` and `outputs.npy` hold the initial rows
  followed by the confirmed round observations, in matching order.
- `accumulated/function_N/observations.csv` provides the same data with stable
  observation IDs and `source_round` (`0` means an initial observation).
- The [results record](../submissions/round-03-results.json) links returned
  values to proposals and source hashes. All eight proposals matched; all
  returned coordinates are new, finite, correctly dimensioned and in `[0, 1)`.

Use `accumulated/` for new analyses. Historical first-query notebooks deliberately
continue to load `initial_data/`, preserving their original evidence. No existing
loader was silently redirected. The initial and accumulated `.npy` arrays use
float64; small non-zero outputs are preserved without decimal-place rounding.

Rebuild the accumulated files with
`.venv/Scripts/python.exe stage-2-bbo/scripts/import_round_results.py --round 3`
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
`notebooks/Function_8/Week_03/observations.csv` hold independent copies of the
round 2 accumulated CSVs used by the Week 3 notebooks. Each retains the existing
`observation_id,source_round,x1,...,y` schema, with initial observations followed
by `round-01` and `round-02`. Values retain their full float64 precision.

Week 3 submission and its subsequent return are confirmed in the
[submission log](../submissions/README.md). These historical snapshots retain
191 observations through round 2; the third return is in the Week 4 copies.

These snapshots are Git-ignored and remain fixed when later rounds are imported.
The local `import-verification-2026-10-08.json` records their hashes, row counts,
exact CSV/NumPy agreement and preservation checks. All eight returned inputs
match the saved Week 2 submission lines; repeated imports produce identical
files. The Week 2 snapshots and existing notebooks remain unchanged.

## Week 4 notebook snapshots

`notebooks/Function_1/Week_04/observations.csv` through
`notebooks/Function_8/Week_04/observations.csv` are independent copies of the
round 3 accumulated CSVs. They contain 199 observations in total, with the
existing Week 3 rows followed by `round-03` and `source_round` set to 3.
All eight inputs match `submissions/Week_03/submissions.txt`. Function 7 is
the only new incumbent, at `2.55241765269229`.

The local `import-verification-2026-10-14.json` records attachment-size checks,
source hashes, reconciliation of the cumulative history, exact CSV/NumPy
agreement, rebuild idempotency and preservation of earlier notebooks and
local edits. The snapshots remain Git-ignored. On a fresh checkout, recover
Week 3 as below, then append each function's `point` and `output` from
[`round-03-results.json`](../submissions/round-03-results.json), retaining
full precision and the existing schema. Expected counts are 13, 13, 18, 33,
23, 23, 33 and 43. Leave the earlier weekly CSVs unchanged.

## Recovering earlier snapshots

The superseded `stage-2-bbo/Week2_Notebooks/` directory was removed from the
current tree after its analyses were rehomed under `notebooks/Function_N/Week_NN/`.
Commit `ef9494d2b58b71cd2c16d7f9d92a0608d4f70ed8` retains its three practice
notebooks and round 1 datasets for all eight functions. No additional archive
is needed. This cleanup does not remove those files from Git history.

To recover the eight Week 2 CSVs locally, run the following from the capstone
repository root in PowerShell. It requires that historical commit to be
available and refuses to overwrite an existing weekly CSV:

```powershell
@'
from pathlib import Path
import subprocess

snapshot = "ef9494d2b58b71cd2c16d7f9d92a0608d4f70ed8"
targets = [Path(f"stage-2-bbo/notebooks/Function_{n}/Week_02/observations.csv")
           for n in range(1, 9)]
if any(path.exists() for path in targets):
    raise SystemExit("A Week 2 CSV already exists; inspect it before restoring.")
payloads = [subprocess.check_output([
    "git", "show", f"{snapshot}:stage-2-bbo/Week2_Notebooks/function_{n}/observations.csv"
]) for n in range(1, 9)]
for path, payload in zip(targets, payloads):
    with path.open("xb") as output:
        output.write(payload)
'@ | python -
```

The restored CSVs contain **183 observations**: 11, 11, 16, 31, 21, 21, 31
and 41 rows for Functions 1–8. They match the historical NumPy arrays exactly
when parsed as float64. `source_round` is numeric: 0 for the initial rows and
1 for the first returned observation.

For Week 3, create separate CSVs from those snapshots and append the matching
function's `point` and `output` from
[`round-02-results.json`](../submissions/round-02-results.json), with
`observation_id` set to `round-02` and `source_round` to 2. Preserve coordinate
order and full float64 precision. The resulting counts must be 12, 12, 17, 32,
22, 22, 32 and 42 (**191 total**). Leave the Week 2 snapshots unchanged.

The 175 initial rows can also be recovered by selecting `source_round == 0`.
The historical `inputs.npy` and `outputs.npy` files contain the same rows plus
round 1; removing their final row reconstructs all sixteen original arrays.
Their reconstructed hashes were verified against the intake hashes retained
in [`round-01-results.json`](../submissions/round-01-results.json).

This recovers the numerical observations, not the original email bodies or
attachments. The normal import script still requires the complete local source
history described above; it does not retrieve files from Git automatically.
All recovered working files should remain in the ignored data or weekly CSV paths.

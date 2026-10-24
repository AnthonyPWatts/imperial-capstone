# Local BBO data

## Module 14 reporting cutoff

The [Module 14 overview](../README.md) and strategy reflection use the fixed
[Week 3 notebook snapshots](#week-3-notebook-snapshots): **191 observations
through round 2**, with 12, 12, 17, 32, 22, 22, 32 and 42 rows for Functions 1–8.
The submitted Week 3 queries were selected from that evidence.

The accumulated data and Week 5 snapshots now include the fourth-query return.
Earlier weekly snapshots remain intact and must not be
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

## Current accumulated observations: round 4

The Week 4 results email arrived on **23 October 2026 at 03:53 BST**. Its
original attachments contain all four rounds and agree exactly with the retained
history. All eight returned inputs match `submissions/Week_04/submissions.txt`.
Importing the eight new returns gives **207 observations**: 14, 14, 19, 34, 24,
24, 34 and 44 rows for Functions 1–8. Function 7 is the only new incumbent,
at `2.6189682154023832`.

`rounds/round-04/` retains the unchanged attachments (1,679 and 661 bytes),
plain email body and source metadata. The existing importer rebuilt `accumulated/`
from all source rounds; importing twice produced identical files. Rebuild from
the capstone root with:

```powershell
.venv\Scripts\python.exe stage-2-bbo/scripts/import_round_results.py --round 4
```

The [round 4 result record](../submissions/round-04-results.json) retains proposal
matches, full-precision outputs, before/after incumbents and source hashes.
The local `import-verification-2026-10-23.json` records attachment byte lengths,
exact CSV/NumPy agreement, idempotency and preservation of earlier files.
The original arrays, raw feedback and weekly CSVs remain Git-ignored.

## Week 5 notebook snapshots

The official Week 5 CSVs for Functions 1–8 are independent copies of the
round 4 accumulated CSVs, containing **207 observations**. Functions 1–4
now keep their copies in `notebooks/Function_N/Week_05/official/observations.csv`
alongside their official notebooks; Functions 5–8 still use
`notebooks/Function_N/Week_05/observations.csv`. They retain the
`observation_id,source_round,x1,...,y` schema and full float64 precision.
Each earlier Week 4 snapshot is preserved exactly, followed by one `round-04`
row with `source_round` set to 4. No pooled cohort observations are included.

For a fresh checkout, recover Week 4 as described below, then append each
function's `point` and `output` from the round 4 result record. Expected counts
are 14, 14, 19, 34, 24, 24, 34 and 44. Preserve the earlier CSVs and load the
new snapshots with `float_precision="round_trip"`.

## Function 1 Week 5 cohort comparison

The notebooks in `notebooks/Function_1/Week_05/cohort/` use **20 rows**:
the unchanged 14 official observations followed by six accepted peer returns
from the retained 20 October study. Both `official/` and `cohort/` contain
`exploration.ipynb`, `model_predictions.ipynb`, their own `observations.csv`
and `figures/`. Discussion lives in the week-level `report.md`. The final
saved coverage query is retained.

Reconstruct the snapshot from the earlier Function 1 study's 19-row CSV.
Verify that its 13 non-peer rows exactly match the current official history
through round 3, then append only its six `peer-` rows to the current
14-row official CSV. Retain the shared initial batch once, preserve the
`observation_id,source_round,x1,x2,y` schema and load full float64 precision.
All 20 outputs are non-zero: preserve scientific notation and sign, including
the very small positive and negative returns. Do not round them to zero.

The local `cohort/data-audit.json` retains source commits, checksums,
previous exclusions, duplicate handling and CSV hashes. All four retained
source files were verified against the pinned audit without refreshing
repositories. CSVs and the detailed audit remain Git-ignored.

Peer round labels do not enter the owner's fixed-plan timeline. The model
comparison fits signed-log and magnitude targets separately, scores identical
official held-out targets and compares fits on a common candidate pool.
These self-reported peer returns do not change the confirmed owner incumbent
or query allowance.

## Function 2 Week 5 cohort comparison

The two notebooks in `notebooks/Function_2/Week_05/cohort/` use a separate local
`observations.csv`: all 14 official Week 5 rows, followed by five accepted
peer rows from the retained 20 October 2026 study. The adjacent `data-audit.json` records
the source commits and checksums, earlier exclusions, duplicate handling and the
new pooled CSV hash. Both files remain Git-ignored. The official CSV is unchanged.

The week-level `report.md` presents both analyses. The `official/` and `cohort/`
folders each contain `exploration-review.ipynb`, `model_predictions.ipynb`,
their own `observations.csv`, `figures/` and local `.coverage-cache/`.

The earlier study's Function 2 CSV contains 18 rows: 13 then-official rows and
five IDs beginning with `peer-`. Reconstruct the new 19-row snapshot by retaining
those five peer rows and combining them with the current official Week 5 CSV.
Verify that the earlier study's non-peer rows match the current CSV through
round 3, and preserve the `observation_id,source_round,x1,x2,y` schema and full
float64 precision. Do not append the shared starting rows again. The accepted
source files, pinned audit and original CSV are needed to verify provenance;
this comparison does not refresh repositories or fill pending results.

Peer round numbers are per-source labels, not an owner query timeline. The
cohort review's fixed-plan storyboard therefore uses only the official history.
The model comparison separately checks pooled LOO metrics, the same 14 official
target rows, and one common candidate pool. Peer records remain self-reported;
the comparison does not establish a new confirmed return or increase the query budget.

## Function 3 Week 5 cohort comparison

The two notebooks in `notebooks/Function_3/Week_05/cohort/` use a local
`observations.csv` with **24 rows**: the 19 official Week 5 rows followed
by five accepted peer records from the retained 20 October study. The companion
`data-audit.json` records commits, source checksums, previous exclusions,
duplicate handling and CSV hashes. Both files remain Git-ignored; the official
CSV is unchanged.

The week-level `report.md` presents the discussion and cohort comparison.
Both `official/` and `cohort/` contain `exploration-review.ipynb`,
`model_predictions.ipynb`, their own `observations.csv`, `figures/` and local
`.coverage-cache/`. Their data and model results were preserved during the move.

Reconstruct the snapshot by taking the five `peer-` rows from the earlier
Function 3 pooled CSV and appending them to the current official Week 5 CSV.
Verify the earlier non-peer rows against the current data through round 3.
Keep `observation_id,source_round,x1,x2,x3,y`, full float64 precision, and the
15 shared initial rows exactly once. Pending outputs and invalid coordinates
remain excluded; equal outputs at distinct inputs remain separate records.

The historical fixed-plan storyboard uses only the official timeline. Model
checks compare the same 19 official targets with and without peer training
rows, then score both GP policies on a common candidate pool. These are
retrospective checks on self-reported, adaptively sampled data; no portal
submission or extra owner query allowance follows from this comparison.

## Function 4 Week 5 cohort comparison

The notebooks in `notebooks/Function_4/Week_05/cohort/` use **40 rows**: all
34 official observations followed by the six accepted `peer-` rows from the
retained 20 October Function 4 study. The sibling `official/` folder contains
the unchanged official CSV, the observation review and a new Week 5 model run.
Both versions use `observation_review.ipynb`, `model_predictions.ipynb`,
`observations.csv`, `figures/` and a local `.coverage-cache/`. Their discussion
and comparison live in the week-level `report.md`.

Reconstruct the pooled snapshot from the earlier study's 39-row CSV: first
check that its 33 non-peer rows exactly match the current official history
through round 3, then append only its six peer rows to the current 34-row
official CSV. Preserve `observation_id,source_round,x1,x2,x3,x4,y` and full
float64 precision. Retain the shared initial batch once. Earlier duplicate
reports, pending returns and invalid coordinates stay excluded; distinct
valid inputs with equal reported outputs remain separate records.

The local `cohort/data-audit.json` retains source commits, checksums, exclusions
and snapshot hashes. All four pinned source files were checked against the
earlier audit without refreshing repositories. The pooled snapshot contains
one positive peer-reported output, 0.712146; values are not clipped to zero.
The CSVs, detailed audit and coverage caches remain Git-ignored.

Peer round labels do not enter the owner's historical progress or the saved
Week 4 forecast check. The model comparison scores identical official targets
with and without peer training rows, then compares the fits on common
candidates and both actual proposals. Neither proposal is a portal submission.

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

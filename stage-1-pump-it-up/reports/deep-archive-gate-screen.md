# Repair-preserving deep/archive gate screen

## Decision

Prepare, but do not upload, a selective gate between the submitted
seed-20260824 deep archive and the earlier 0.8288 archive synthesis. The fixed
gate adds **13 net correct rows** on leakage-safe development meta-predictions
and **3 net correct rows** on the used local test. It changes only 31
competition decisions and never restores an archive decision when the current
deep model predicts `functional needs repair`.

This is a plausible high-variance submission option, not a strong statistical
result. Exact paired sign tests are 0.353 on development and 0.736 locally.
The public deep/archive score difference says the deep model is better overall;
the gate relies on its component evidence to isolate the minority of rows where
the archive appears preferable.

## Bounded screen

The screen used only existing aligned probability caches. No model component
was retrained and no raw competition feature was introduced. The gate sees:

- the archive-to-deep hard transition;
- the shared five-component probability gap and hard-vote difference;
- the accepted and deep primary-model gaps;
- the identity-CatBoost gap; and
- each ensemble's two-class decision margin.

A regularised logistic classifier (`C=1`) was fitted only on disagreement rows
where one of the two recipes was correct. Its development estimate is itself
cross-fitted: each meta-prediction comes from a gate fitted on the other four
frozen folds. A fixed probability threshold of 0.50 chooses the archive. The
task-aligned constraint then retains every deep-model repair prediction.

The deliberately small alternatives were unpromising: single coarse
probability/vote thresholds were fold-unstable, decision stumps lost accuracy,
and stronger logistic shrinkage reduced the gain. The unconstrained logistic
gate added 12 rows over development but reduced repair recall, so it was not
promoted.

## Evidence

| Evidence | Deep accuracy | Gated accuracy | Net rows | Reversions | Archive-only / deep-only / third | Repair recall change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Five-fold development | 81.8981% | **81.9255%** | **+13** | 173 | 90 / 77 / 6 | **+0.1737 pp** |
| Used local test | 81.3973% | **81.4226%** | **+3** | 38 | 19 / 16 / 3 | **+0.1159 pp** |

Development fold nets are **-1, +1, +6, +6 and +1**. Functional recall also
improves by 0.2402 points on development and 0.1395 points locally;
non-functional recall falls by 0.3012 and 0.1533 points respectively.

## Competition candidate

The final gate is fitted to out-of-sample base predictions from development and
local-test rows, then applied to the full-data component caches. It restores 31
of the 226 archive/deep disagreements:

| Archive decision | Deep decision | Rows restored |
| --- | --- | ---: |
| `functional` | `non functional` | 26 |
| `non functional` | `functional` | 4 |
| `functional needs repair` | `non functional` | 1 |

The net prediction-count changes versus the 0.8298 incumbent are +22
`functional`, +1 `functional needs repair` and -23 `non functional`.

The 31 gate rows are disjoint from the 12-row strict repair candidate on all
three evidence sets. A simple union therefore adds the effects exactly:
**+29** development rows with fold nets +2, +4, +7, +10 and +6; **+8** local
rows; and 43 competition changes. That combined file is intentionally not
created here so ownership of the independently developed repair candidate
remains separate.

- File: `01-repair-preserving-archive-gate.csv`
- Rows: 14,850 unique IDs in template order
- SHA-256: `830d7a39e4b129df2ab67e4730323ba5b7363a7c9bf11e565474e720aca1f7b2`
- Status: prepared, not uploaded

Row-level cross-fitted, local and competition audits are written under
`.runtime/deep-archive-gate-screen/`. The submission manifest records all
selection counts and reconstruction checks.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Recover a small number of incumbent errors without broadly replacing the proven 0.8298 model. |
| 2. Gather the data | Reuse supplied labels and seven existing OOF, local and competition probability components. |
| 3. Explore the data | Restrict exploration to 804 development rows where the old and current ensembles disagree. |
| 4. Clean and preprocess the data | Validate component probability shapes, folds, IDs and exact reconstruction of both submitted CSVs. |
| 5. Select and engineer features | Derive only probability gaps, hard transitions, shared votes and ensemble margins. |
| 6. Define the machine-learning task | Binary selection of the archive decision on consequential disagreement rows. |
| 7. Partition the data | Cross-fit the gate over the five frozen development folds, then apply it once to the used local test. |
| 8. Select and train candidate methods | Use one regularised logistic gate and a fixed repair-preservation rule after a bounded coarse screen. |
| 9. Evaluate and interpret the results | Require positive aggregate, fold and local direction while reporting weak paired uncertainty. |
| 10. Deploy and iterate | Refit the unchanged gate on all out-of-sample labelled evidence and prepare a validated competition CSV. |

## Reproduction

Run from the project root:

```powershell
.\.venv\Scripts\python.exe .\stage-1-pump-it-up\scripts\run_deep_archive_gate_screen.py
.\.venv\Scripts\python.exe -m unittest .\stage-1-pump-it-up\tests\test_deep_archive_gate.py
```

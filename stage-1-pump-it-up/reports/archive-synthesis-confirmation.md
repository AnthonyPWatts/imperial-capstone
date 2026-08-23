# Archive-synthesis local confirmation

## Decision

**Leaderboard outcome, 23 August 2026:** the unchanged candidate scored
**0.8288**, establishing a new project best and exceeding the 0.826 target.

Place the fixed archive-derived synthesis first in tomorrow's submission order,
with the already-prepared complete-identity candidate second. The synthesis
reaches **81.1869%** on the previously used 11,880-row local test, **+0.1010
percentage points** and 12 net correct rows above the identity candidate's
81.0859%. It also led development accuracy at **81.7929%** and improved all
five frozen folds.

The six weights are unchanged from the algebraic average that produced the
development result. No component, tree count or weight was selected on the
local test. A validated full-labelled-data competition file was prepared but
not uploaded while the 22 August allowance remained 3/3. It was subsequently
submitted unchanged on 23 August.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Decide whether the strongest development-only synthesis deserves a second submission slot beside the confirmed identity candidate. |
| 2. Gather the data | Reuse frozen development/local-test partitions, cached accepted and identity probabilities, and the two archive-derived representation trials. |
| 3. Explore the data | Use only the already-fixed six-component recipe; do not inspect errors to create variants. |
| 4. Clean and preprocess the data | Reuse accepted preprocessing, ten-neighbour height reconstruction and all-categorical occurrence counts. |
| 5. Select and engineer features | Combine the three previously evaluated representations without adding fields. |
| 6. Define the machine-learning task | Unchanged three-class accuracy with recall and probability-quality diagnostics. |
| 7. Partition the data | Refit uncached components on 47,520 development rows and score the full 11,880-row local test once. |
| 8. Select and train candidate methods | Refit spatial XGBoost, spatial Random Forest and frequency Random Forest; reuse identically fitted accepted and identity components. |
| 9. Evaluate and interpret the results | Compare the fixed synthesis directly with the ready identity candidate and report paired uncertainty. |
| 10. Deploy and iterate | Refit all six components on 59,400 labels, validate the competition file and queue it first after the allowance resets. |

## Fixed recipe

| Component | Weight |
| --- | ---: |
| Accepted XGBoost | 33% |
| Spatial-height XGBoost | 11% |
| Accepted Random Forest | 18% |
| Categorical-frequency Random Forest | 9% |
| Spatial-height Random Forest | 9% |
| Complete-identity CatBoost | 20% |

These weights are the exact equal average of the previously fixed frequency-
forest and spatial-height representation votes. They were derived
algebraically before either the local-test confirmation or competition refit.

## Local-test results

| Metric | Ready identity | Archive synthesis | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 81.0859% | **81.1869%** | **+0.1010 pp** |
| Balanced accuracy | 66.6233% | **66.7792%** | +0.1560 pp |
| Macro F1 | 69.1655% | **69.4263%** | +0.2609 pp |
| Functional recall | 89.9411% | **90.1271%** | +0.1860 pp |
| Repair recall | 32.0973% | **32.4450%** | +0.3476 pp |
| Non-functional recall | **77.8313%** | 77.7656% | -0.0657 pp |
| Log loss | 0.463230 | **0.461905** | -0.001325 |
| Multiclass Brier | 0.266671 | **0.265755** | -0.000916 |

The hard predictions differ on 80 rows. The synthesis alone gets 40 correct
and the identity candidate alone gets 28, yielding 12 net correct. Exact
McNemar p is 0.182 and the paired row-bootstrap 95% interval for the accuracy
change is -0.0337 to +0.2357 points. The direction is consistent with all five
development folds, but the interval appropriately warns that this is still a
small difference on a repeatedly used local test.

## Prepared competition candidate

The validated file contains 14,850 unique IDs in template order, predicts only
the three allowed labels and has SHA-256
`46be08ced5e9a3ac3922a33e2aaf657306a08a406829cc69b4d25a5396e6ed0c`.
It disagrees with the identity candidate on 0.8215% of competition rows and
predicts 60.8822% functional, 3.5286% repair and 35.5892% non-functional.

Tomorrow's order is therefore:

1. archive-derived synthesis;
2. complete-identity 44:36:20 candidate;
3. hold the remaining slot until the first scores are observed.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_archive_synthesis_confirmation.py
..\.venv\Scripts\python.exe .\scripts\generate_archive_synthesis_submission.py
```

Runtime evidence remains ignored under `.runtime/archive-synthesis-confirmation/`
and `.runtime/archive-synthesis-competition/`.

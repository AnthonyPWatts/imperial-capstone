# CatBoost identity physical-backoff screen

## Decision

Retain the promoted complete-identity vote unchanged. Giving only its CatBoost
member all four deterministic physical parents as explicit native categorical
back-offs reaches **81.7466%**, merely **+0.0063 percentage points** or three
development rows. It wins **2/5** folds and the back-off CatBoost standalone is
0.091 points weaker than the complete-identity CatBoost.

The fixed all-parent hypothesis fails the +0.10-point gate. Do not mine parent
subsets. No local-test or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether deterministic physical parents provide useful back-off layers specifically for the identity specialist. |
| 2. Gather the data | Reuse extraction class, source class, quality group and waterpoint group from the raw labelled frame. |
| 3. Explore the data | Reuse completed hierarchy ablations and the source-plus-class public result. |
| 4. Clean and preprocess the data | Apply the same conservative categorical normalisation as the deferred identities. |
| 5. Select and engineer features | Add all four parents to the 35-field complete-identity CatBoost frame; do not change global components. |
| 6. Define the machine-learning task | Unchanged three-class classification. |
| 7. Partition the data | Preserve frozen outer folds and nested tree-count selection. |
| 8. Select and train candidate methods | Fit one depth-8 CatBoost and retain its fixed 20% contribution. |
| 9. Evaluate and interpret the results | Compare directly with the promoted complete-identity CatBoost and vote. |
| 10. Deploy and iterate | Reject the all-parent candidate and stop without subset tuning or local-test use. |

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Back-off identity vote | 81.7466% | +0.0063 pp | 2/5 | -0.0210 pp | -0.0290 pp | No |
| Promoted identity vote | **81.7403%** | — | — | — | — | incumbent |
| Complete-identity CatBoost | 80.6229% | -1.1174 pp | 0/5 | -1.5888 pp | -5.8489 pp | No |
| Back-off identity CatBoost | 80.5324% | -1.2079 pp | 0/5 | -1.6625 pp | -6.4857 pp | No |

The candidate has 39 engineered fields and 25 native categorical fields. It
selected 2,051, 1,818, 2,213, 2,299 and 2,376 trees. The tiny aggregate change
and weaker specialist give no evidence that a parent-subset screen would be
anything other than adaptive reuse of the same folds.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_catboost_identity_backoff_screen.py
```

Runtime evidence remains ignored under `.runtime/catboost-identity-backoff-screen/`.

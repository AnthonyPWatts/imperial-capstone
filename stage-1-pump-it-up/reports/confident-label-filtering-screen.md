# Nested confident-label filtering screen

## Decision

Stop model-based label-error filtering. Removing rows whose inner out-of-fold
XGBoost prediction gives at least 90% probability to a different class and no
more than 5% to the observed label lowers filtered XGBoost from 81.0143% to
**80.9070%**. Replacing the promoted vote's XGBoost member lowers the ensemble
from 81.7403% to **81.6835%**.

The candidate wins only **2/5** folds and reaches the -0.25-point worst-fold
guardrail. Do not tune disagreement thresholds. Every outer-validation row
remained in scoring, and none participated in removal decisions. No local-test
or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether only extremely confident model disagreements identify harmful training-label errors. |
| 2. Gather the data | Reuse each outer-training partition and its observed three-class labels. |
| 3. Explore the data | Build on earlier physical, duplicate and Isolation Forest filters, which did not test nested label disagreement. |
| 4. Clean and preprocess the data | Fit the accepted preprocessing separately inside each inner teacher fold and final outer refit. |
| 5. Select and engineer features | Add no features; remove only rows crossing both fixed probability conditions. |
| 6. Define the machine-learning task | Unchanged supervised three-class classification. |
| 7. Partition the data | Generate removal evidence with three inner OOF folds inside every outer-training fold; keep all validation rows. |
| 8. Select and train candidate methods | Fit accepted XGBoost teachers and one filtered student per outer fold. |
| 9. Evaluate and interpret the results | Compare filtered XGBoost and its unchanged 44:36:20 vote with the promoted recipe. |
| 10. Deploy and iterate | Reject the filter and stop outlier removal rather than tune thresholds. |

## Removal audit

| Fold | Training rows removed | Fraction | Observed functional | Observed repair | Observed non-functional |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 549 | 1.444% | 102 | 239 | 208 |
| 2 | 487 | 1.281% | 100 | 207 | 180 |
| 3 | 504 | 1.326% | 91 | 200 | 213 |
| 4 | 424 | 1.115% | 96 | 176 | 152 |
| 5 | 434 | 1.142% | 79 | 198 | 157 |

Repair rows are heavily over-represented among removals. The alternative
prediction is almost always functional or non-functional: only 4–7 removed
rows per fold are predicted as repair. This is consistent with the minority
class being intrinsically difficult, not proof that its labels are wrong.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Promoted identity vote | **81.7403%** | — | — | — | — | incumbent |
| Filtered identity vote | 81.6835% | -0.0568 pp | 2/5 | -0.2525 pp | -0.3187 pp | No |
| Accepted XGBoost | 81.0143% | -0.7260 pp | 0/5 | -0.8733 pp | -2.4034 pp | No |
| Filtered XGBoost | 80.9070% | -0.8333 pp | 0/5 | -1.0943 pp | -2.6351 pp | No |

Nested provenance removes the leakage risk that would arise from using global
OOF predictions to filter outer-training rows. It does not rescue the idea:
extreme disagreement still identifies valid hard examples often enough that
discarding them reduces generalisation.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_confident_label_filtering_screen.py
```

Runtime evidence remains ignored under `.runtime/confident-label-filtering-screen/`.

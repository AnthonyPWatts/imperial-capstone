# CatBoost deferred-identity confirmation

## Decision

Promote the frozen 44% XGBoost, 36% Random Forest and 20% complete-identity
CatBoost recipe to a full-labelled-data competition refit. On the 11,880-row
reserved local test it reaches **81.086%** accuracy, **+0.387 percentage points**
above the identically refitted accepted 55:45 ensemble. It converts 118 rows
that only the candidate gets right while losing 72 rows that only the accepted
recipe gets right: a net **46 additional correct classifications**.

This result confirms rather than selects the recipe. No fields, settings,
weights or tree counts were changed after the frozen-fold screen. The local test
has been used by earlier workflows, so it is no longer a pristine final estimate.
No competition prediction or upload was made at this checkpoint.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Confirm whether the preselected identity-CatBoost contribution transfers beyond the development folds. |
| 2. Gather the data | Reuse the frozen development and 11,880-row local-test partitions. |
| 3. Explore the data | Use only already-recorded fold evidence; do not inspect local-test outcomes before fitting. |
| 4. Clean and preprocess the data | Reuse the accepted fold-fitted preprocessing and conservative identity normalisation. |
| 5. Select and engineer features | Keep the accepted 29 features for XGBoost and Random Forest, plus the frozen six native identity categories for CatBoost. |
| 6. Define the machine-learning task | Unchanged supervised three-class classification scored by accuracy. |
| 7. Partition the data | Fit all components on 47,520 development rows and score all 11,880 local-test rows once. |
| 8. Select and train candidate methods | Use the preselected specifications, 1,014 XGBoost trees, 2,173 CatBoost trees and unchanged Random Forest. |
| 9. Evaluate and interpret the results | Compare the frozen 55:45 baseline with the frozen 44:36:20 candidate, including paired row changes and probability quality. |
| 10. Deploy and iterate | Promote to a full-labelled-data refit and validated competition file, but do not upload after the daily submission allowance has been used. |

## Results

| Metric | Accepted 55:45 | Identity candidate 44:36:20 | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 80.699% | **81.086%** | **+0.387 pp** |
| Balanced accuracy | 66.586% | 66.623% | +0.037 pp |
| Macro F1 | 68.905% | 69.166% | +0.261 pp |
| Functional recall | 89.135% | 89.941% | +0.806 pp |
| Repair recall | 32.793% | 32.097% | -0.695 pp |
| Non-functional recall | 77.831% | 77.831% | 0.000 pp |
| Log loss | 0.467882 | **0.463230** | **-0.004652** |
| Multiclass Brier | 0.269341 | **0.266671** | **-0.002670** |

The two hard predictions differ on 209 rows. Among the 190 rows where their
correctness differs, the candidate wins 118 to 72. An exact paired McNemar test
gives p = 0.00104. A deterministic 10,000-resample paired row bootstrap places
the 95% interval for the accuracy change at **+0.160 to +0.623 points**. These
are secondary uncertainty descriptions, not additional model-selection gates.

The gain is concentrated in functional recall. Repair recall falls by 0.695
points and non-functional recall is unchanged. That repair change remains
inside the predeclared two-point guardrail, but it is a real trade-off rather
than a free improvement.

## Full-data candidate

The frozen components were refitted or reused from their already-validated
full-data cache over all 59,400 labelled rows. The resulting 14,850-row
competition CSV has unique IDs in template order, valid labels and SHA-256
`76cf24053f136d89262b23b626c932e2f5b609ca197e16d2cb7e4b46897bb006`.
It disagrees with the accepted submission on 1.576% of rows and predicts 60.788%
functional, 3.623% repair and 35.589% non-functional. These shares were recorded,
not used to retune the recipe. The file remains unuploaded while the recorded
22 August allowance is 3/3.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_catboost_identity_confirmation.py
```

The first completed run is cached and subsequent runs verify and replay that
same evidence under `.runtime/catboost-identity-confirmation/`.

# Deep occurrence-support XGBoost screen

## Decision

Retain the top-common-identity archived deep XGBoost without occurrence-count
features. Adding fold-fitted category support for all 29 categorical fields
reduces the component from **81.4710%** to **81.3047%**. Replacing the deep
component in the fixed archive synthesis reaches **81.7866%**, 0.0063
percentage points below the archive incumbent.

The substitution wins three folds, its worst fold loses 0.1578 points and
repair recall increases by 0.8391 points. It fails the fixed accuracy gate, so
the local test remains closed and no competition predictions are generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether category occurrence support complements the useful top-common identity representation. |
| 2. Gather the data | Reuse only supplied predictors, frozen development rows and cached archive components. |
| 3. Explore the data | Combine two previously useful archive signals without opening a new feature or weight grid. |
| 4. Clean and preprocess the data | Fit category normalisation and support counts only on each outer-training partition; map support below five to the same rare marker. |
| 5. Select and engineer features | Add one occurrence-count feature for each of the 29 categorical fields to the accepted and top-50 identity representation. |
| 6. Define the machine-learning task | Retain unchanged three-class probabilistic classification and accuracy. |
| 7. Partition the data | Reuse the five frozen stratified development folds; fit every transformer within its training fold. |
| 8. Select and train candidate methods | Fit one predeclared seed of the archived depth-17, 600-tree XGBoost specification. |
| 9. Evaluate and interpret the results | Compare the component and one fixed archive substitution with their occurrence-free counterparts. |
| 10. Deploy and iterate | Stop before local or competition use because the candidate weakens both component and ensemble accuracy. |

## Results

| Candidate | Accuracy | Change vs archive | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing archived deep substitution | **81.8771%** | +0.0842 pp | 3/5 | -0.0210 pp | +0.6945 pp |
| Existing archive synthesis | **81.7929%** | — | — | — | — |
| Occurrence-support deep substitution | 81.7866% | -0.0063 pp | 3/5 | -0.1578 pp | +0.8391 pp |
| Archived deep XGBoost | **81.4710%** | — | — | — | — |
| Occurrence-support deep XGBoost | 81.3047% | — | — | — | — |

The occurrence component disagrees with the occurrence-free deep model on
4.3245% of development rows. The original is uniquely correct on 2.0476%,
versus 1.8813% for occurrence support. The added signal changes decisions, but
more of those changes are wrong.

This closes the occurrence-support/deep-XGBoost cross. The result does not
justify support thresholds, per-field selection, transformations or blend
weights on the repeatedly used folds.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_deep_occurrence_xgboost_screen.py
```

Runtime evidence remains ignored under
`.runtime/deep-occurrence-xgboost-screen/` at project level.

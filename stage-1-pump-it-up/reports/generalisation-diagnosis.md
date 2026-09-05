# Generalisation diagnosis

## Conclusion

The evidence supports the **base modelling path**, but not the late
row-targeted postprocessing path. Across eight formal base candidates, internal
development ordering tracked public score closely (Spearman rank correlation
**1.000**; Pearson correlation **0.934**). A new, fixed five-fold diagnosis of
the incumbent deep top-50 XGBoost component found a moderately harder
historical local subset, but the uncertainty interval narrowly included no
difference and every class moved in the same direction.

By contrast, all three 4 September micro-postprocessing submissions improved
the repeatedly used development and local evidence but scored below the
**0.8298** incumbent. That pattern is more consistent with adaptive
postprocessing overfit than with a generally wrong model family or a broken
competition test set.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether disappointing late submissions indicate a failed base path, a materially different subset, or overfitting to reused evidence. The diagnosis informs further model-level experiments; it does not select competition rows. |
| 2. Gather the data | Use the 59,400 supplied labelled rows, the supplied 14,850-row competition predictors, frozen submission records and aggregate public scores. |
| 3. Explore the data | Reconstruct scored submissions, compare internal and public ordering, and inspect performance by membership of the historical local subset. |
| 4. Clean and preprocess the data | Reuse the validated modelling handoff and fold-fitted accepted plus top-50 identity preprocessing. Validate complete OOF probabilities, IDs and fold caches. |
| 5. Select and engineer features | Make no new selection. Freeze the archived accepted-feature plus six top-50 deferred-identity representation. |
| 6. Define the machine-learning task | Preserve three-class classification by accuracy, with class recall retained as a diagnostic for the rare repair class. |
| 7. Partition the data | Assign all 59,400 rows to five fresh stratified folds with seed `20260905`. Historical local membership is attached only after prediction for subgroup reporting. |
| 8. Select and train candidate methods | Fit the archived depth-17 XGBoost specification for exactly 600 trees in each fold. Perform no early stopping, hyperparameter search or candidate selection. |
| 9. Evaluate and interpret the results | Report full OOF accuracy, Wilson intervals, subgroup accuracy and recall, plus a class-stratified binomial-count bootstrap for the subgroup contrast. |
| 10. Deploy and iterate | Cache reproducible diagnostic evidence only. Prefer fresh, predeclared model-level candidates; stop selecting tiny row masks from reused folds, local labels or leaderboard movements. |

## Evidence of transfer and overfit

The repository's train-versus-competition audit found little covariate shift:

| Audit | Labelled/random-local reference | Competition | Largest discrepancy |
| --- | ---: | ---: | ---: |
| Numeric distributions | — | — | Maximum KS statistic **0.0112** |
| Categorical distributions | — | — | Maximum total-variation distance **0.0233** |
| Structurally missing cells | 1.94% | 1.93% | 0.01 pp |
| Rows with structural missingness | 53.18% | 53.24% | 0.06 pp |
| `wpt_name` identity coverage | 43.70% | 43.10% | -0.60 pp |
| LGA plus ward identity coverage | 99.88% | 99.92% | +0.04 pp |
| Exact-coordinate identity coverage | 3.30% | 3.31% | +0.01 pp |
| `scheme_name` identity coverage | 98.44% | 98.74% | +0.30 pp |

The incumbent's prediction distributions are similarly stable. Mean confidence
is **80.925%** on both samples; mean top-two margin is **65.147%** on the local
reference and **65.174%** on competition rows; normalised entropy is **42.325%**
and **42.108%** respectively. The three constant-class public submissions imply
public-scored class shares near the supplied label priors: functional
**54.61% versus 54.31%**, repair **7.19% versus 7.27%**, and non-functional
**38.20% versus 38.42%**. These aggregate diagnostics do not prove identical
conditional label distributions, but they provide no evidence of a large
test-population shift.

The submission audit reconstructed **18 scored submissions**. For the eight
formal base candidates with comparable internal evidence, higher development
accuracy always implied a higher public score (Spearman **1.000**), and the
linear association was strong (Pearson **0.934**). The sample is small and the
submission history is not an independent experiment, but it provides no sign
of a broad validation-to-competition rank reversal.

The late micro-overrides tell a different story:

| Submitted override | Reused OOF net | Used-local net | Public score | Change from 0.8298 |
| --- | ---: | ---: | ---: | ---: |
| 22-row repair union | +23 | +8 | 0.8296 | -0.0002 |
| 43-row gate plus strict repair | +29 | +8 | 0.8293 | -0.0005 |
| 16-row strict, meta and consensus repair | +21 | +7 | 0.8294 | -0.0004 |

Every override looked positive on evidence that had already influenced the
search, yet every public result regressed. The exact number of public errors
cannot be recovered because the locally recorded metric wording does not
disclose the public leaderboard denominator. The direction of all three
results is nevertheless sufficient to reject this micro-postprocessing regime.

## Fresh full-label OOF diagnosis

The diagnostic used seed `20260905`, five folds, all **59,400** labelled rows
and the fixed archived depth-17, 600-tree top-50 XGBoost component. Each row was
predicted only by a model that excluded that row. The historical split did not
affect fold construction, fitting or selection; it was a post-hoc ID tag.

| Group | Correct / rows | Accuracy | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| All labelled rows | 48,461 / 59,400 | **81.5842%** | 81.2704–81.8938% |
| Historical local IDs | 9,628 / 11,880 | **81.0438%** | 80.3290–81.7385% |
| All other IDs | 38,833 / 47,520 | **81.7193%** | 81.3692–82.0642% |

Historical-local accuracy was **0.6755 percentage points lower**. A 20,000-draw
class-stratified parametric binomial-count bootstrap gave a 95% interval of
**-1.4078 to +0.0463 percentage points** for historical-local minus other
accuracy. The groups are disjoint, so this is explicitly an unpaired contrast
at each group's observed class composition, not a paired test.

| Group | Class | Actual rows | Recall |
| --- | --- | ---: | ---: |
| All labelled | Functional | 32,259 | 90.4337% |
| All labelled | Functional needs repair | 4,317 | 32.9164% |
| All labelled | Non-functional | 22,824 | 78.2816% |
| Historical local | Functional | 6,452 | 89.9256% |
| Historical local | Functional needs repair | 863 | 32.2132% |
| Historical local | Non-functional | 4,565 | 77.7218% |
| All other IDs | Functional | 25,807 | 90.5607% |
| All other IDs | Functional needs repair | 3,454 | 33.0921% |
| All other IDs | Non-functional | 18,259 | 78.4216% |

The historical subset is slightly harder across all three classes, but the
overall interval narrowly includes no difference and there is no class-specific
collapse. This does not support a claim that the earlier local split was
fundamentally misleading. The persistent weakness is repair recall, which is a
model limitation shared by both subgroups rather than evidence of a new test
population.

## No-cheating boundary

This investigation uses only the supplied training values and labels, supplied
competition predictors, frozen local artefacts and aggregate leaderboard
outcomes. The [local competition reference](../instructions/competition-reference.md#rules-relevant-to-the-project)
records that external data is prohibited. Aggregate scores may confirm or
reject a preselected model, but they must not be converted into assumed
row-level outcomes, used for row-wise leaderboard probing, or treated as hidden
labels. No competition row was selected or changed by this diagnosis.

## Reproduction and retained evidence

Run from the project root:

```powershell
.\.venv\Scripts\python.exe .\stage-1-pump-it-up\scripts\run_generalisation_diagnosis.py
.\.venv\Scripts\python.exe -m unittest .\stage-1-pump-it-up\tests\test_generalisation_diagnosis.py -v
```

The first run writes fold caches, OOF probabilities, accuracy and recall tables,
fold timings and `result.json` under `.runtime/generalisation-diagnosis/`.
Subsequent runs validate and reuse compatible fold caches. `.runtime/` is
ignored by Git, so this evidence remains reproducible local working data rather
than committed competition data.

# Transductive categorical-frequency forest screen

## Decision

Retain the inductive archive-derived synthesis and the promoted competition
candidate. Counting categorical support across the fixed development and
supplied competition covariates strengthens the compact frequency forest from
80.6103% to **80.8270%**, a **+0.2167 percentage-point** standalone gain.
However, its best fixed ensemble reaches only **81.7824%**, and substituting it
into the archive synthesis lowers that vote from 81.7929% to **81.7740%**.

No mix of inductive and transductive forests, count transform, threshold or
blend weight is screened. No local-test or competition prediction was
generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Reproduce the archived solution's target-free use of the supplied labelled and inference covariates for categorical support. |
| 2. Gather the data | Combine frozen development predictors with the supplied competition predictor frame; do not use local-test rows or any additional data. |
| 3. Explore the data | Measure how often fixed-universe support moves a validation category across the predeclared rare boundary. |
| 4. Clean and preprocess the data | Reuse identity normalisation, accepted numeric cleaning and the fixed support-below-five sentinel. |
| 5. Select and engineer features | Replace all 29 categorical/date fields with counts learned from the 62,370-row development-plus-competition feature universe. |
| 6. Define the machine-learning task | Unchanged three-class classification and accuracy metric. |
| 7. Partition the data | Keep labels confined to each outer-training model fit; validation and competition covariates influence counts only. |
| 8. Select and train candidate methods | Refit one accepted-spec Random Forest and reuse fixed ensemble weights. |
| 9. Evaluate and interpret the results | Compare inductive and transductive forests, their fixed representation votes and archive syntheses. |
| 10. Deploy and iterate | Reject ensemble promotion and close transductive count variants despite the standalone gain. |

## Transductive contract

The competition predictor frame is part of the supplied competition data, not
an external source. The count mapping uses no labels. For every outer fold:

- the Random Forest sees target labels only for the outer-training rows;
- count support is fixed from all development predictors plus the unlabelled
  competition predictors;
- the untouched local-test rows are absent from both counts and model fitting;
- the original outer-validation rows remain unchanged for scoring.

This protocol asks a transductive question: can the known inference covariate
universe improve support estimates for this fixed competition? It is not a
claim about generalisation to future, unseen batches.

Across the five folds, 27.94–29.59% of validation rows contain at least one
field whose outer-training count is below five but whose fixed-universe count
is at least five. The representation therefore changes materially at the
archived rare-support boundary.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Inductive archive synthesis | **81.7929%** | +0.0526 pp | 5/5 | +0.0105 pp | -0.2026 pp | No |
| Transductive frequency representation vote | 81.7824% | +0.0421 pp | 4/5 | -0.0421 pp | -0.6660 pp | No |
| Transductive archive synthesis | 81.7740% | +0.0337 pp | 4/5 | -0.0631 pp | -0.2605 pp | No |
| Promoted complete-identity vote | 81.7403% | — | — | — | — | incumbent |
| Transductive frequency Random Forest | 80.8270% | — | — | — | — | diagnostic |
| Inductive frequency Random Forest | 80.6103% | — | — | — | — | diagnostic |

The stronger standalone forest confirms that competition covariates improve
category-support estimation. Its errors overlap the promoted components in a
less useful way than the inductive forest's errors: direct transductive voting
gains 0.042 points, while the more consistent inductive synthesis gains 0.053
points on every fold. More voter mixing would be an adaptive blend search, not
new evidence, so the branch stops.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_transductive_frequency_forest_screen.py
```

Runtime evidence remains ignored under
`.runtime/transductive-frequency-forest-screen/`.

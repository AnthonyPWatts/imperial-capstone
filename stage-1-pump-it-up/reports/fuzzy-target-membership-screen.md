# Fuzzy target-membership screen

## Conclusion

Ordinally adjacent fuzzy class memberships do not improve the accepted hard-
label model. The closest fuzzy-derived candidate uses 2.5% raw adjacent overlap
for XGBoost only and retains the hard-label Random Forest. It reaches
**81.534%** development accuracy, 0.090 percentage points below the accepted
81.625%, and loses all five folds.

Fuzzifying both components is weaker. The best fully fuzzy vote uses 5% overlap
and reaches **81.397%**, 0.227 points below baseline with zero fold wins. It
gains 22 correct repair decisions and 84 non-functional decisions but loses 214
functional decisions: a net loss of 108 rows.

No candidate passed the established promotion gate. The labelled local test was
not refitted or rescored, and no competition prediction or submission was
generated.

## Course-aligned lifecycle

| Step | Application in this experiment |
| --- | --- |
| 1. Define the goal and scope | Test whether overlapping condition memberships handle an ambiguous repair class better than crisp labels under ordinary three-class accuracy. |
| 2. Gather the data | Reuse the validated labelled modelling data and accepted out-of-fold baseline. Competition rows do not participate in fitting or evaluation. |
| 3. Explore the data | Quantify how each overlap changes effective class membership, crisp accuracy, class recall, confusion and probability quality. |
| 4. Clean and preprocess the data | Reuse the established fold-fitted feature engineering, imputation, rare grouping and one-hot encoding. Fit preprocessing on original source rows before weighted expansion. |
| 5. Select and engineer features | Hold the accepted feature policy fixed so that the target representation is the only modelling change. |
| 6. Define the machine-learning task | Train with triangular, row-normalised fuzzy memberships; defuzzify by maximum probability and score against unchanged crisp three-class labels. |
| 7. Partition the data | Reuse the frozen five development folds. Fuzzy expansion happens only inside the current outer-training partition; inner stopping and outer validation retain crisp labels. |
| 8. Select and train candidate methods | Refit accepted-spec XGBoost and Random Forest at four predeclared overlaps, reconstruct the 55:45 vote, then cross one fuzzy component at a time with its hard counterpart. |
| 9. Evaluate and interpret the results | Apply the established accuracy, fold, worst-fold and repair-recall gate; also inspect log loss, Brier score and correct-decision changes. |
| 10. Deploy and iterate | Not applicable: no fuzzy-derived candidate passes. Retain hard-label training and stop this overlap range. |

## What “fuzzy membership” means here

The three labels use the tentative order:

$$
\text{functional} \rightarrow \text{functional needs repair}
\rightarrow \text{non functional}.
$$

For raw adjacent overlap $\lambda$, an observed class receives membership one
and each immediate neighbour receives membership $\lambda$. Each row is then
normalised to total membership one. At $\lambda=0.05$, for example:

| Observed label | Functional membership | Repair membership | Non-functional membership |
| --- | ---: | ---: | ---: |
| Functional | 0.9524 | 0.0476 | 0 |
| Functional needs repair | 0.0455 | 0.9091 | 0.0455 |
| Non-functional | 0 | 0.0476 | 0.9524 |

Functional and non-functional are not direct neighbours and never receive
membership from each other. This is an ordinally informed soft-target
experiment, not a fuzzy rule engine or a claim that the target is truly
continuous.

The soft matrix is expressed as weighted labelled observations because the
selected classifiers require hard class indices. A source row contributes once
to every class with non-zero membership, using its membership as sample weight.
The weights for each source row sum exactly to one. Preprocessing is fitted on
the original rows before expansion, so rows with more memberships cannot distort
imputation or category-frequency estimates.

For multinomial XGBoost, the weighted expansion represents the corresponding
soft cross-entropy target. For bootstrapped Random Forests it is a practical
weighted approximation: expected class mass is preserved, although duplicate
memberships can alter bootstrap variance.

## Bounded membership policies

The four overlap widths were declared before results were compared. They span a
small perturbation through a deliberately strong treatment without creating a
post-result micro-grid.

| Raw adjacent overlap | Effective functional share | Effective repair share | Effective non-functional share |
| ---: | ---: | ---: | ---: |
| Hard labels | 54.308% | 7.269% | 38.424% |
| 0.010 | 53.841% | 8.044% | 38.115% |
| 0.025 | 53.156% | 9.184% | 37.660% |
| 0.050 | 52.052% | 11.024% | 36.925% |
| 0.100 | 49.976% | 14.487% | 35.536% |

Even apparently small adjacent memberships materially increase effective repair
mass because the two outer classes contain far more rows than the repair class.
This is why the width must be interpreted through the effective shares rather
than by $\lambda$ alone.

## Component results

| Training target | XGBoost accuracy | XGBoost repair recall | Random Forest accuracy | Random Forest repair recall |
| --- | ---: | ---: | ---: | ---: |
| Hard labels | 81.014% | 31.529% | **80.591%** | 36.856% |
| Fuzzy 0.010 | 81.054% | 32.803% | 79.836% | **37.667%** |
| Fuzzy 0.025 | 81.086% | 32.890% | 79.893% | 37.493% |
| Fuzzy 0.050 | 81.088% | **33.237%** | **79.916%** | 36.885% |
| Fuzzy 0.100 | **81.164%** | 33.180% | 79.798% | 36.509% |

Fuzzy XGBoost contains a small genuine signal: its strongest point improves the
hard XGBoost component by 0.149 points. Fuzzy Random Forest moves the other way;
even its best accuracy is 0.676 points below the hard forest. This justified a
fixed no-refit component cross rather than assuming both voters should use the
same target treatment.

## Fully fuzzy 55:45 votes

The accepted component weights remain fixed at 55% XGBoost and 45% Random
Forest.

| Overlap | Accuracy | Change | Fold wins | Worst fold | Repair recall | Repair change | Log loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.050 | **81.397%** | -0.227 pp | 0/5 | -0.379 pp | 35.496% | +0.637 pp | 0.48969 |
| 0.100 | 81.378% | -0.246 pp | 0/5 | -0.442 pp | 35.380% | +0.521 pp | 0.51964 |
| 0.025 | 81.370% | -0.255 pp | 0/5 | -0.400 pp | **35.872%** | **+1.013 pp** | 0.47741 |
| 0.010 | 81.315% | -0.309 pp | 0/5 | -0.505 pp | 35.814% | +0.955 pp | 0.47238 |
| Hard baseline | **81.625%** | — | — | — | 34.859% | — | **0.46851** |

All four fuzzy votes trade accuracy and probability quality for a modest repair-
recall improvement. The relationship is not monotonic: wider overlap increases
effective repair mass sharply, but repair recall peaks at 2.5% and then falls.
Target ambiguity is therefore not solved by assigning ever more neighbouring
membership.

For the best fully fuzzy 5% vote, the crisp confusion change from baseline is:

| Actual class | Change in correct decisions |
| --- | ---: |
| Functional | -214 |
| Functional needs repair | +22 |
| Non-functional | +84 |
| **Net** | **-108** |

The treatment redirects some decisions from the majority functional class to
both other classes. Those gains are real, but ordinary accuracy values every
row equally and the functional loss is larger.

## One-fuzzy-component crosses

| Fuzzy component and overlap | Accuracy | Change | Fold wins | Worst fold | Repair recall | Log loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fuzzy XGBoost 0.025 + hard forest | **81.534%** | **-0.090 pp** | 0/5 | -0.137 pp | 35.061% | 0.47127 |
| Fuzzy XGBoost 0.100 + hard forest | 81.528% | -0.097 pp | 1/5 | -0.221 pp | **35.611%** | 0.49082 |
| Fuzzy XGBoost 0.050 + hard forest | 81.511% | -0.114 pp | 1/5 | -0.295 pp | 35.235% | 0.47685 |
| Fuzzy XGBoost 0.010 + hard forest | 81.496% | -0.128 pp | 0/5 | -0.189 pp | 35.033% | 0.46939 |
| Hard XGBoost + fuzzy forest, best accuracy | 81.473% | -0.152 pp | 1/5 | -0.421 pp | 34.859% | 0.48532 |
| Hard baseline | **81.625%** | — | — | — | 34.859% | **0.46851** |

The closest cross loses only 43 net rows: 68 fewer functional decisions are
correct, offset by seven more repair and 18 more non-functional decisions. It
still loses every fold and worsens log loss, so the fuzzy XGBoost improvement is
not sufficiently complementary to the accepted hard Random Forest.

## Decision

Retain crisp three-class labels and the accepted hard-label 55% XGBoost / 45%
Random Forest vote. The experiment supports two narrower observations:

- the repair label does share useful statistical structure with its proposed
  neighbours, because fuzzy XGBoost improves as a standalone component;
- uniform triangular overlap is too blunt for the ensemble and primarily moves
  errors between classes rather than adding correct decisions.

Stop this uniform-overlap loop. A materially different future experiment would
need row-specific uncertainty—derived from duplicate inspections, annotator
agreement, temporal condition changes or another defensible evidence source—
rather than another global value of $\lambda$ selected on these same folds.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_fuzzy_target_screen.py --force
```

Omit `--force` to reuse the ignored per-policy out-of-fold caches. The forced
run is deliberately expensive because it refits both model components across
all five folds for every membership width. Stable logic lives in
`src/fuzzy_target_evaluation.py`.

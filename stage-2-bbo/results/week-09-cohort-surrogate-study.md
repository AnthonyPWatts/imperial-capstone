# Week 9 cohort surrogate study

The clearest gain from adding model families was on **Function 7**. The wider
selection reduced mean outer-fold RMSE from 0.316 for GP-only selection to
0.238, choosing RBF kernel ridge in nine of ten folds. On the eight later
official returns, the same policy had higher error. **Function 6** improved
with the Matérn GP alone. On **Function 5** the blend had the lowest mean
outer error of the four policies. The GPs ranked the best measured choice
more reliably.

The 23 November 2026 snapshot contains 239 official measurements through round
8 and 217 self-reported peer measurements. All experiments fit on the pooled
cohort and score on held-out official measurements. Each observation notebook
compares coordinate distributions, score distributions, portal repeats and
distances across sources. Functions 7 and 8 have six and eight input
coordinates, respectively. Peers extend coverage towards the portal
boundaries in Functions 5–8. These differences may reflect the inputs each
group chose to evaluate. [Week 9 cohort review](week-09-cohort-review.md)

## Course techniques tested

The Module 15.1 discussion proposed SVMs, small neural networks and a binary
good/bad framing. Module 18 asked how to tune under a small data budget. The
experiments use a small declared grid and isolate model and blend selection
inside validation folds. The Module 19 prompt-engineering discussion also
included peer reflections on how they chose GP length scales.
[Module 15.1](https://classroom.emeritus.org/courses/18642/discussion_topics/969759) ·
[Module 18](https://classroom.emeritus.org/courses/18642/discussion_topics/969742?module_item_id=3237569) ·
[Module 19](https://classroom.emeritus.org/courses/18642/discussion_topics/969741?module_item_id=3237612)

| Course topic | Comparison |
| --- | --- |
| Preparation, features and bias–variance | Compare a pooled mean, ridge, quadratic ridge and ridge on standardised principal components retaining 95% of training variance. |
| Nearest neighbours and SVMs | Distance-weighted 5-NN, RBF kernel ridge and RBF SVR with three regularisation settings. A matched SVR pair changes only the coordinate scaler, from standard deviation to interquartile range. |
| Trees and ensembles | A depth-3 regression tree, random forest, shallow gradient boosting, a three-seed tanh network and fixed SVR scaling and GP–SVR–MLP blends. |
| Naïve Bayes and logistic regression | Estimate the probability of a top-quartile score. Use the training means of the high and low groups to convert the probability to a numerical prediction. |
| Bayesian optimisation and target preparation | Compare the existing GP families, Matérn 3/2, greater noise for peer rows, median scores at repeated inputs and a signed asinh target transform. Matérn 3/2 allows less smooth responses than the saved RBF kernel. |
| Unsupervised preprocessing | Fit K-means to the training coordinates and use a separate regularised regression in each of two regions. |

The study first scores 24–27 fixed candidates per function with grouped
leave-one-out. Each test input withholds all official and peer rows at that
six-decimal coordinate. Inner folds also continue to withhold any peer rows
matching the outer test points. Fitted transformations and duplicate
aggregation use training rows only. Function 1 uses MAE in signed-log height
with scale `1e-200`. Functions 2–8 use RMSE in the original score units.

| Function | Official + peer rows | Saved GP error | Best fixed candidate | Candidate error |
| --- | ---: | ---: | --- | ---: |
| [1](../notebooks/Function_1/Week_09/full-cohort/model_predictions.ipynb) | 18 + 29 | 92.436 | 5-NN distance weighting | 69.506 |
| [2](../notebooks/Function_2/Week_09/full-cohort/model_predictions.ipynb) | 18 + 27 | 0.12662 | Matérn 3/2 GP | 0.11402 |
| [3](../notebooks/Function_3/Week_09/full-cohort/model_predictions.ipynb) | 23 + 28 | 0.01945 | GP with duplicate medians | 0.01925 |
| [4](../notebooks/Function_4/Week_09/full-cohort/model_predictions.ipynb) | 38 + 27 | 1.3056 | GP with higher peer noise | 1.1073 |
| [5](../notebooks/Function_5/Week_09/full-cohort/model_predictions.ipynb) | 28 + 24 | 348.31 | Matérn 3/2 GP | 218.00 |
| [6](../notebooks/Function_6/Week_09/full-cohort/model_predictions.ipynb) | 28 + 28 | 0.20047 | Matérn 3/2 GP | 0.14958 |
| [7](../notebooks/Function_7/Week_09/full-cohort/model_predictions.ipynb) | 38 + 28 | 0.39277 | RBF kernel ridge | 0.22158 |
| [8](../notebooks/Function_8/Week_09/full-cohort/model_predictions.ipynb) | 48 + 26 | 0.28914 | Matérn 3/2 GP | 0.27489 |

## Model choice and blending

Two five-fold repeats with seeds 42 and 73 evaluate the selection procedure.
Four inner folds choose either a GP variant or any candidate in the full
bank. For the GP blend, inner predictions choose a GP, a non-GP partner and
a GP weight of 0.25, 0.50 or 0.75. The saved GP family is refitted on the
same outer training rows. Each table entry averages the two outer errors.
An asterisk marks an improvement over the saved GP on **both** splits.

| Function | Saved GP | Nested GP choice | Nested full-bank choice | Nested GP blend |
| --- | ---: | ---: | ---: | ---: |
| 1 | 92.329 | 96.929 | 93.102 | 92.712 |
| 2 | 0.13082 | 0.13637 | 0.15023 | 0.13380 |
| 3 | 0.04307 | 0.04334 | 0.04628 | 0.04258 |
| 4 | 1.3134 | 1.4478 | 1.4478 | 1.5414 |
| 5 | 363.34 | 317.15* | 349.95* | 307.86* |
| 6 | 0.19105 | 0.14485* | 0.17364* | 0.14593* |
| 7 | 0.44729 | 0.31633* | 0.23767* | 0.24892* |
| 8 | 0.32692 | 0.31521 | 0.32920 | 0.32127* |

**Preprocessing.** The fixed GP with median duplicates shaved about 1% off
Function 3 error. Higher peer noise, modelled as three times the observation
noise standard deviation, helped most on Function 4 in the fixed screen. On
outer folds Function 4's GP selector chose Matérn in eight folds and higher
peer noise once. Relative to the saved GP, the selector reduced the mean gap
to the best measured choice from 0.226 to 0.097 while raising RMSE. The matched SVR
scaling blend beat both component scalers only on Function 4 and its
leave-one-out error was 2.425 versus 1.306 for the saved GP. PCA ridge,
K-means local ridge and the classifier-derived score predictions all
trailed the GP in the full leave-one-out screens.

**Ensembles.** The fixed GP–SVR–MLP average beat the saved GP in four of eight
screens. In nested validation the blend improved on the saved GP on both
splits for Functions 5–8. Function 7's full-bank selector was better than the
blend on both splits. The Function 5 blend had the lowest mean outer RMSE
but split the two repeats with the GP selector. On the eight later Function 5
returns its error was 478 versus 276 for GP selection. The small neural
ensemble was selected twice for Function 5 and once for Function 8. The
neural fits raised iteration-limit warnings, which are recorded alongside
the GP length-scale bound warnings in each run record.

## Ranking the measured choices

For each outer fold, take the point with the largest predicted score. The
selection gap is the best measured score in that fold minus the measured
score of the chosen point. The gap assesses ranking on shortlists of observed
inputs. A prospective UCB evaluation also depends on uncertainty
and on where the proposed points sit outside the measured coverage.

- **Function 1.** Nearest neighbours had the lowest fixed error. The saved
  and tuned GPs picked the measured best in 10 of 10 shortlists each.
  The full-bank selector picked it in 5 of 10.
- **Function 5.** The saved and tuned GPs also picked the measured best in
  10 of 10. The GP blend did so in 9 of 10, with a mean gap of 5.2.
- **Function 6.** Matérn was selected in all ten GP-selection folds and
  reduced both RMSE and the mean shortlist gap, from 0.083 to 0.025.
- **Function 7.** All three tuned policies reduced the mean shortlist gap
  from 0.137 to 0.064. The full-bank selector cut all-point RMSE by 25% from
  the tuned GP. Across the eight later returns its mean RMSE was 0.137
  versus 0.094 for the tuned GP and 0.081 for the saved GP. These rows are a
  subset of the same folds, and the reversal matters when the next query is
  likely to resemble later exploitation points.

The two split repeats reuse the same observations. Their agreement checks
sensitivity to partitioning, and both are retrospective. The cohort has
sparse coverage in higher dimensions. Nearby points from either source can
still fall on opposite sides of a fold, so the errors measure generalisation
within the sampled regions better than across new regions.

## Reproduction

The run used 639 training and validation partitions across the eight
functions. Each `surrogate-study.json` records the predictions, per-fold
model and weight choices, warning counts and snapshot checksums. The 16
executed cohort notebooks include 24 new figures covering data diagnostics,
the fixed screen and nested validation. [Execution guide](../docs/reproducing-the-analyses.md#week-9-full-cohort-review)

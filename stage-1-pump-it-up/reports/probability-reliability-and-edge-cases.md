# Probability reliability and edge cases after 2.5× oversampling

## Decision

Freeze 2.5× as the final oversampling multiplier examined in this session. It produces competition hard-label repair shares of 6.91–7.10%, close to the independently inferred 7.19% competition prevalence. On the labelled local test, the two XGBoost blends lose only 0.40–0.43 percentage points of accuracy while gaining 10.9–11.9 points of repair recall and 1.24–1.37 points of macro-F1.

This makes 2.5× a credible hard-classification experiment when repair recall matters. It does not make the resulting raw probabilities reliable repair-risk estimates.

## Why matching class balance is not enough

A model can predict exactly the right number of each class while assigning those labels to the wrong rows. Hard-label share is therefore checked separately from probability reliability:

- hard-label balance asks how often each class wins;
- probability balance asks how much probability mass the model assigns to each class;
- reliability asks whether events assigned probability $p$ occur about proportion $p$ of the time;
- row accuracy asks whether the winning class is correct for each individual row.

For the 55% child-weight-1 XGBoost plus 45% Random Forest blend, 2.5× produces a 6.80% repair hard-label share on the local test, close to the observed 7.26%. However, its mean repair probability is 10.12%. The final decisions are balanced because repair often receives more probability without becoming the largest class.

## Probability-quality comparison

For the representative XGBoost/Random Forest blend:

| Diagnostic | Original training | 2.5× repair training | Interpretation |
|---|---:|---:|---|
| Accuracy | 80.70% | 80.30% | Small nominal cost |
| Log loss | 0.468 | 0.477 | Confident probability errors worsen |
| Multiclass Brier score | 0.269 | 0.276 | Overall probability squared error worsens |
| Top-label ECE | 0.56% | 1.34% | Winning-confidence reliability remains fairly close, but worsens |
| Repair one-v-rest ECE | 0.94% | 2.86% | Repair probabilities become materially too optimistic |
| Mean repair probability | 7.28% | 10.12% | Above the observed 7.26% share |
| Component disagreement | 9.76% | 11.62% | More decisions depend on which component is trusted |
| Close-decision share | 5.88% | 6.54% | More rows have a winning margin below ten points |

ECE is a binned descriptive summary and depends on binning; it is not a complete calibration test. Log loss and Brier score avoid the same binning dependence and point in the same direction.

## Repair trade-off and edge groups

For the same blend, 2.5× changes the local-test repair-related groups as follows:

| Group | Original | 2.5× | Change |
|---|---:|---:|---:|
| Repair predicted correctly | 283 | 377 | +94 |
| Repair false alarm | 224 | 431 | +207 |
| Repair missed as functional | 441 | 375 | -66 |
| Repair missed as non-functional | 139 | 111 | -28 |
| High-confidence errors across all classes | 484 | 444 | -40 |
| Close decisions | 699 | 777 | +78 |

The oversampled model finds more real repair cases, but each extra correct repair prediction comes with more than two additional repair false alarms. Whether that is worthwhile is a decision-cost question rather than a universal modelling win.

Close decisions and component disagreements are useful error-analysis cohorts. They identify fragile classifications, not necessarily bad data. High-confidence errors remain more concerning because the model is both wrong and apparently certain.

## Stop rule and next legitimate iteration

No further multiplier or probability-calibration adjustment should be selected against the observed local test in this session. Trying nearby values would progressively convert the holdout into a tuning set.

A later iteration may pre-declare a small multiplier grid and probability-calibration candidates, select them entirely within nested development cross-validation, then use genuinely new evidence—such as a future leaderboard submission or newly reserved labels—for confirmation. Until then, use 2.5× probabilities for class decisions only and do not describe them as calibrated repair risks.

The executable tables are in [`10-probability-reliability-and-edge-cases.ipynb`](../notebooks/10-probability-reliability-and-edge-cases.ipynb).

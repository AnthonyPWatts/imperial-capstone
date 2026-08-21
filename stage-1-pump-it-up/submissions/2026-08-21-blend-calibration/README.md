# 2026-08-21 blend-calibration challenger

This directory holds the candidate produced by the bounded blend comparison in
[`../../notebooks/06-blend-calibration-and-submission.ipynb`](../../notebooks/06-blend-calibration-and-submission.ipynb).

The comparison keeps the selected Random Forest and histogram-boosting
components and the initial 29-feature policy unchanged. It tests two fixed
probability weights around the incumbent equal vote and a regularised
probability stack. The stack creates its combiner inputs inside an inner
four-fold split for every frozen outer development fold.

The reserved local test is not reopened for this selection. A competition CSV
is generated only if the strongest challenger improves the development-fold
mean and wins at least three of the five folds against the incumbent.

The executed notebook records the selected method, validation evidence,
prediction shares, filename and SHA-256 hash. Leaderboard results remain blank
until the candidate is submitted.

## Result

No challenger passed the predeclared selection gate, so no competition CSV was
generated and no submission slot was used.

| Candidate | Mean accuracy | Change | Fold wins | Worst fold change |
| --- | ---: | ---: | ---: | ---: |
| 40% Random Forest + 60% boosting | 81.425% | +0.023 pp | 2/5 | -0.147 pp |
| 60% Random Forest + 40% boosting | 81.271% | -0.130 pp | 0/5 | -0.189 pp |
| Calibrated stack | 81.406% | +0.004 pp | 2/5 | -0.042 pp |

The recreated equal-weight incumbent reached 81.402%. The 40:60 vote produced
the largest mean increase, but that increase came from only two folds. The
calibrated stack took about 10.9 minutes across the five outer folds, produced
almost no mean gain and lowered repair recall from 33.64% to 30.72%.

This run used Python 3.14.5 and scikit-learn 1.9.0. The recreated incumbent is
0.032 percentage points above the 81.37% retained in the earlier executed
notebook, which used a Python 3.13 runtime whose package versions were not
recorded. Every method in this comparison used the same current runtime, so the
relative selection is internally consistent; the difference is retained as a
runtime-reproducibility caveat.

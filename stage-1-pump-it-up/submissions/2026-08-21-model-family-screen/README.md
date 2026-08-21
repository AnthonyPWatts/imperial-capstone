# 2026-08-21 model-family screen

This directory records a bounded model and probability-combination screen on
the unchanged initial 29-feature policy, followed by an explicitly gated
second wave around the strongest ensemble member. The work covers CatBoost,
XGBoost, LightGBM and Extra Trees shapes plus seed-stability runs on the frozen
five development folds. The reserved local test was not reopened.

Every tree count was selected on a fixed stratified split inside the relevant
outer-training fold. The selected count was then used for a fresh refit on the
complete outer-training fold before its untouched validation fold was scored.
The XGBoost and LightGBM models use the existing fold-fitted sparse one-hot
preprocessor. CatBoost uses the same source feature policy with native
categorical handling.

The comparison deliberately tests each new model against each incumbent
component, then checks a narrow weight grid for accepted XGBoost members. This
matters here: no new standalone beats the incumbent ensemble, but XGBoost is a
better complement to Random Forest than the incumbent histogram booster.

## Selection evidence

The recreated current-runtime incumbent is the 50:50 Random Forest and
histogram-boosting vote at 81.402% mean accuracy. A challenger must improve it
by at least 0.10 percentage points, win at least three of five folds, lose no
fold by more than 0.25 percentage points and avoid reducing repair recall by
more than 2 percentage points.

| Candidate | Mean accuracy | Change | Fold wins | Worst fold | Repair recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 55% child-weight-1 depth-8 XGBoost + 45% Random Forest | **81.625%** | **+0.223 pp** | 4/5 | -0.021 pp | 34.859% |
| 60% three-seed child-weight-1 bag + 40% Random Forest | 81.618% | +0.217 pp | 4/5 | -0.042 pp | 34.714% |
| 60% equal depth-6/7/8 bag + 40% Random Forest | 81.595% | +0.194 pp | 5/5 | +0.021 pp | 34.396% |
| 60% second-wave XGBoost bag + 40% Random Forest | 81.597% | +0.196 pp | 4/5 | 0.000 pp | 34.309% |
| Committed leader: 60% depth-8 XGBoost + 40% Random Forest | 81.555% | +0.154 pp | 4/5 | -0.042 pp | 34.309% |
| Incumbent 50% Random Forest + 50% boosting | 81.402% | — | — | — | 33.643% |

The second-wave gate uses the committed 81.555% candidate as its reference and
requires at least +0.05 percentage points, three fold wins and the same
worst-fold and repair-recall safeguards. The new leader gains 0.069 percentage
points, wins four folds and loses its weakest fold by 0.011 percentage points.
Its three standalone XGBoost seeds score 81.014%, 81.019% and 80.955%; their
probability bag also clears the second-wave gate when paired 60:40 with Random
Forest. This supports a repeatable child-weight effect rather than one lucky
seed.

## Generated candidates

The two recommended CSVs contain 14,850 rows in submission-template order, no
missing values and only the three permitted labels. Generated CSVs are ignored
by Git; the filenames and hashes below identify the local artefacts. The two
earlier validated candidates remain locally available as fallbacks.

| Rank | Candidate | Prediction shares (functional / repair / non-functional) | Public score | SHA-256 |
| --- | --- | --- | ---: | --- |
| 1 | 55% child-weight-1 depth-8 XGBoost + 45% Random Forest | 60.256% / 3.953% / 35.791% | **0.8241** | `53889121943054fa8ad8a5cea4e20b0ab357c151e9e88a9b4ee7a0898ba90e4a` |
| 2 | 60% equal depth-6/7/8 XGBoost bag + 40% Random Forest | 60.545% / 3.751% / 35.704% | **0.8240** | `cf87aabe4646cdddad58b7443e361e419c51e4f1f9d9a25269482ae4f4ce63e2` |

The full-data XGBoost refits use the median tree counts selected across the
outer folds: 1,014 trees for child-weight-1 depth 8 and 1,444, 1,352 and 1,090
trees for the depth-6/7/8 bag. The recommended candidates disagree on 1.172%
of OOF labels and 1.192% of competition labels.

Candidate 1 scored **0.8241** on the public leaderboard, improving the prior
project best of 0.8223 by 0.18 percentage points. Candidate 2 was preselected
independently of that result and scored **0.8240**, only 0.01 percentage points
behind candidate 1. The near-identical results support the frozen-fold
selection evidence and preserve candidate 1 as the project leader.

GPU and CPU tree builders need not reproduce bit-identical models. These
XGBoost candidates were selected and refitted consistently with the CUDA
histogram backend on the RTX 4070; their sparse input matrices remain ordinary
CPU-hosted SciPy data and are copied for GPU prediction.

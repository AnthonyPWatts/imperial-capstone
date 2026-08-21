# 2026-08-21 model-family screen

This directory records a bounded model and probability-combination screen on
the unchanged initial 29-feature policy. Four CatBoost GPU shapes, three
XGBoost GPU shapes, two LightGBM CPU shapes and two additional XGBoost seeds
were evaluated on the frozen five development folds. The reserved local test
was not reopened.

Every tree count was selected on a fixed stratified split inside the relevant
outer-training fold. The selected count was then used for a fresh refit on the
complete outer-training fold before its untouched validation fold was scored.
The XGBoost and LightGBM models use the existing fold-fitted sparse one-hot
preprocessor. CatBoost uses the same source feature policy with native
categorical handling.

The comparison deliberately tests each new model at 40:60, 50:50 and 60:40
against each incumbent component. This matters here: no new standalone beats
the incumbent, but XGBoost is a better complement to Random Forest than the
incumbent histogram booster.

## Selection evidence

The recreated current-runtime incumbent is the 50:50 Random Forest and
histogram-boosting vote at 81.402% mean accuracy. A challenger must improve it
by at least 0.10 percentage points, win at least three of five folds, lose no
fold by more than 0.25 percentage points and avoid reducing repair recall by
more than 2 percentage points.

| Candidate | Mean accuracy | Change | Fold wins | Worst fold | Repair recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60% depth-8 XGBoost + 40% Random Forest | **81.555%** | **+0.154 pp** | 4/5 | -0.042 pp | 34.309% |
| 60% three-seed depth-8 bag + 40% Random Forest | 81.551% | +0.149 pp | 4/5 | 0.000 pp | 34.251% |
| 60% XGBoost variant bag + 40% Random Forest | 81.534% | +0.133 pp | 4/5 | -0.063 pp | 34.280% |
| 50% depth-8 XGBoost + 50% Random Forest | 81.519% | +0.118 pp | 3/5 | -0.042 pp | 34.714% |
| 60% conservative depth-11 XGBoost + 40% Random Forest | 81.511% | +0.109 pp | 4/5 | -0.032 pp | 34.453% |
| Incumbent 50% Random Forest + 50% boosting | 81.402% | — | — | — | 33.643% |

The three-seed bag finishes only 0.004 percentage points behind the leading
single-seed recipe and never loses a fold, which is useful stability evidence.
It is not selected for the second slot because its labels are too similar to
the leading candidate. The depth-11 recipe passes independently and disagrees
with the primary on 1.307% of OOF labels.

## Generated candidates

Both CSVs contain 14,850 rows in submission-template order, no missing values
and only the three permitted labels. Generated CSVs are ignored by Git; the
filenames and hashes below identify the local artefacts.

| Rank | Candidate | Prediction shares (functional / repair / non-functional) | SHA-256 |
| --- | --- | --- | --- |
| 1 | 60% depth-8 XGBoost + 40% Random Forest | 60.539% / 3.751% / 35.710% | `34740dd7972f4a44cdfc747bb314de68cc24fc92d0cf8ff708196a67d7d8a8b6` |
| 2 | 60% conservative depth-11 XGBoost + 40% Random Forest | 60.418% / 3.838% / 35.744% | `69bf072d7662ab1a3a5024ecb38ae4eb286d4e6b434c6b26eafc3c024ba89071` |

The full-data XGBoost refits use the median tree counts selected across the
outer folds: 1,090 trees for depth 8 and 900 for conservative depth 11. The two
competition candidates disagree on 1.205% of labels. Leaderboard scores remain
blank until the files are submitted.

GPU and CPU tree builders need not reproduce bit-identical models. These
XGBoost candidates were selected and refitted consistently with the CUDA
histogram backend on the RTX 4070; their sparse input matrices remain ordinary
CPU-hosted SciPy data and are copied for GPU prediction.

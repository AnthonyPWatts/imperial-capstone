# Numeric state and imputation screen

## Decision

Retain the accepted numeric representation. None of the eleven bounded
state, ablation or imputation challengers passed the frozen-fold promotion
gate, and every complete vote scored below the 81.625% baseline.

The current policy remains:

- retain `amount_tsh` magnitude plus `amount_tsh_recorded`;
- treat zero `gps_height` and `population` as unavailable, retain explicit
  missing-state flags and impute magnitude with the training-fold median;
- retain the sparse raw `num_private` magnitude;
- leave all other numeric preprocessing unchanged.

Fold-fitted geographic population imputation is worth retaining as a named
transfer technique rather than as the competition policy. It trailed by 0.034
percentage points on the primary folds, but improved 0.128 points and won four
of five LGA-disjoint folds when it had to fall back to region or global medians.

The labelled local test remained unopened. No competition predictions were
generated or uploaded, and no oversampling data or code participated in this
experiment.

## Audit evidence and scope

The four fields have very different sentinel and support patterns:

| Predictor | Zero rows | One rows | Positive rows | Distinct values | Positive median | Positive 95th percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `amount_tsh` | 41,639 (70.1%) | 3 | 17,761 | 98 | 250 | 5,000 |
| `gps_height` | 20,438 (34.4%) | 34 | 37,466 | 2,428 | 1,194 | 1,910 |
| `population` | 21,381 (36.0%) | 7,025 | 38,019 | 1,049 | 150 | 897 |
| `num_private` | 58,643 (98.7%) | 73 | 757 | 65 | 15 | 102 |

There are 19,668 labelled rows where height, population and construction year
are all zero. Their class shares are close to the complete-data shares, so the
shared state is more plausibly a collection pattern than a direct outcome
marker.

The baseline already implements most of the audit recommendation. The screen
therefore focuses on genuinely different representations rather than repeating
equivalent transforms:

- remove amount magnitude while retaining the recorded-state flag;
- retain height or population zero as a numeric value;
- add an explicit population-one state;
- remove `num_private` or replace it with a non-zero state;
- add the shared measurement-block state;
- impute missing height or population from training-fold LGA, region and global
  medians, separately and together;
- combine the three new state indicators once.

Logarithmic magnitude variants were not run. The accepted tree methods split on
value order, so a monotonic `log1p` transform does not provide a materially new
ordering. PCA was also excluded: it would turn the sparse, interpretable mixed
feature space into dense components without addressing the identified sentinel
problem.

## Evaluation contract

Every policy used the unchanged 55% child-weight-1 depth-8 XGBoost and 45%
Random Forest probability vote. XGBoost selected its tree count on a separate
inner stopping split within each outer-training fold before refitting. Every
median and preprocessing statistic was fitted inside the current training
partition.

The primary comparison used the five frozen stratified development folds. A
challenger had to improve mean accuracy by at least 0.10 percentage points, win
at least three folds, lose no fold by more than 0.25 points and avoid reducing
repair recall by more than 2 points.

## Frozen-fold screen

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall | Fold-1 transformed features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted numeric policy | **81.625%** | — | — | — | 34.859% | 301 |
| Population geographic median | 81.591% | -0.034 pp | 1/5 | -0.095 pp | 34.656% | 300 |
| Population-one state | 81.589% | -0.036 pp | 3/5 | -0.179 pp | 34.801% | 302 |
| Shared measurement-block state | 81.587% | -0.038 pp | 1/5 | -0.116 pp | 34.830% | 302 |
| Remove `num_private` | 81.576% | -0.048 pp | 1/5 | -0.105 pp | 34.801% | 300 |
| Population zero retained | 81.553% | -0.072 pp | 1/5 | -0.147 pp | 34.656% | 300 |
| Amount recorded state only | 81.543% | -0.082 pp | 2/5 | -0.253 pp | 35.119% | 300 |
| `num_private` non-zero state only | 81.540% | -0.084 pp | 1/5 | -0.221 pp | 34.714% | 301 |
| Height and population geographic medians | 81.532% | -0.093 pp | 1/5 | -0.168 pp | 34.801% | 299 |
| GPS height geographic median | 81.517% | -0.107 pp | 1/5 | -0.200 pp | 34.917% | 300 |
| Combined numeric state indicators | 81.486% | -0.139 pp | 0/5 | -0.221 pp | 34.598% | 303 |
| GPS height zero retained | 81.469% | -0.156 pp | 0/5 | -0.231 pp | 34.540% | 300 |

The result supports the current sentinel semantics. Treating zero height or
population as a measured value is worse than masking it. Removing amount
magnitude shows that the positive values carry information beyond the recorded
state. The raw sparse `num_private` field is also slightly better than either
deletion or a binary replacement.

Population-one and shared-block flags are near baseline but do not add value.
The existing magnitude and separate state columns already let the trees isolate
those cases when useful.

## Geographic population imputation

For unavailable population, the fitted transformer uses the training-partition
median at decreasing specificity:

1. LGA, when at least 20 non-zero training measurements are available;
2. region;
3. the training-partition global median.

The original `population_missing = 1` state remains visible. Across the frozen
validation folds, 17,139 missing-population rows used LGA medians for 1,261
rows (7.4%), region medians for 6,137 (35.8%) and the global median for 9,741
(56.8%). Four complete regions have no non-zero population measurements in the
source data, which explains the frequent global fallback.

Both standalone components improve slightly under this treatment: XGBoost
moves from 81.014% to 81.088%, and Random Forest from 80.591% to 80.654%.
Their changed errors do not form a better 55:45 decision boundary. Two bounded
no-refit cross-policy votes confirm that:

| Vote | Mean accuracy | Change | Fold wins |
| --- | ---: | ---: | ---: |
| Baseline XGBoost + population-median Random Forest | 81.587% | -0.038 pp | 2/5 |
| Population-median XGBoost + baseline Random Forest | 81.551% | -0.074 pp | 1/5 |

## LGA-disjoint sensitivity

The closest geography-dependent challenger received one robustness-only check.
No validation LGA appeared in training, so none of the 17,139 missing rows could
use an LGA median. Region medians covered 6,592 rows (38.5%) and the global
median covered 10,547 (61.5%).

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accepted numeric policy | 72.101% | — | — | — | 3.497% |
| Population geographic median | **72.228%** | **+0.128 pp** | **4/5** | -0.306 pp | 3.348% |

The five fold changes are +0.316, +0.010, +0.426, +0.192 and -0.306 points.
The mean improvement is mainly associated with Random Forest, which rises from
72.192% to 72.364%; XGBoost rises from 71.694% to 71.706%.

This is useful transfer evidence, but it does not reopen primary selection.
The policy already failed the competition-aligned frozen folds and the grouped
result breaches the worst-fold limit.

## Stop rule and next step

The accepted numeric policy remains frozen. Do not tune more imputation
thresholds, bins, logarithms or PCA variants against these same folds in this
iteration. Geographic population imputation remains available for future work
whose deployment question is transfer to unfamiliar LGAs.

The next data-focused loop should rationalise categorical hierarchies, beginning
with `management`, `scheme_management` and `management_group`, then applying
the same bounded ablation principle to extraction, source, quality and
waterpoint families.

Executable screening lives in
[`scripts/run_numeric_screen.py`](../scripts/run_numeric_screen.py), with
reusable policies in [`src/numeric_features.py`](../src/numeric_features.py)
and evaluation in [`src/numeric_evaluation.py`](../src/numeric_evaluation.py).
Runtime models and CSV summaries remain ignored local artefacts under
`.runtime/numeric-screen/`.

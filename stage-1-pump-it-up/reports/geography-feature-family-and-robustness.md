# Geography feature family and robustness

## Decision

Retain the accepted geography representation for the competition-accuracy
recipe: the plausible raw coordinate pair, one coordinate-missingness flag,
basin, region, region code, district code and LGA. None of the twelve bounded
alternatives passed the frozen-fold promotion gate.

Keep fold-fitted hierarchical coordinate-centroid imputation as a named
robustness technique rather than as the selected competition feature policy.
It trailed the accepted recipe by 0.059 percentage points on the primary frozen
folds, but improved 0.161 points and won four of five folds when validation LGAs
were completely absent from training.

The labelled local test remained unopened. No competition predictions were
generated or uploaded, and no oversampling data or code participated in this
experiment.

## Coordinate validity

The supplied missing-location sentinel is `(0, -2e-08)`, not exactly `(0, 0)`.
It occurs in 1,812 training rows and 457 competition rows. A longitude-only
epsilon check happened to catch these rows but did not express the real paired
coordinate contract.

The shared feature engineering now accepts a coordinate pair only when it falls
inside a conservative Tanzania envelope:

- longitude from 28° to 42°;
- latitude from -13° to 0°.

This rejects the origin and suspicious nearby values together. It retains the
legitimate northern points around latitude -1° and longitude 31° in
Kagera/Misenyi. The same rule drives raw-coordinate masking, the
`coordinates_missing` flag, centroid fitting and grid construction.

## Evaluation contract

Every policy used the unchanged 55% child-weight-1 depth-8 XGBoost and 45%
Random Forest probability vote. XGBoost selected its tree count on a separate
inner stopping split within each outer-training fold before refitting. All
preprocessing, rare-category handling and coordinate centroids were fitted
inside the current training partition.

The primary comparison used the five frozen stratified development folds. A
challenger had to improve mean accuracy by at least 0.10 percentage points, win
at least three folds, lose no fold by more than 0.25 points and avoid reducing
repair recall by more than 2 points.

## Frozen-fold screen

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall | Fold-1 transformed features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted current geography | **81.625%** | — | — | — | 34.859% | 301 |
| Hierarchical coordinate centroids | 81.566% | -0.059 pp | 1/5 | -0.179 pp | 34.859% | 299 |
| Remove raw codes; retain LGA | 81.547% | -0.078 pp | 1/5 | -0.210 pp | 34.627% | 257 |
| Coordinate centroids plus imputation level | 81.538% | -0.086 pp | 0/5 | -0.147 pp | 34.743% | 303 |
| Region-district composite | 81.526% | -0.099 pp | 1/5 | -0.210 pp | 34.685% | 383 |
| Add ~10 km coordinate grid | 81.507% | -0.118 pp | 2/5 | -0.284 pp | 34.598% | 815 |
| Coordinates plus basin and region only | 81.500% | -0.124 pp | 0/5 | -0.179 pp | 34.280% | 133 |
| Add ~20 km coordinate grid | 81.456% | -0.168 pp | 0/5 | -0.210 pp | 34.251% | 844 |
| Add LGA-ward composite | 81.454% | -0.171 pp | 1/5 | -0.421 pp | 34.946% | 987 |
| Add both coordinate grids | 81.410% | -0.215 pp | 1/5 | -0.589 pp | 34.366% | 1,358 |
| Administrative composites plus both grids | 81.370% | -0.255 pp | 0/5 | -0.463 pp | 34.859% | 2,126 |
| Coordinates without named geography | 81.326% | -0.299 pp | 0/5 | -0.516 pp | 33.643% | 103 |
| Named geography without coordinates | 80.890% | -0.734 pp | 0/5 | -0.915 pp | 34.453% | 296 |

Coordinates and named geography are complementary. Removing coordinates costs
0.734 points; removing all named geography costs 0.299 points. Coarser
administrative representations retain most but not all of the accepted score.
Sparse grid and ward features add hundreds or thousands of transformed columns
without improving accuracy.

## Hierarchical coordinate imputation

For an implausible or unavailable coordinate pair, the fitted transformer uses
coordinate-wise training medians at decreasing specificity:

1. LGA plus ward, when at least five valid training sites are available;
2. LGA, when at least ten valid training sites are available;
3. region;
4. basin;
5. the training-partition global median.

Medians are used instead of means so one poor GPS observation cannot pull a
centre substantially. The original `coordinates_missing = 1` state is retained.
An alternative also exposed the selected fallback level as a categorical
feature, but that reduced accuracy further.

Across the five frozen validation folds, the 1,438 unavailable coordinate rows
were imputed from ward for 152 rows (10.6%), LGA for 896 (62.3%) and region for
390 (27.1%). Basin and global fallbacks were not needed.

Centroid imputation slightly improved standalone Random Forest accuracy from
80.591% to 80.608%, while reducing XGBoost from 81.014% to 80.979%. Two bounded
cross-policy votes did not recover a blend improvement:

| Vote | Mean accuracy | Change | Fold wins |
| --- | ---: | ---: | ---: |
| Baseline XGBoost + centroid Random Forest | 81.593% | -0.032 pp | 2/5 |
| Centroid XGBoost + baseline Random Forest | 81.521% | -0.103 pp | 1/5 |

## LGA-disjoint sensitivity

The sensitivity design assigned 22 to 27 complete LGAs to each of five
validation folds. No LGA appeared in both training and validation for a fold.
Class shares remained reasonably close, but this is deliberately a harder
transfer question than the competition-aligned random folds.

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accepted current geography | 72.101% | — | — | — | 3.497% |
| Hierarchical coordinate centroids | **72.261%** | **+0.161 pp** | **4/5** | -0.190 pp | 2.990% |

All 1,438 missing-coordinate validation rows fell into held-out LGAs and wards.
The hierarchy therefore used region medians for every one of them, providing a
real back-off rather than memorising the held-out administrative unit.

The roughly 9.5-point accuracy drop and near-disappearance of repair recall are
the more important result. Random development folds substantially reward
within-geography reuse, and the accepted competition score should not be read
as evidence of transfer to entirely unseen LGAs. Centroid imputation makes that
transfer slightly less poor; it does not solve it.

## Stop rule and next step

The accepted geography policy remains frozen. Further grid resolutions,
centroid thresholds or hybrid weights should not be selected against these
same folds in this iteration.

The next feature-family loop should move to organisation fields, beginning with
fold-fitted conservative treatments of `funder` and `installer`, while retaining
the accepted model recipe and primary evaluation gate. The LGA-disjoint result
remains a separate robustness warning rather than a replacement selection
metric.

Executable screening lives in
[`scripts/run_geography_screen.py`](../scripts/run_geography_screen.py), with
reusable policies in [`src/geography_features.py`](../src/geography_features.py)
and evaluation in
[`src/geography_evaluation.py`](../src/geography_evaluation.py). Runtime models
and CSV summaries remain ignored local artefacts under `.runtime/geography-screen/`.

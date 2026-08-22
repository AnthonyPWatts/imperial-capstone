# Cross-fitted spatial-outcome screen

## Decision

Do not add local spatial class-rate features. The fixed ten-neighbour,
20-observation-smoothed feature policy reaches 80.995% mean accuracy, 0.629
percentage points below the accepted ensemble with zero fold wins. A direct 15%
spatial voter is much closer at 81.623%, but remains below baseline and loses
2.635 points of repair recall.

The local test remained closed and no competition prediction was generated.

## Course-aligned lifecycle

| Step | Application in this experiment |
| --- | --- |
| 1. Define the goal and scope | Test whether geographically local outcome structure supplies residual signal beyond coordinates and administrative categories. |
| 2. Gather the data | Reuse the frozen development predictors, labels, folds and accepted OOF probabilities. |
| 3. Explore the data | Compare leakage-safe local voters at 10, 25, 50 and 100 neighbours before choosing one feature-level policy. |
| 4. Clean and preprocess the data | Apply the accepted Tanzania coordinate envelope; invalid coordinates receive the training class prior and a missing-radius indicator. |
| 5. Select and engineer features | Add three smoothed neighbour class rates and the kth-neighbour radius to the accepted feature frame. |
| 6. Define the machine-learning task | Unchanged supervised three-class classification, plus a diagnostic spatial-probability voter. |
| 7. Partition the data | Cross-fit training features over five inner folds; use all outer-training neighbours for outer validation; never use a row as its own neighbour. |
| 8. Select and train candidate methods | Refit the accepted XGBoost and Random Forest once at ten neighbours and smoothing 20. |
| 9. Evaluate and interpret the results | Compare the complete vote, direct 15% voter and two fixed component crosses with the accepted baseline. |
| 10. Deploy and iterate | Stop spatial class-rate features; retain the accepted representation. |

## OOF neighbourhood audit

Each audit probability uses only pumps in the current outer-training partition.
Great-circle neighbours are found with a haversine Ball Tree. Class counts are
mixed with the outer-training prior using a fixed smoothing mass of 20.

| Neighbours | Standalone accuracy | Median radius | Best audited blend | Blend accuracy change |
| ---: | ---: | ---: | ---: | ---: |
| 10 | **65.040%** | 3.28 km | 15% | -0.002 pp |
| 25 | 64.021% | 6.69 km | 10% | -0.004 pp |
| 50 | 62.854% | 10.56 km | 5% | -0.057 pp |
| 100 | 61.877% | 16.30 km | 5% | -0.063 pp |

Ten neighbours is the only feature-level policy. The audit did not justify a
neighbour or smoothing grid.

## Leakage and fallback contract

For training transformations, every row receives neighbours from a different
inner-training partition. After producing all inner-OOF training values, the
transformer fits a full outer-training Ball Tree for validation. Exact self
neighbours are therefore impossible in training features. The five validation
folds have 276–308 invalid-coordinate rows; they receive the outer-training
class prior and a missing radius that the numeric pipeline imputes while adding
an indicator.

Five focused tests cover cross-fitting versus self-inclusive transformation,
determinism, prior fallback, finite preprocessing and invariance of one outer
validation fold's probabilities to changes in that fold's labels.

## Model results

| Candidate | Accuracy | Change | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accepted ensemble | **81.625%** | — | — | — | — |
| Accepted ensemble + 15% spatial voter | 81.623% | -0.002 pp | 3/5 | -0.179 pp | -2.635 pp |
| Spatial-feature 55:45 vote | 80.995% | -0.629 pp | 0/5 | -0.884 pp | -4.922 pp |
| Spatial voter alone | 65.040% | -16.585 pp | 0/5 | -16.877 pp | -33.817 pp |

The feature policy lowers XGBoost from 81.014% to 80.764% and Random Forest
from 80.591% to 80.427%. Pairing spatial XGBoost with the accepted forest reaches
81.378%; pairing the accepted XGBoost with the spatial forest reaches 81.370%.
Neither component contains a hidden improvement.

## Interpretation

Nearby outcomes are informative in isolation, but the accepted model already
receives coordinates, LGA, region and other correlated physical context. The
spatial rates mostly duplicate that information and push probability away from
the minority repair class. A tree does not recover additional conditional
signal from them.

Stop spatial-neighbour and smoothing searches. A further gain now needs a
materially different training strategy or genuinely new information, not
another aggregation of the existing geography and labels.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_spatial_outcome_screen.py
```

Runtime evidence remains ignored under `.runtime/spatial-outcome-screen/`.

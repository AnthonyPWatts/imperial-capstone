# Radial Random Forest screen

## Decision

Retain the existing accepted and archive components. Adding geodesic distance
from `(0, 0)` to the accepted one-hot Random Forest reaches **80.5955%**, only
0.0042 percentage points above the original forest's 80.5913%.

Splitting the archive's 18% accepted-forest allocation equally between the
original and radial variants reaches **81.8056%**, a +0.0126-point change with
only two of five fold wins. Direct replacement reaches 81.7950%. Neither
passes the +0.10-point gate, so the local test remains closed and no
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Determine whether the radial feature that improved the compact count forest also strengthens the accepted one-hot Random Forest. |
| 2. Gather the data | Reuse supplied coordinates, frozen development folds and cached promoted/archive components. |
| 3. Explore the data | Use the single previously fixed origin-distance definition without testing alternative origins or projections. |
| 4. Clean and preprocess the data | Apply the accepted coordinate envelope; invalid pairs become missing and are median-imputed inside each training fold with an indicator. |
| 5. Select and engineer features | Append one target-free radial distance to the unchanged accepted feature frame. |
| 6. Define the machine-learning task | Unchanged three-class accuracy. |
| 7. Partition the data | Fit imputation, encoding and the forest strictly within each frozen outer-training fold. |
| 8. Select and train candidate methods | Refit one accepted-spec Random Forest and make fixed direct-substitution and equal within-forest bag comparisons. |
| 9. Evaluate and interpret the results | Compare standalone, promoted-vote and archive-vote accuracy, fold stability, repair recall and diversity. |
| 10. Deploy and iterate | Close without local-test or competition inference because all full-vote gains miss the gate. |

## Feature and evaluation contract

The shared radial calculation is the same valid-coordinate great-circle
distance used in the compact archive experiment. The accepted categorical
policy, all original numeric fields, 300 trees, square-root feature sampling,
leaf size one and seed remain unchanged.

The new frame has 30 engineered features. The distance supplies a diagonal
projection of longitude and latitude that can be easier for axis-aligned forest
splits, but it adds no target or external data.

## Results

| Candidate | Accuracy | Change vs archive | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal accepted/radial forest archive bag | **81.8056%** | **+0.0126 pp** | 2/5 | -0.0526 pp | -0.0290 pp |
| Radial forest archive substitution | 81.7950% | +0.0021 pp | 4/5 | -0.0947 pp | -0.0579 pp |
| Existing archive synthesis | 81.7929% | — | — | — | — |
| Equal accepted/radial forest promoted bag | 81.7508% | -0.0421 pp | 2/5 | -0.1473 pp | -0.0869 pp |
| Promoted complete-identity vote | 81.7403% | -0.0526 pp | 0/5 | -0.0842 pp | +0.2026 pp |
| Radial forest promoted substitution | 81.7319% | -0.0610 pp | 1/5 | -0.1263 pp | -0.0579 pp |
| Radial Random Forest | 80.5955% | -1.1974 pp | 0/5 | -1.3468 pp | +3.1552 pp |
| Accepted Random Forest | 80.5913% | -1.2016 pp | 0/5 | -1.3047 pp | +3.1262 pp |

The two forests disagree on 2.6515% of rows and have nearly symmetric unique-
correct shares: 1.1827% for the accepted forest and 1.1869% for the radial
forest. The archive bag changes 0.2189% of incumbent decisions, gaining 0.1052%
and losing 0.0926% for six net rows.

That small gain is consistent with variance averaging, not a new signal large
enough to promote after extensive fold reuse. It is also slightly below the
separate exploratory CatBoost-boundary archive result of 81.8077%.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_radial_random_forest_screen.py
```

Runtime evidence remains ignored under `.runtime/radial-random-forest-screen/`.

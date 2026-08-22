# Exact archive frequency-forest screen

## Decision

Retain the existing archive synthesis and its ordinary 300-tree frequency
forest. Completing the adapted second-place archive specification with 1,000
trees and exactly five candidate features per split reaches **80.7281%** as a
standalone component. This is above the original occurrence-count forest's
80.6103%, but below the 300-tree radial forest's 80.7471%.

Substitution into the complete archive vote reaches **81.7761%**, 0.0168
percentage points below the 81.7929% incumbent, with only one of five fold
wins. The local test remains closed and no competition prediction was
generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Complete the archived occurrence-count model's documented forest settings rather than approximate them with the general project defaults. |
| 2. Gather the data | Reuse supplied coordinates and categories, frozen development folds and cached archive components. |
| 3. Explore the data | Compare the one faithful adapted specification with the already-evaluated original and radial frequency forests. |
| 4. Clean and preprocess the data | Reuse fold-fitted occurrence counts, Tanzania coordinate validity and radial-distance median imputation. |
| 5. Select and engineer features | Keep the 44-feature radial occurrence-count representation unchanged. |
| 6. Define the machine-learning task | Unchanged three-class accuracy. |
| 7. Partition the data | Fit preprocessing and every forest within each frozen outer-training partition. |
| 8. Select and train candidate methods | Fit 1,000 Random Forest trees with `max_features=5`, leaf size one and the existing reproducible seed. |
| 9. Evaluate and interpret the results | Compare standalone accuracy, diversity and one fixed archive substitution. |
| 10. Deploy and iterate | Close the archive reproduction without local-test or competition inference because the substitution fails the gate. |

## Exact adapted specification

DrivenData's official archive records a 0.8265 public score for the
[`madRid` solution](https://github.com/drivendataorg/pump-it-up/blob/master/madRid/PumpItUp_DataDriven_2nd_place_madRid_team.R).
Its code uses 1,000 Rborist trees, five candidate predictors per split, leaf
size one, occurrence-count categorical values and distance from `(0, 0)`.

The adaptation preserves those observable choices while retaining the project's
safer coordinate sentinel handling, immutable folds and scikit-learn forest.
It is therefore faithful at the feature and main forest-parameter level, not a
claim of binary equivalence between Rborist and scikit-learn implementations.

## Results

| Candidate | Accuracy | Change vs archive | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing archive synthesis | **81.7929%** | — | — | — | — |
| Exact-spec archive substitution | 81.7761% | -0.0168 pp | 1/5 | -0.0421 pp | -0.0290 pp |
| Radial frequency forest, 300 trees and square-root sampling | **80.7471%** | -1.0459 pp | 0/5 | -1.2416 pp | +2.1421 pp |
| Exact adapted frequency forest | 80.7281% | -1.0648 pp | 0/5 | -1.2311 pp | +2.0554 pp |
| Original frequency forest | 80.6103% | -1.1827 pp | 0/5 | -1.5152 pp | +1.9973 pp |

The exact forest and radial default forest disagree on 1.9676% of development
rows. The exact model alone is correct on 0.8775% and the default radial model
alone on 0.8965%, explaining the small standalone advantage for the latter.
The exact archive substitution changes 0.2357% of archive decisions but loses
eight net development rows.

The evidence isolates the useful historical clue: radial distance improved the
compact component, while the documented tree count and feature-sampling value
did not improve this implementation or the final ensemble.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_exact_archive_frequency_forest_screen.py
```

Runtime evidence remains ignored under
`.runtime/exact-archive-frequency-forest-screen/`.

# One-vs-rest XGBoost screen

## Decision

Retain multiclass XGBoost inside the promoted identity vote. Three independent
one-vs-rest boundaries improve standalone XGBoost from 81.0143% to **81.0816%**,
but direct substitution lowers the final ensemble. One fixed equal boundary
bag reaches **81.7593%**, only **+0.0189 percentage points** above the promoted
vote. It wins 3/5 folds and loses none, but remains well below the +0.10-point
promotion gate.

No class weights, thresholds or blend grids are tuned. No local-test or
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether independent binary class boundaries improve flat multiclass decisions. |
| 2. Gather the data | Reuse accepted features and unchanged three-class labels. |
| 3. Explore the data | Build on prior flat multiclass, cumulative ordinal and repair-removed binary results. |
| 4. Clean and preprocess the data | Fit accepted preprocessing separately inside every binary boundary's training partition. |
| 5. Select and engineer features | No feature change. |
| 6. Define the machine-learning task | Three one-vs-rest binary tasks normalised back to one three-class membership vector. |
| 7. Partition the data | Keep preprocessing, inner stopping and refitting inside each frozen outer fold. |
| 8. Select and train candidate methods | Reuse unweighted depth-8 child-weight-1 XGBoost for all three boundaries. |
| 9. Evaluate and interpret the results | Compare direct substitution and one equal multiclass/OvR boundary bag. |
| 10. Deploy and iterate | Retain the promoted multiclass component because the bounded gain misses the gate. |

## Probability construction

Each binary model estimates the positive probability for one class. The three
positive probabilities need not sum to one, so each row is divided by its
positive-probability sum. This preserves the largest independent membership
while producing a finite, non-negative three-class probability vector. No
threshold is learned from outer-validation labels.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Equal boundary-bag identity vote | **81.7593%** | +0.0189 pp | 3/5 | 0.0000 pp | -0.4053 pp | No |
| Promoted identity vote | 81.7403% | — | — | — | — | incumbent |
| OvR-XGBoost identity vote | 81.7235% | -0.0168 pp | 3/5 | -0.0737 pp | -0.9845 pp | No |
| One-vs-rest XGBoost | 81.0816% | -0.6587 pp | 0/5 | -0.9575 pp | -4.4885 pp | No |
| Equal multiclass/OvR XGBoost bag | 81.0690% | -0.6713 pp | 0/5 | -0.8838 pp | -3.4171 pp | No |
| Accepted multiclass XGBoost | 81.0143% | -0.7260 pp | 0/5 | -0.8733 pp | -2.4034 pp | No |

The independent task improves standalone accuracy but changes probability
calibration relative to Random Forest and CatBoost. Equal averaging stabilises
the final vote, yet the aggregate gain is only nine development rows and is not
sufficiently large for promotion.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_one_vs_rest_screen.py
```

Runtime evidence remains ignored under `.runtime/one-vs-rest-screen/`.

# One-vs-rest complete-identity CatBoost screen

## Decision

Retain multiclass complete-identity CatBoost. Three independent binary
boundaries improve standalone CatBoost from 80.6229% to **81.1301%**. The fixed
equal multiclass/OvR CatBoost boundary bag raises the promoted vote to
**81.7761%**, a **+0.0358 percentage-point** gain with **4/5** fold wins, but it
remains below the +0.10-point promotion gate.

Combining both XGBoost and CatBoost boundary bags reaches 81.7593% and does not
compound their separate gains. No class weights, thresholds or blend grids are
tuned. No local-test or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test independent class boundaries in the successful native-identity representation. |
| 2. Gather the data | Reuse accepted features, six deferred identities and unchanged labels. |
| 3. Explore the data | Build on the standalone improvement from one-vs-rest XGBoost. |
| 4. Clean and preprocess the data | Engineer and normalise native identities separately inside every training partition. |
| 5. Select and engineer features | No feature change from complete-identity CatBoost. |
| 6. Define the machine-learning task | Three OvR binary tasks normalised to one three-class membership vector. |
| 7. Partition the data | Keep inner stopping and refitting inside each frozen outer fold. |
| 8. Select and train candidate methods | Reuse the unweighted depth-8 CatBoost policy for all boundaries. |
| 9. Evaluate and interpret the results | Test direct substitution, one equal CatBoost boundary bag and one combined boundary vote. |
| 10. Deploy and iterate | Retain multiclass CatBoost because all gains miss the gate and OvR is expensive. |

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| CatBoost boundary-bag identity vote | **81.7761%** | +0.0358 pp | 4/5 | -0.0210 pp | -0.0579 pp | No |
| OvR-CatBoost identity vote | 81.7698% | +0.0295 pp | 2/5 | -0.0421 pp | -0.1739 pp | No |
| Combined XGBoost/CatBoost boundary vote | 81.7593% | +0.0189 pp | 3/5 | -0.1578 pp | -0.6081 pp | No |
| Promoted identity vote | 81.7403% | — | — | — | — | incumbent |
| One-vs-rest identity CatBoost | 81.1301% | -0.6103 pp | 1/5 | -1.3468 pp | -3.9382 pp | No |
| Equal CatBoost boundary bag | 81.0606% | -0.6797 pp | 0/5 | -1.3258 pp | -4.7780 pp | No |
| Multiclass identity CatBoost | 80.6229% | -1.1174 pp | 0/5 | -1.5888 pp | -5.8489 pp | No |

The five OvR fold evaluations took 334.5, 383.3, 394.8, 251.4 and 368.3
seconds: about 28.9 fold-minutes in total. That is roughly three times the
already expensive CatBoost component architecture. The accuracy gain is only
17 development rows and is not large enough to justify this operational cost
or promotion after extensive reuse of the same folds.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_one_vs_rest_catboost_screen.py
```

Runtime evidence remains ignored under `.runtime/one-vs-rest-catboost-screen/`.

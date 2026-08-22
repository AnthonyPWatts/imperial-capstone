# CatBoost identity follow-up screen

## Decision

Retain the promoted depth-8 complete-identity vote unchanged. Two tightly
bounded follow-ups fail to improve it materially:

- replacing the accepted XGBoost and Random Forest representations with the
  source-plus-class versions reaches **81.7466%**, only **+0.0063 percentage
  points** or three development rows;
- replacing depth-8 identity CatBoost with the earlier diversity-motivated
  depth-7 specification reaches **81.7319%**, **-0.0084 points**.

Neither passes the +0.10-point promotion gate. No local test or competition
prediction was generated. Stop physical crosses and CatBoost depth tuning.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test two already-motivated ways to improve the promoted identity vote without broad tuning. |
| 2. Gather the data | Reuse cached complete-identity, accepted and source-plus-class fold probabilities. |
| 3. Explore the data | Use the physical-hierarchy public lead and earlier depth-7 CatBoost diversity result as prior evidence. |
| 4. Clean and preprocess the data | Reuse each component's already fold-fitted policy. |
| 5. Select and engineer features | Make no new features; cross one existing physical policy and retain the six identities. |
| 6. Define the machine-learning task | Unchanged three-class accuracy. |
| 7. Partition the data | Preserve the five frozen development folds. |
| 8. Select and train candidate methods | Reuse the physical probabilities; fit one depth-7 CatBoost with unchanged identity representation and 20% weight. |
| 9. Evaluate and interpret the results | Compare both candidates directly with the promoted depth-8 identity vote. |
| 10. Deploy and iterate | Reject both and move to a distinct text representation without opening local test. |

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Depth-8 complete-identity vote | **81.7403%** | — | — | — | — | incumbent |
| Source-plus-class identity cross | 81.7466% | +0.0063 pp | 4/5 | -0.1473 pp | -0.0292 pp | No |
| Depth-7 complete-identity vote | 81.7319% | -0.0084 pp | 2/5 | -0.1578 pp | -0.1738 pp | No |
| Source-plus-class 55:45 vote | 81.6351% | -0.1052 pp | 0/5 | -0.2525 pp | +0.9263 pp | No |
| Depth-7 complete-identity CatBoost | 80.5829% | -1.1574 pp | 0/5 | -1.7151 pp | -5.8777 pp | No |

The physical cross is directionally consistent in four folds but far below the
minimum material gain and does not justify another local-test use. The depth-7
model selected 2,892, 2,741, 2,423, 2,639 and 1,853 trees; its slightly weaker
standalone and ensemble results reject the diversity hypothesis.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_catboost_identity_physical_cross.py
..\.venv\Scripts\python.exe .\scripts\run_catboost_identity_depth7_screen.py
```

Runtime evidence remains ignored under the corresponding `.runtime/` folders.

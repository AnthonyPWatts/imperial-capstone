# Learned numeric-imputation screen

## Decision

Retain the original `amount_tsh` handling and the promoted complete-identity
vote. A fold-fitted two-stage reconstruction—ten-neighbour spatial height,
followed by target-free log-amount regression—slightly strengthens the
standalone Random Forest from 80.5913% to **80.6418%**. Every fixed ensemble
variant is weaker, however. The best learned-numeric vote reaches **81.7003%**,
0.0400 percentage points below the promoted model, and the equal
original/learned representation bag reaches only **81.6919%**.

No amount threshold, regressor family or blend weight is screened. No local-
test or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test the archived solution's learned replacement of unavailable numeric values without changing the target task. |
| 2. Gather the data | Use only the outer-training predictors and labels already present in the frozen development partition. |
| 3. Explore the data | Treat non-positive or missing `amount_tsh` as unavailable while retaining its original recorded-state indicator. |
| 4. Clean and preprocess the data | Reconstruct unavailable GPS height first, then fit amount regression on positive outer-training observations only. |
| 5. Select and engineer features | Predict `log1p(amount_tsh)` from the 28 remaining compact numeric and category-frequency features. |
| 6. Define the machine-learning task | Keep three-class classification and accuracy as the scored task; numeric regression is an auxiliary target-free preprocessing fit. |
| 7. Partition the data | Fit the height index, frequency mapping and amount regressor separately inside every outer-training partition. |
| 8. Select and train candidate methods | Refit the accepted XGBoost and Random Forest specifications and apply fixed promoted-vote crosses. |
| 9. Evaluate and interpret the results | Compare standalone components, one-component substitutions, full replacement and one equal representation bag. |
| 10. Deploy and iterate | Reject promotion and close learned-amount variants because none improves the frozen leader. |

## Leakage contract

For every outer fold, the spatial index contains only measured-height rows
from that fold's training partition. Category frequencies are learned from the
same rows. The amount regressor is trained only on their positive
`amount_tsh` values and never receives `status_group`. It then reconstructs
unavailable amounts in both training and validation predictors. The original
`amount_tsh_recorded` flag remains zero, allowing the classifiers to retain
the distinction between a measured and reconstructed value.

The fixed amount regressor is a 200-iteration histogram gradient booster over
`log1p(amount_tsh)`, with 31 leaves, 20 samples per leaf and L2 regularisation
of 1. This is one bounded mechanism test, not a regression-model screen.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Spatial-height representation bag | **81.7677%** | +0.0274 pp | 4/5 | -0.0526 pp | +0.0869 pp | No |
| Promoted complete-identity vote | 81.7403% | — | — | — | — | incumbent |
| Learned-numeric forest vote | 81.7003% | -0.0400 pp | 1/5 | -0.1052 pp | -0.4054 pp | No |
| Learned-numeric representation bag | 81.6919% | -0.0484 pp | 1/5 | -0.1473 pp | -0.2026 pp | No |
| Learned-numeric XGBoost vote | 81.6667% | -0.0737 pp | 1/5 | -0.1263 pp | -0.2605 pp | No |
| Learned-numeric full vote | 81.6288% | -0.1115 pp | 1/5 | -0.2630 pp | -0.4053 pp | No |
| Learned-numeric XGBoost | 80.9996% | -0.0147 pp vs original XGBoost | — | — | — | diagnostic |
| Original XGBoost | 81.0143% | — | — | — | — | diagnostic |
| Learned-numeric Random Forest | 80.6418% | +0.0505 pp vs original forest | — | — | — | diagnostic |
| Original Random Forest | 80.5913% | — | — | — | — | diagnostic |

The small standalone forest improvement does not survive voting. The full
replacement also breaches the worst-fold guard at -0.2630 points. This is
consistent with zero `amount_tsh` carrying useful acquisition or operating
context that a plausible reconstructed value partially obscures, despite the
retained flag. The evidence does not justify tuning the auxiliary regressor.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_learned_numeric_imputation_screen.py
```

Runtime evidence remains ignored under
`.runtime/learned-numeric-imputation-screen/`.

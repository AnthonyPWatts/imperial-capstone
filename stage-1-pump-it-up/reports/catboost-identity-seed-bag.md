# Complete-identity CatBoost seed bag

## Decision

Retain the single promoted identity CatBoost seed. Averaging a second fixed seed
equally inside the existing 20% identity allocation lowers accuracy from
81.7403% to **81.7193%**, a **-0.0210 percentage-point** change with only **1/5**
fold win. No additional seeds or blend weights are screened.

No local-test or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether variance reduction strengthens the promoted identity voter. |
| 2. Gather the data | Reuse the unchanged development data and six deferred identities. |
| 3. Explore the data | Reuse the promoted CatBoost's OOF probabilities as the first seed. |
| 4. Clean and preprocess the data | Apply the same deterministic feature engineering and identity normalisation. |
| 5. Select and engineer features | No feature change. |
| 6. Define the machine-learning task | Unchanged three-class classification. |
| 7. Partition the data | Reuse the same five frozen outer folds. |
| 8. Select and train candidate methods | Fit one extra depth-8 CatBoost using seed 20260822. |
| 9. Evaluate and interpret the results | Compare the second seed and equal seed bag inside the 44:36:20 vote. |
| 10. Deploy and iterate | Reject seed expansion and retain seed 20260821 alone. |

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Promoted identity vote | **81.7403%** | — | — | — | — | incumbent |
| Equal-seed-bag identity vote | 81.7193% | -0.0210 pp | 1/5 | -0.0737 pp | -0.1159 pp | No |
| Second-seed identity vote | 81.7130% | -0.0274 pp | 2/5 | -0.1578 pp | -0.4054 pp | No |
| Equal identity CatBoost seed bag | 80.6923% | -1.0480 pp | 0/5 | -1.6204 pp | -6.0804 pp | No |
| First-seed identity CatBoost | 80.6229% | -1.1174 pp | 0/5 | -1.5888 pp | -5.8489 pp | No |
| Second-seed identity CatBoost | 80.5934% | -1.1469 pp | 0/5 | -1.6519 pp | -5.7037 pp | No |

The standalone seeds disagree on 4.057% of hard predictions. That diversity is
not sufficiently useful: the second seed is weaker, and the equal probability
average improves standalone CatBoost but slightly reduces the final ensemble.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_catboost_identity_seed_bag.py
```

Runtime evidence remains ignored under `.runtime/catboost-identity-seed-bag/`.

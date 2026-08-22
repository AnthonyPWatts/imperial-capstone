# CatBoost deferred-identity screen

## Decision

Promote one candidate to frozen local-test confirmation: 44% accepted XGBoost,
36% accepted Random Forest and 20% depth-8 CatBoost trained with six additional
native categorical identities. It reaches **81.740%** mean accuracy, **+0.116
percentage points** against the accepted ensemble, wins **4/5** folds, loses at
most **0.084 points** on one fold and reduces repair recall by **0.926 points**.
It passes every predeclared promotion condition.

This is the first promotion-gate pass from the regional, expanded-voter,
binary, fuzzy, outlier, explicit target-encoding and spatial-outcome loops. It
is still far short of a full percentage-point gain. No local-test or competition
prediction was generated during selection.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether CatBoost's ordered categorical statistics can use deferred identities more safely than explicit target encoding. |
| 2. Gather the data | Reuse the six audited source fields and frozen development partition. |
| 3. Explore the data | Reuse prior level, singleton, unseen and explicit target-encoding evidence. |
| 4. Clean and preprocess the data | Apply conservative case and whitespace normalisation while preserving missing, blank and literal sentinel states. |
| 5. Select and engineer features | Add `funder`, `installer`, `wpt_name`, `subvillage`, `ward` and `scheme_name` as six native categoricals to the accepted engineered frame. |
| 6. Define the machine-learning task | Unchanged supervised three-class classification. |
| 7. Partition the data | Retain the five frozen outer folds and CatBoost's inner early-stopping split. |
| 8. Select and train candidate methods | Fit one depth-8 CatBoost specification; do not open a depth or ordered-statistics grid. |
| 9. Evaluate and interpret the results | Compare standalone CatBoost and four fixed low-weight additions using the unchanged gate. |
| 10. Deploy and iterate | Freeze the passing 20% recipe for one local-test confirmation, then refit only if that confirmation is treated as reporting rather than further tuning. |

## Representation

The existing native CatBoost candidate receives the accepted 29 engineered
features, including fifteen categorical fields. The new candidate adds all six
deferred identities, producing 35 engineered fields and 21 native categoricals.
CatBoost may construct ordered target statistics and category interactions
without explicit full-data target means. The depth-8 model, seed, loss,
early-stopping design and other settings remain unchanged.

Reusable GPU evaluation now accepts an optional CatBoost feature-engineering
callback. The default callback is unchanged, so earlier candidates retain their
representation. Tests confirm the complete identity list, normalisation,
missing states, category order and evaluator plumbing.

## Results

| Candidate | Accuracy | Change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **20% identity CatBoost** | **81.740%** | **+0.116 pp** | **4/5** | **-0.084 pp** | **-0.926 pp** | **Yes** |
| 15% identity CatBoost | 81.723% | +0.099 pp | 4/5 | -0.105 pp | -0.782 pp | No |
| 10% identity CatBoost | 81.646% | +0.021 pp | 3/5 | -0.074 pp | -0.666 pp | No |
| Accepted ensemble | 81.625% | — | — | — | — | — |
| 5% identity CatBoost | 81.595% | -0.029 pp | 0/5 | -0.053 pp | -0.145 pp | No |
| Complete-identity CatBoost alone | 80.623% | -1.002 pp | 0/5 | -1.410 pp | -6.775 pp | No |
| Current-feature CatBoost alone | 80.284% | -1.340 pp | 0/5 | -1.831 pp | -5.299 pp | No |

Complete identities improve CatBoost standalone by 0.339 points, but it remains
weaker than either accepted component. Its value is complementary error, not
replacement strength. The selected blend weights are exactly 44%, 36% and 20%:
the accepted 55:45 ratio is preserved inside the remaining 80% probability
mass. Selected CatBoost iteration counts are 2,106, 2,062, 2,239, 2,173 and
2,277 across the five folds.

## Next checkpoint

Freeze the recipe now. Refit the three components on the complete development
partition and compare it with an identically refitted accepted 55:45 recipe on
the already-reserved local test exactly once. Do not change weights, fields or
CatBoost settings in response to that result. The local test has been used by
earlier workflows, so it is confirmation evidence rather than a pristine final
estimate.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_catboost_identity_screen.py
```

Runtime evidence remains ignored under `.runtime/catboost-identity-screen/`.

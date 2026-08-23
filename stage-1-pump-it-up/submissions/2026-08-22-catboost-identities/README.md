# Complete-identity CatBoost candidate

## Status

Prepared and validated on 22 August 2026, then submitted second on 23 August.
The recipe was frozen before its one-time local-test confirmation and was not
tuned from competition prediction shares.

## Evidence and recipe

| Evidence | Accuracy | Change from accepted | Fold wins |
| --- | ---: | ---: | ---: |
| Five-fold development | 81.740% | +0.116 pp | 4/5 |
| Reserved local test | 81.086% | +0.387 pp | — |

The probability vote is:

- 44% depth-8, child-weight-1 XGBoost over the accepted one-hot features;
- 36% accepted Random Forest;
- 20% depth-8 CatBoost over the accepted features plus native `funder`,
  `installer`, `wpt_name`, `subvillage`, `ward` and `scheme_name` identities.

The full-data refit uses the median fold counts: 1,014 XGBoost trees and 2,173
CatBoost trees. Existing validated full-data XGBoost and Random Forest
probabilities were reused; reconstructing their 55:45 accepted vote reproduced
the existing accepted submission exactly.

## Candidate file

`01-44-xgboost-depth-8-child-1-36-random-forest-20-catboost-complete-identities.csv`
contains 14,850 unique IDs in template order. Prediction shares are 60.788%
`functional`, 3.623% `functional needs repair` and 35.589% `non functional`.
It disagrees with the accepted 55:45 submission on 1.576% of rows.

- SHA-256: `76cf24053f136d89262b23b626c932e2f5b609ca197e16d2cb7e4b46897bb006`
- Status: `submitted_public_scored`
- Public score: **0.8259**

`manifest.json` records the recipe, evidence, timings, class shares and hash.
`accepted-to-candidate-transitions.csv` records all label changes.

## Reproduction

From `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\generate_catboost_identity_submission.py
```

The script refuses a changed candidate CSV and requires the reconstructed
accepted predictions to match the previously generated accepted submission.

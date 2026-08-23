# Deep-XGBoost archive-substitution candidate

## Status

Prepared and validated on 23 August 2026 for the remaining daily submission
slot. It has not been uploaded.

## Evidence and recipe

| Evidence | Accuracy | Change from submitted archive |
| --- | ---: | ---: |
| Five-fold development | 81.8771% | +0.0842 pp |
| Used local test | 81.3552% | +0.1684 pp |

The candidate keeps five submitted archive components and their weights. It
replaces only the 33% accepted depth-8 XGBoost allocation with the previously
screened seed-20260822 depth-17 XGBoost over 300 fold-fitted top-common
identity indicators. The model uses exactly 600 trees. Nothing was retuned
after local confirmation.

## Candidate file

`01-33-deep-top50-xgboost-11-spatial-xgboost-18-random-forest-09-frequency-forest-09-spatial-forest-20-identity-catboost.csv`
contains 14,850 unique IDs in template order. It differs from the submitted
archive synthesis on 237 rows (1.5960%). Prediction shares are 60.694%
`functional`, 3.684% `functional needs repair` and 35.623% `non functional`.

- SHA-256: `ef9b4e0fa2c696ae5a203f25cd9b2c4ca4e4e5d83aa4e010595e39af60b7a1d2`
- Status: `prepared_not_uploaded`

`manifest.json` records the recipe, selection evidence, timings, class shares
and hash. The generated CSV remains ignored because it contains competition
test identifiers.

## Reproduction

From `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_deep_archive_confirmation.py
..\.venv\Scripts\python.exe .\scripts\generate_deep_archive_submission.py
```

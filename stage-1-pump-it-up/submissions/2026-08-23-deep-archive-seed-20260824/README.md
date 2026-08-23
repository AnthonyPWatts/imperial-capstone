# Deep-archive seed-20260824 submission

## Status

Prepared, validated and submitted unchanged in the remaining daily slot on
23 August 2026. It scored **0.8298** publicly and was observed at leaderboard
position **2**, 0.0001 behind the leader. The rank is time-specific and may
change as other competitors submit.

## Evidence and recipe

| Evidence | Accuracy | Change from seed 20260822 |
| --- | ---: | ---: |
| Five-fold development | 81.8981% | +0.0210 pp |
| Used local test | 81.3973% | +0.0421 pp |

The candidate changes only the primary deep XGBoost seed from 20260822 to
20260824. It retains depth 17, 600 trees, the top-50 identity representation
and the existing 33:11:18:9:9:20 component weights.

## Candidate file

`01-deep-archive-seed-20260824.csv` contains 14,850 unique IDs in template
order. It differs from the seed-20260822 candidate on 75 rows (0.5051%).
Prediction shares are 60.721% `functional`, 3.643% `functional needs repair`
and 35.636% `non functional`.

- SHA-256: `fe5de9ea46bad2b35226bc97ebdfb743807fb758df1d259e8f8a51609808fee2`
- Public score: `0.8298`
- Observed public rank: `2` on 23 August 2026
- Status: `submitted_public_scored`

`manifest.json` records the recipe, selection evidence, timings, class shares
and hash. The generated CSV remains ignored because it contains competition
test identifiers.

## Reproduction

From `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_deep_seed_20260824_confirmation.py
..\.venv\Scripts\python.exe .\scripts\generate_deep_archive_submission.py --seed 20260824
```

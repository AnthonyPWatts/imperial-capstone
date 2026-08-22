# Recording-time feature screen

## Decision

Retain the single elapsed-days recording feature. Adding recording year, month
and year-month categories reduces both accepted components. The promoted
identity vote falls from 81.7403% to **81.7109%**, a **-0.0295 percentage-point**
change with only **2/5** fold wins and 0.550 points lower repair recall.

No calendar-bin or seasonal follow-up is justified. No local-test or
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test the audited but previously untested recording year and month recommendation. |
| 2. Gather the data | Reuse complete source `date_recorded` values. |
| 3. Explore the data | Reuse the audit showing a 2002–2013 survey span and recording-time target differences. |
| 4. Clean and preprocess the data | Parse dates strictly and retain the accepted elapsed-days value. |
| 5. Select and engineer features | Add zero-padded year, month and year-month categorical fields as one fixed policy. |
| 6. Define the machine-learning task | Unchanged three-class classification. |
| 7. Partition the data | Fit rare-category handling and all models separately inside frozen folds. |
| 8. Select and train candidate methods | Refit accepted XGBoost and Random Forest specifications only. |
| 9. Evaluate and interpret the results | Compare components, their 55:45 vote and the 44:36:20 identity vote. |
| 10. Deploy and iterate | Reject explicit calendar categories without further bin tuning. |

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Promoted identity vote | **81.7403%** | — | — | — | — | incumbent |
| Temporal identity vote | 81.7109% | -0.0295 pp | 2/5 | -0.0842 pp | -0.5502 pp | No |
| Temporal 55:45 vote | 81.5278% | -0.2125 pp | 0/5 | -0.3051 pp | +0.7238 pp | No |
| Accepted XGBoost | 81.0143% | -0.7260 pp | 0/5 | -0.8733 pp | -2.4034 pp | No |
| Temporal XGBoost | 80.9028% | -0.8375 pp | 0/5 | -1.0101 pp | -2.3746 pp | No |
| Accepted Random Forest | 80.5913% | -1.1490 pp | 0/5 | -1.2311 pp | +2.9236 pp | No |
| Temporal Random Forest | 80.5450% | -1.1953 pp | 0/5 | -1.4415 pp | +2.7212 pp | No |

The three fields expand the engineered frame from 29 to 32 features. XGBoost
selects 1,133, 1,161, 964, 914 and 1,059 trees. Both families weakening under
the same representation is stronger evidence than a single-model miss: the
elapsed-day feature already supplies the useful ordering, while explicit
campaign categories do not generalise across the random folds.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_temporal_feature_screen.py
```

Runtime evidence remains ignored under `.runtime/temporal-feature-screen/`.

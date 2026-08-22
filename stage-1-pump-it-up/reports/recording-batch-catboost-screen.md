# Exact recording-batch CatBoost screen

## Decision

Retain the complete-identity CatBoost and the existing archive-derived
synthesis. Adding the exact recording date as one native survey-batch category
raises the promoted direct vote from 81.7403% to **81.7614%**, a small **+0.0210
percentage-point** gain with 3/5 fold wins. It does not pass the +0.10-point
promotion gate. Substituting it into the fixed archive synthesis reaches
**81.7866%**, below that synthesis's existing **81.7929%**.

No date resolution, CatBoost parameter or ensemble weight is screened. No
local-test or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether repeated recording dates act as useful survey-batch identities beyond the accepted continuous elapsed-days feature. |
| 2. Gather the data | Reuse the complete supplied `date_recorded` field and the frozen development partition. |
| 3. Explore the data | Audit exact-date support inside every outer fold before interpreting the candidate. |
| 4. Clean and preprocess the data | Parse dates under the existing strict complete-date contract and format one stable ISO date category. |
| 5. Select and engineer features | Append exact date to the six complete deferred identities while retaining elapsed days. |
| 6. Define the machine-learning task | Unchanged three-class classification and accuracy metric. |
| 7. Partition the data | Reuse the five frozen outer folds and CatBoost's nested stopping selection. |
| 8. Select and train candidate methods | Fit one depth-8 native-categorical CatBoost; evaluate direct replacement and one equal within-component bag. |
| 9. Evaluate and interpret the results | Compare against the promoted vote and make one no-refit substitution into the already fixed archive synthesis. |
| 10. Deploy and iterate | Reject promotion and close date-category variants because the best direct gain is only 0.021 points. |

## Motivation and support

DrivenData's [official solution archive](https://github.com/drivendataorg/pump-it-up)
records a 0.8264 public score for the `benedekrozemberczki` entry. Its
[data-cleaning code](https://github.com/drivendataorg/pump-it-up/blob/master/benedekrozemberczki/2_data_cleaner.R)
uses recording-year and month indicators, a weekend flag and frequent exact
dates. The earlier project screen found that broad year/month categories weaken
the accepted one-hot models; this experiment isolates the untested mechanism:
CatBoost's ordered treatment of exact, repeated survey batches.

The development partition contains 356 dates. Across the five outer folds,
only 0.042–0.095% of validation rows have a date absent from outer training,
and 1.60–2.08% have training support below 20. The category is therefore well
supported rather than a disguised row identifier.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Existing archive synthesis | **81.7929%** | +0.0526 pp | 5/5 | +0.0105 pp | -0.2026 pp | No |
| Archive synthesis with batch CatBoost | 81.7866% | +0.0463 pp | 3/5 | -0.0316 pp | -0.1739 pp | No |
| Archive synthesis with equal identity/batch CatBoost | 81.7761% | +0.0358 pp | 3/5 | -0.0210 pp | -0.2606 pp | No |
| Recording-batch direct vote | 81.7614% | +0.0210 pp | 3/5 | -0.0737 pp | -0.1161 pp | No |
| Equal identity/recording-batch direct bag | 81.7466% | +0.0063 pp | 3/5 | -0.0210 pp | -0.1159 pp | No |
| Promoted complete-identity vote | 81.7403% | — | — | — | — | incumbent |
| Complete-identity CatBoost | 80.6229% | — | — | — | — | diagnostic |
| Recording-batch CatBoost | 80.5787% | -0.0442 pp vs identity CatBoost | — | — | — | diagnostic |

The exact date changes useful decisions in the direct ensemble, despite making
the standalone CatBoost 0.044 points weaker. The equal CatBoost bag dilutes most
of that small gain, and the stronger archive-derived representations already
recover the same structure more effectively. Further calendar bins or batch
weights would tune a weak effect on heavily reused folds, so the branch stops.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_recording_batch_catboost_screen.py
```

Runtime evidence remains ignored under
`.runtime/recording-batch-catboost-screen/`.

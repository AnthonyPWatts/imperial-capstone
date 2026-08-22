# Pump-age cohort screen

## Decision

Retain the promoted 44:36:20 identity vote. A fixed categorical pump-age
cohort produces the strongest subsequent development result, **81.8035%**, but
its **+0.0631 percentage-point** gain remains below the predeclared +0.10-point
promotion gate. It wins **3/5** folds, its worst fold changes by only -0.0316
points, and repair recall changes by -0.1740 points.

The cohort cut points and component weights are not tuned. No local-test or
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether broad maintenance-lifecycle cohorts complement continuous pump age. |
| 2. Gather the data | Reuse recording date and construction year. |
| 3. Explore the data | Confirm support for six fixed age bands plus unknown and inconsistent states. |
| 4. Clean and preprocess the data | Preserve accepted continuous age, missing flag and inconsistent flag. |
| 5. Select and engineer features | Add one categorical cohort: 0–2, 3–5, 6–10, 11–20, 21–30 or 31+ years. |
| 6. Define the machine-learning task | Unchanged three-class classification. |
| 7. Partition the data | Fit category handling and all models inside the frozen folds. |
| 8. Select and train candidate methods | Refit accepted XGBoost, Random Forest and complete-identity CatBoost specifications. |
| 9. Evaluate and interpret the results | Test the fixed cohort in the global trees, identity CatBoost and all components. |
| 10. Deploy and iterate | Retain the incumbent because the strongest placement misses the promotion gate. |

## Fixed representation

Age is `recording year - construction year`, using the accepted validity
rules. The categorical states and full labelled-row support are:

| Cohort | Rows |
| --- | ---: |
| 0–2 | 5,020 |
| 3–5 | 6,610 |
| 6–10 | 5,627 |
| 11–20 | 9,456 |
| 21–30 | 5,633 |
| 31+ | 6,336 |
| Unknown construction year | 20,709 |
| Inconsistent construction year | 9 |

The nine inconsistent rows are handled by the existing fold-fitted rare
category policy.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cohort in all three components | **81.8035%** | +0.0631 pp | 3/5 | -0.0316 pp | -0.1740 pp | No |
| Cohort in global trees only | 81.7719% | +0.0316 pp | 4/5 | -0.0631 pp | -0.3475 pp | No |
| Promoted identity vote | 81.7403% | — | — | — | — | incumbent |
| Cohort in identity CatBoost only | 81.7193% | -0.0210 pp | 2/5 | -0.0631 pp | -0.1160 pp | No |
| Cohort 55:45 tree vote | 81.6183% | -0.1221 pp | 0/5 | -0.2104 pp | +0.7234 pp | No |
| Cohort XGBoost | 80.9238% | -0.8165 pp | 0/5 | -0.9891 pp | -2.3745 pp | No |
| Cohort identity CatBoost | 80.5324% | -1.2079 pp | 0/5 | -1.7151 pp | -5.5885 pp | No |
| Cohort Random Forest | 80.4503% | -1.2900 pp | 0/5 | -1.4415 pp | +3.0684 pp | No |

Every refitted component weakens in isolation. The small ensemble gain therefore
comes from changed error complementarity rather than a stronger base learner.
That makes it unsuitable for promotion after repeated reuse of the same five
folds, despite the favourable fold stability.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_age_cohort_screen.py
```

Runtime evidence remains ignored under `.runtime/age-cohort-screen/`.

# CatBoost context-qualified identity screen

## Decision

Retain the complete-identity CatBoost candidate unchanged. Adding three
predeclared, well-supported context pairs produces **exactly the same 81.7403%
ensemble accuracy**, wins only **2/5** folds against the complete-identity vote
and reduces repair recall by another **0.463 percentage points**. The context
CatBoost standalone is 0.141 points weaker than the complete-identity model.

No local-test or competition prediction was generated. Stop explicit identity-
context construction; CatBoost's existing pairwise categorical interaction
setting already receives the constituent categories.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether explicit context can disambiguate repeated native identities. |
| 2. Gather the data | Reuse the six source identities and accepted LGA field. |
| 3. Explore the data | Audit support and competition unseen exposure before declaring composites. |
| 4. Clean and preprocess the data | Reuse conservative case and whitespace normalisation with explicit missing and blank states. |
| 5. Select and engineer features | Add only LGA×ward, LGA×scheme and funder×installer; exclude sparse pump-name and subvillage contexts. |
| 6. Define the machine-learning task | Unchanged three-class classification. |
| 7. Partition the data | Preserve the five frozen outer folds and nested stopping split. |
| 8. Select and train candidate methods | Fit one depth-8 CatBoost; preserve the fixed 20% ensemble contribution. |
| 9. Evaluate and interpret the results | Compare directly with the promoted complete-identity CatBoost and its 20% vote. |
| 10. Deploy and iterate | Reject the tied context representation and retain the already-prepared complete-identity candidate. |

## Pre-fit support audit

| Proposed identity | Labelled levels | Singleton row share | Competition unseen share | Labelled rows in groups ≥20 |
| --- | ---: | ---: | ---: | ---: |
| LGA × ward | 2,191 | 0.064% | 0.081% | 81.788% |
| Funder × installer | 3,526 | 3.160% | 3.104% | 84.660% |
| LGA × scheme name | 2,794 | 1.093% | 1.098% | 79.104% |
| LGA × subvillage | 24,543 | 20.965% | 21.387% | 5.377% |
| LGA × pump name | 44,292 | 67.352% | 68.640% | 10.569% |

The first three were fixed before model fitting. The last two were excluded;
qualifying already sparse names with LGA would mostly make them less reusable.
The candidate contains 38 engineered fields and 24 native categorical fields.

## Results

| Candidate | Accuracy | Change from identity leader | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Complete-identity 20% vote | **81.7403%** | — | — | — | — | incumbent |
| Context-identity 20% vote | **81.7403%** | 0.0000 pp | 2/5 | -0.1052 pp | -0.4634 pp | No |
| Complete-identity CatBoost | 80.6229% | -1.1174 pp | 0/5 | -1.5888 pp | -5.8489 pp | No |
| Context-identity CatBoost | 80.4819% | -1.2584 pp | 0/5 | -1.8203 pp | -6.4568 pp | No |

The context model selected 1,875, 2,305, 2,045, 2,199 and 1,677 trees across
the five folds. Its lower standalone result and unchanged ensemble mean give no
case for a local-test confirmation or alternative context subset screen.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_catboost_identity_context_screen.py
```

Evidence is cached under `.runtime/catboost-identity-context-screen/`.

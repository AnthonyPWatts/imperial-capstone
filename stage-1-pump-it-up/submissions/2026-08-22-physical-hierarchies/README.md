# 2026-08-22 physical-hierarchy candidates

Three exploratory entries were prepared from the bounded physical-category
hierarchy screen. Source plus source class and the cross-policy hybrid were
uploaded on 22 August; waterpoint type plus group remains local. The reserved
local test was not opened, and the competition rows were used only for
full-data prediction.

All entries retain the accepted 55% XGBoost and 45% Random Forest probability
blend. XGBoost uses the fixed depth-8, minimum-child-weight-1 configuration.
Each model's preprocessing was fitted only with its training rows during the
five-fold comparison and was refitted on all 59,400 labelled rows for the
competition predictions.

## Selection evidence

The accepted baseline scored 81.625% mean accuracy on the five frozen
development folds. None of these small hierarchy changes passed the
predeclared promotion gate: at least +0.10 percentage points, three fold wins,
no fold worse than -0.25 percentage points, and no material repair-recall
loss. They are therefore clearly labelled exploratory entries rather than
promoted model replacements.

| Entry | Mean accuracy | Change | Fold wins | Worst fold | Repair recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source plus source class | 81.635% | +0.011 pp | 2/5 | -0.105 pp | 34.859% |
| Waterpoint type plus group | 81.627% | +0.002 pp | 2/5 | -0.084 pp | 35.032% |
| Waterpoint-aware XGBoost plus source-class Random Forest | 81.629% | +0.004 pp | 2/5 | -0.095 pp | 35.004% |

The third entry is a no-refit cross of the best hierarchy-specific components:
waterpoint-both XGBoost and source-plus-class Random Forest. The other two use
the same hierarchy policy for both components.

## Prepared files

Each CSV was reloaded after writing and contains exactly 14,850 unique IDs in
submission-template order, the two required columns, no missing values and only
the three permitted labels. CSV prediction rows remain ignored by Git; the
manifest and hashes identify the local files.

| Entry | Prediction shares (functional / repair / non-functional) | Public score | XGBoost trees | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `01-source-plus-class.csv` | 60.296% / 3.933% / 35.771% | **0.8246** | 1,024 | `95569ad19bec46c94089207db06d4bb240fc27c3ccb40d308aff39535ecdbbf8` |
| `02-waterpoint-both-levels.csv` | 60.323% / 3.973% / 35.704% | Not submitted | 1,031 | `74daa161b5f1f9d8e2dff72808664158f166d21a3745959c74a7649935e98baa` |
| `03-waterpoint-xgb-source-rf.csv` | 60.350% / 3.973% / 35.677% | 0.8244 | 1,031 | `7457d578fbe59e46cabbe3de2b3bd92590829b4077180548e959ea354e32ced3` |

Competition hard-label disagreement is 1.071% between entries 1 and 2, 0.620%
between entries 1 and 3, and 0.788% between entries 2 and 3. Full machine-
readable provenance is in [`manifest.json`](manifest.json).

The two public scores are stronger than the previous 0.8241 leader, but the
differences are small and the frozen-fold promotion gate was not passed. The
leaderboard result records useful external evidence without retrospectively
changing the feature-selection decision.

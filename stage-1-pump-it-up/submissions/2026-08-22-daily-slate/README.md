# 2026-08-22 three-entry slate

These three files formed the bounded submission slate uploaded on 22 August
2026.

| Order | Entry | Evidence | Public score | Prediction shares (functional / repair / non-functional) |
| ---: | --- | --- | ---: | --- |
| 1 | Source plus source class | 81.635% five-fold mean; +0.011 pp versus accepted recipe | **0.8246** | 60.296% / 3.933% / 35.771% |
| 2 | Waterpoint-aware XGBoost plus source-class Random Forest | 81.629% five-fold mean; +0.004 pp versus accepted recipe | 0.8244 | 60.350% / 3.973% / 35.677% |
| 3 | Frozen 2.5× repair-count depth-6/7/8 XGBoost bag | 80.328% local-test accuracy; 44.496% repair recall | 0.8174 | 57.926% / 7.104% / 34.970% |

Entries 1 and 2 are exploratory hierarchy challengers; neither passed the
predeclared promotion gate. Entry 3 is the strongest accuracy/repair trade-off
from the already-completed oversampling replays. Against its original-training
counterpart it lost 0.429 percentage points of local-test accuracy while gaining
11.935 points of repair recall. It is included as a deliberately different
class-distribution experiment, not because nominal accuracy evidence favours it.

Source plus source class is the new project best, improving the prior 0.8241
leader by 0.0005. The hybrid finished 0.0002 behind it. The oversampled bag
finished 0.0066 behind its unoversampled 0.8240 counterpart, confirming that
the measured repair-recall gain came with a material accuracy cost rather than
revealing a leaderboard improvement.

No new oversampling level was generated, no model was retuned, and the local
test was not reopened while assembling this slate. The earlier 2.5× result was
adaptive to prior oversampling results, so its evidence is exploratory rather
than a fresh confirmation.

## Files

1. [`01-source-plus-class.csv`](../2026-08-22-physical-hierarchies/01-source-plus-class.csv)
   - SHA-256: `95569ad19bec46c94089207db06d4bb240fc27c3ccb40d308aff39535ecdbbf8`
2. [`03-waterpoint-xgb-source-rf.csv`](../2026-08-22-physical-hierarchies/03-waterpoint-xgb-source-rf.csv)
   - SHA-256: `7457d578fbe59e46cabbe3de2b3bd92590829b4077180548e959ea354e32ced3`
3. [`03-60-xgboost-depth-6-7-8-local-bag-40-random-forest-oversampled.csv`](../2026-08-21-oversampled-replay-250x/03-60-xgboost-depth-6-7-8-local-bag-40-random-forest-oversampled.csv)
   - SHA-256: `7f113c60c272db2e08ed12f9051637f6b448790af5359f85fa0857bfcc17a53f`

All three were independently reloaded and checked for exactly 14,850 unique
IDs in submission-template order, the two required columns, valid labels and no
missing values. The machine-readable selection record is in
[`manifest.json`](manifest.json).

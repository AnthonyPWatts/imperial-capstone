# Model-search conclusion

## Decision

**Final 23 August update:** a bounded follow-up replaced the prepared deep
candidate's seed 20260822 with the already-predeclared seed 20260824. The
unchanged 600-tree recipe reaches **81.8981%** on development and **81.3973%**
on the used local test, adding ten and five net correct rows. Exact-date,
residual-voter and inner-stopping alternatives were rejected. The replacement
CSV was submitted unchanged in the third daily slot and scored **0.8298**
publicly. It was observed at leaderboard position **2**, 0.0001 behind the
leader. See the [deep follow-up report](deep-follow-up-search.md).

**Earlier 23 August update:** the held final submission slot was explicitly
reopened for the strongest recorded near-miss. The unchanged single-seed
depth-17 XGBoost archive substitution confirmed at **81.3552%** on the used
local test, **+0.1684 percentage points** and 20 net correct rows above the
submitted archive. A validated seed-20260822 full-data candidate was prepared
but not uploaded, then superseded by the seed-20260824 replacement. See the
[deep-archive confirmation](deep-archive-confirmation.md). The conclusion
below records the earlier stopping decision and why the candidate had
originally been held.

Conclude the bounded Pump It Up model search with the seed-20260824
deep-archive synthesis as the selected competition model. It scored
**0.8298** publicly, exceeded the 0.826 project target and was observed at
public-leaderboard position **2** on 23 August 2026. The rank is a time-specific
observation and may change as other competitors submit.

[![DrivenData Pump It Up public leaderboard showing anthonypwatts at rank 2 with a score of 0.8298](../../assets/pump-it-up-public-leaderboard-rank-2.png)](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/leaderboard/)

*Public leaderboard observed on 23 August 2026; the linked live ranking may
change.*

The original aspiration was approximately one additional percentage point of
frozen validation accuracy. That exact magnitude was not reached: the selected
model improves the accepted 81.6246% development result to **81.8981%**, an
increase of **0.2735 percentage points**. The search nevertheless delivered a
reproducible competition improvement, coherent development/local/public
ordering and an observed second-place public result. Further selection on the
repeatedly used folds now carries more overfitting risk than evidential value.

## Course-aligned lifecycle

| Step | Final position |
| --- | --- |
| 1. Define the goal and scope | Improve three-class competition accuracy through bounded, reproducible experiments while retaining repair-case interpretation. |
| 2. Gather the data | Use the supplied labelled and competition tables; external-data proposals were not used. |
| 3. Explore the data | Audit predictors, geography, class imbalance, identities, uncertainty, regional effects and representation diversity. |
| 4. Clean and preprocess the data | Keep all transformations fold-fitted; bounded training-only outlier filters were tested first and rejected when they reduced unchanged-fold accuracy. |
| 5. Select and engineer features | Retain accepted features and add only representations supported by frozen-fold evidence: complete identities, categorical occurrence support and spatial-height reconstruction. |
| 6. Define the machine-learning task | Preserve nominal three-class probabilistic classification scored by accuracy; binary-only and fuzzy-label alternatives were evaluated but rejected. |
| 7. Partition the data | Use one fixed 20% local test and five fixed stratified development folds; do not reopen either for post-result tuning. |
| 8. Select and train candidate methods | Screen tree, boosting, neural, ordinal, regional and ensemble candidates under bounded gates and fixed recipes. |
| 9. Evaluate and interpret the results | The seed-20260824 deep-archive synthesis led development at 81.8981%, the local test at 81.3973% and the public leaderboard at 0.8298. |
| 10. Deploy and iterate | Retain the validated, unchanged seed-20260824 full-data CSV as the selected model; stop adaptive modelling and record private-leaderboard performance when available. |

## Final evidence

| Measure | Earlier reference | Selected result | Change |
| --- | ---: | ---: | ---: |
| Frozen development accuracy | 81.6246% accepted vote | **81.8981%** | +0.2735 pp |
| Used local-test accuracy | 80.8333% source-plus-class leader | **81.3973%** | +0.5640 pp |
| Public score | 0.8246 former leader | **0.8298** | +0.0052 |
| Project public target | 0.8260 | **0.8298** | +0.0038 |

The preceding archive synthesis scored 81.7929% on development, 81.1869% on
the local test and 0.8288 publicly. The complete-identity companion scored
81.7403%, 81.0859% and 0.8259 respectively. Their consistent ordering provides
supporting evidence that the final improvement is not solely a leaderboard
accident, while the small margins still require caution.

## Stop rationale

The third 23 August submission slot was used only after the deep-XGBoost
replacement showed directionally consistent development and local-test gains.
The stronger of the already-predeclared seeds was selected without changing
features, depth, tree count or component weights. Its 0.8298 public score is
recorded as final outcome evidence; tuning against this result would weaken the
validation discipline without independent supporting evidence.

The exact submitted files, hashes, component weights and public scores are in
the [submission log](../submissions/README.md).

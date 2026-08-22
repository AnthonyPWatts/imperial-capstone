# Physical categorical hierarchy screen

## Decision

Retain the accepted granular representations for extraction type, source,
water quality and waterpoint type. Twenty challengers across the four families
and six bounded no-refit component hybrids were evaluated on the five frozen
development folds. None passed the accuracy-promotion gate.

Two additions finished fractionally above the accepted 81.625% vote, but the
changes were negligible and unstable: source plus source class reached 81.635%
(+0.011 percentage points) and waterpoint type plus group reached 81.627%
(+0.002 points). Both won only two folds. The strongest component hybrid also
won two folds and reached 81.629%, just +0.004 points above baseline.

Extraction group-only is retained as a named parsimony candidate rather than an
accuracy promotion. It reduced the fold-1 encoded matrix from 301 to 296
features and reached 81.593%, a 0.032-point cost with slightly higher repair
recall. The accepted competition recipe remains granular extraction type.

The labelled local test remained unopened. No competition predictions were
generated or uploaded, and no oversampling or class rebalancing participated.

## Hierarchy audit

All four families are complete deterministic mappings. None has missing values
or competition-only levels, so this experiment measures granularity and
redundancy rather than missing-data treatment.

| Family | Granular levels | Intermediate levels | Broad levels | Deterministic | Missing rows | Competition-only levels |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Extraction type | 18 | 13 | 7 | Yes | 0 | 0 |
| Source | 10 | 7 | 3 | Yes | 0 | 0 |
| Water quality | 8 | — | 6 | Yes | 0 | 0 |
| Waterpoint type | 7 | — | 6 | Yes | 0 | 0 |

The waterpoint hierarchy collapses only `communal standpipe` and `communal
standpipe multiple`. The quality hierarchy hides `fluoride abandoned` and
`salty abandoned` inside their broad chemical groups. The three-level source
class reduces source to groundwater, surface and unknown.

## Evaluation contract

Every challenger used the unchanged 55% child-weight-1 depth-8 XGBoost and 45%
Random Forest probability vote. XGBoost selected its tree count on a separate
inner stopping split within each outer-training fold before refitting. Numeric
imputation, rare-category grouping and one-hot encoding remained fold-fitted.

Each three-level family compared removal, every level alone, every two-level
combination and all three levels. Each two-level family compared removal, the
parent alone and both levels. The saved accepted baseline was reused rather
than refitted four times.

Promotion required at least +0.10 percentage points of mean accuracy, three
fold wins, no fold worse than -0.25 points and no more than a two-point
repair-recall loss. A separate parsimony flag required fewer encoded features,
no more than a 0.05-point mean loss and the same worst-fold and repair-recall
safeguards.

## Extraction hierarchy

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Granular type | **81.625%** | — | — | — | 34.859% | Retain |
| Group only | 81.593% | -0.032 pp | 2/5 | -0.168 pp | **35.032%** | Parsimony candidate |
| Type plus class | 81.576% | -0.048 pp | 1/5 | -0.105 pp | 35.004% | Reject |
| All levels | 81.576% | -0.048 pp | 1/5 | -0.168 pp | 34.888% | Reject |
| Type plus group | 81.574% | -0.051 pp | 1/5 | -0.221 pp | 34.772% | Reject |
| Group plus class | 81.540% | -0.084 pp | 2/5 | -0.168 pp | 34.627% | Reject |
| Class only | 81.402% | -0.223 pp | 1/5 | -0.347 pp | 34.656% | Reject |
| No extraction feature | 81.174% | -0.450 pp | 0/5 | -0.673 pp | 34.540% | Reject |

Extraction carries substantial independent signal. The 13-level group retains
nearly all of it, while the seven-level class loses useful pump-specific detail.
Adding deterministic parents to the granular child does not improve accuracy.

## Source hierarchy

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Source plus class | **81.635%** | +0.011 pp | 2/5 | -0.105 pp | 34.859% | Below gate |
| Granular source | 81.625% | — | — | — | 34.859% | Retain |
| Type only | 81.572% | -0.053 pp | 2/5 | -0.147 pp | **34.946%** | Reject |
| Type plus class | 81.536% | -0.088 pp | 0/5 | -0.137 pp | 34.888% | Reject |
| All levels | 81.534% | -0.091 pp | 0/5 | -0.232 pp | 34.830% | Reject |
| Source plus type | 81.505% | -0.120 pp | 1/5 | -0.263 pp | 34.541% | Reject |
| Class only | 81.479% | -0.145 pp | 1/5 | -0.295 pp | 34.280% | Reject |
| No source feature | 81.353% | -0.272 pp | 0/5 | -0.347 pp | 34.338% | Reject |

The +0.011-point source-class result is too small to distinguish from fold
noise and wins only two folds. Source type almost preserves the granular mean,
but its 0.053-point loss narrowly exceeds the parsimony tolerance.

## Quality and waterpoint hierarchies

| Family and policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Granular quality | **81.625%** | — | — | — | **34.859%** | Retain |
| Quality plus group | 81.561% | -0.063 pp | 1/5 | -0.221 pp | 34.772% | Reject |
| Quality group only | 81.538% | -0.086 pp | 1/5 | -0.168 pp | 34.395% | Reject |
| No quality feature | 81.399% | -0.225 pp | 0/5 | -0.421 pp | 34.569% | Reject |
| Waterpoint type plus group | **81.627%** | +0.002 pp | 2/5 | -0.084 pp | **35.032%** | Below gate |
| Granular waterpoint type | 81.625% | — | — | — | 34.859% | Retain |
| Waterpoint group only | 81.383% | -0.242 pp | 0/5 | -0.368 pp | 34.222% | Reject |
| No waterpoint feature | 81.048% | -0.577 pp | 0/5 | -0.747 pp | 34.309% | Reject |

The granular quality distinctions remain useful. Waterpoint form is the
strongest independent family in this screen: removing it costs 0.577 points,
and collapsing only the single-versus-multiple communal standpipe distinction
still costs 0.242 points.

## Component hybrid follow-up

Policy-specific component means suggested several bounded no-refit crosses.
Only components whose mean improved under a hierarchy policy were crossed with
the corresponding baseline or complementary improved component.

| Hybrid | Mean accuracy | Change | Fold wins | Worst fold | Repair recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Waterpoint-both XGBoost + source-class Random Forest | **81.629%** | +0.004 pp | 2/5 | -0.095 pp | **35.004%** |
| Waterpoint-both XGBoost + baseline Random Forest | 81.582% | -0.042 pp | 1/5 | -0.116 pp | 34.917% |
| Baseline XGBoost + source-class Random Forest | 81.580% | -0.044 pp | 1/5 | -0.137 pp | 34.714% |
| Extraction-group XGBoost + baseline Random Forest | 81.574% | -0.051 pp | 1/5 | -0.147 pp | 34.656% |
| Quality-group XGBoost + baseline Random Forest | 81.561% | -0.063 pp | 0/5 | -0.126 pp | 34.453% |
| Baseline XGBoost + extraction-group Random Forest | 81.559% | -0.065 pp | 2/5 | -0.242 pp | 34.830% |

The best hybrid remains 0.096 points short of the minimum promotion gain and
wins only two folds. No new feature fit, local-test access or competition
prediction was needed for this follow-up.

## Robustness and stop rule

No challenger passed the primary promotion gate, so none triggered an
LGA-disjoint sensitivity run. Running grouped robustness after failure would
create an unplanned second selection opportunity rather than confirm a promoted
candidate.

Stop tuning deterministic hierarchy combinations on these folds. Retain the
four granular features in the competition recipe; retain extraction group-only
only as a future compact-model candidate under an explicitly non-inferiority
objective. The next data-focused loop should be materially different, such as
the remaining sparse name fields or a small predeclared set of physical
interactions, rather than further hierarchy permutations.

Executable screening lives in
[`scripts/run_physical_hierarchy_screen.py`](../scripts/run_physical_hierarchy_screen.py),
with reusable transformations and evaluation in
[`src/physical_hierarchy_features.py`](../src/physical_hierarchy_features.py)
and
[`src/physical_hierarchy_evaluation.py`](../src/physical_hierarchy_evaluation.py).
Runtime models and CSV summaries remain ignored local artefacts under
`.runtime/physical-hierarchy-screen/`.

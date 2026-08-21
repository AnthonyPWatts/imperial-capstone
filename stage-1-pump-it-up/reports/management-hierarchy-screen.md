# Management hierarchy screen

## Decision

Retain the accepted `management` plus `scheme_management` representation for
the competition recipe. None of the nine bounded hierarchy challengers passed
the frozen-fold promotion gate.

`management` alone was the best challenger at 81.637%, only 0.013 percentage
points above the 81.625% baseline. It won three folds and reduced the fold-1
matrix from 301 to 289 columns, but the accuracy difference is far below the
predeclared 0.10-point threshold. Treat it as a plausible future simplification,
not as a selected accuracy improvement.

The labelled local test remained unopened. No competition predictions were
generated or uploaded, no grouped robustness run was triggered, and no
oversampling data or code participated.

## Audit evidence

The three raw fields describe related but different levels:

| Predictor | Levels | Missing rows | Largest level | Structural relationship |
| --- | ---: | ---: | ---: | --- |
| `management` | 12 | 0 | `vwc`: 40,507 | Maps deterministically to one management group |
| `management_group` | 5 | 0 | `user-group`: 52,490 | Each group contains up to four management levels |
| `scheme_management` | 12 including missing | 3,878 (6.53%) | `VWC`: 36,793 | Parallel source; not determined by management |

After conservative case and whitespace normalisation, management and scheme
labels match in 49,336 rows (83.1%), disagree in 6,186 (10.4%) and have missing
scheme management in 3,878 (6.5%). The training frame contains 94 observed
management/scheme pairs. Competition contains 66 pairs and only one row whose
pair is unseen in training; fold-fitted rare handling covers that case.

The screen compares:

- each source alone;
- the deterministic group alone;
- management or scheme paired with the group;
- all three fields;
- the normalised management/scheme pair alone or added to baseline;
- a three-level matching, conflicting or missing-scheme relationship state.

No fuzzy label matching or target-informed encoding was used.

## Evaluation contract

Every policy used the unchanged 55% child-weight-1 depth-8 XGBoost and 45%
Random Forest probability vote. XGBoost selected its tree count on a separate
inner stopping split within each outer-training fold before refitting. All
preprocessing and rare-category handling were fitted inside the current
training partition.

The primary comparison used the five frozen stratified development folds. A
challenger had to improve mean accuracy by at least 0.10 percentage points, win
at least three folds, lose no fold by more than 0.25 points and avoid reducing
repair recall by more than 2 points.

## Frozen-fold screen

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall | Fold-1 transformed features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Management only | **81.637%** | +0.013 pp | 3/5 | -0.126 pp | 34.830% | 289 |
| Accepted management plus scheme | 81.625% | — | — | — | 34.859% | 301 |
| Management plus group | 81.610% | -0.015 pp | 2/5 | -0.126 pp | 34.772% | 294 |
| Management group only | 81.593% | -0.032 pp | 2/5 | -0.189 pp | 34.569% | 282 |
| Baseline plus composite | 81.578% | -0.046 pp | 2/5 | -0.210 pp | 34.830% | 344 |
| Management-scheme composite only | 81.572% | -0.053 pp | 2/5 | -0.179 pp | 34.975% | 320 |
| Scheme management only | 81.557% | -0.067 pp | 1/5 | -0.189 pp | 34.743% | 289 |
| Scheme management plus group | 81.553% | -0.072 pp | 1/5 | -0.137 pp | 34.714% | 294 |
| Baseline plus relationship state | 81.549% | -0.076 pp | 1/5 | -0.189 pp | 34.975% | 304 |
| Management, scheme and group | 81.496% | -0.128 pp | 0/5 | -0.274 pp | 34.888% | 306 |

The complete `management` source is more useful than the incomplete scheme
source or its deterministic coarse group. The group preserves most signal by
itself, but loses useful within-group distinctions. Adding it alongside the
granular field provides equivalent split opportunities rather than new
information; adding all three is the worst policy.

Explicit source interactions do not help. The 94-level composite adds 43
fold-1 columns to baseline and reduces accuracy. The three-state relationship
is cheaper but also weaker, so disagreement is not an independently useful
feature for the accepted trees.

For management-only, XGBoost rises from 81.014% to 81.031% while Random Forest
falls from 80.591% to 80.558%. Its fold changes are +0.032, -0.063, +0.116,
+0.105 and -0.126 points. This is consistent with statistical noise around a
slightly simpler representation, not a stable accuracy gain.

## Stop rule and next step

The accepted management representation remains frozen. Do not tune groupings,
pair thresholds or fuzzy aliases against these folds. If a future deployment
prioritises a smaller matrix over the last uncertain basis point, management-
only is the evidence-backed simplification candidate and should be assessed
under a separately declared non-inferiority rule.

The next hierarchy loop should move to the physical categorical families,
starting with `extraction_type` versus `extraction_type_group` and
`extraction_type_class`, then source, quality and waterpoint hierarchies.

Executable screening lives in
[`scripts/run_management_screen.py`](../scripts/run_management_screen.py),
with reusable policies in
[`src/management_features.py`](../src/management_features.py) and evaluation in
[`src/management_evaluation.py`](../src/management_evaluation.py). Runtime
models and CSV summaries remain ignored local artefacts under
`.runtime/management-screen/`.

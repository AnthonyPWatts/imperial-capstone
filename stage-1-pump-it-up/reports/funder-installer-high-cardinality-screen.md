# Funder and installer high-cardinality screen

## Decision

Retain the accepted 29-feature policy without `funder` or `installer`. None of
the ten bounded organisation-feature challengers passed the frozen-fold
promotion gate.

Fold-fitted funder frequency was the only challenger above baseline, reaching
81.656% mean accuracy against 81.625%. Its 0.032 percentage-point gain was too
small and unstable: it won two folds, tied one and lost two. Organisation
identity through rare-grouped one-hot columns did not help, whether funder and
installer were introduced separately or together.

The labelled local test remained unopened. No competition predictions were
generated or uploaded, and no oversampling data or code participated in this
experiment.

## Audit evidence and treatment boundary

Conservative case and whitespace normalisation leaves `funder` at 1,897 levels
and reduces `installer` from 2,146 raw levels to 1,919 normalised levels. It does
not fuzzy-match aliases or reinterpret literal sentinels. Source blanks,
pandas missing values and strings such as `0`, `none`, `unknown`, `not known`,
`-` and `unknown installer` are not merged with those literal sentinels. The
transformer can distinguish an explicit blank string from a pandas missing
value; the normal CSV loader parses source-empty fields as pandas missing rather
than inventing a distinction that is absent from the loaded frame.

Both columns have long tails:

| Predictor | Normalised levels | Singleton levels | Rows in levels below 20 | Competition rows with unseen levels |
| --- | ---: | ---: | ---: | ---: |
| `funder` | 1,897 | 974 | 4,973 (8.4%) | 254 (1.710%) |
| `installer` | 1,919 | 963 | 5,041 (8.5%) | 233 (1.569%) |

The normalised values are exactly equal in 37.9% of labelled rows, so the
fields are associated but not duplicates. That supports explicit individual
and joint comparisons rather than assuming they are interchangeable.

Every learned mapping was fitted inside the current training partition:

- rare grouping used a separate one-hot encoder with a 20-row minimum;
- frequency was training-partition count divided by training-partition rows;
- unseen validation values mapped to frequency zero;
- one bounded joint sensitivity raised the rare threshold from 20 to 50;
- no target encoding, fuzzy aliasing, hashing or unrestricted raw one-hot
  representation was used.

## Evaluation contract

Every policy used the unchanged 55% child-weight-1 depth-8 XGBoost and 45%
Random Forest probability vote. XGBoost selected its tree count on a separate
inner stopping split within each outer-training fold before refitting. The
primary comparison used the five frozen stratified development folds.

A challenger had to improve mean accuracy by at least 0.10 percentage points,
win at least three folds, lose no fold by more than 0.25 points and avoid
reducing repair recall by more than 2 points.

## Frozen-fold screen

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall | Fold-1 transformed features |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Funder frequency | **81.656%** | +0.032 pp | 2/5 | -0.105 pp | 34.917% | 302 |
| Accepted feature policy | 81.625% | — | — | — | 34.859% | 301 |
| Funder rare-grouped | 81.599% | -0.025 pp | 1/5 | -0.147 pp | 34.453% | 499 |
| Funder rare-grouped plus frequency | 81.589% | -0.036 pp | 1/5 | -0.074 pp | 34.396% | 500 |
| Both frequencies | 81.574% | -0.051 pp | 2/5 | -0.263 pp | 34.453% | 303 |
| Installer rare-grouped | 81.538% | -0.086 pp | 0/5 | -0.231 pp | 34.511% | 476 |
| Both rare-grouped plus frequencies | 81.532% | -0.093 pp | 1/5 | -0.168 pp | 33.961% | 676 |
| Both rare-grouped | 81.524% | -0.101 pp | 0/5 | -0.137 pp | 34.425% | 674 |
| Both 50-row rare groups plus frequencies | 81.507% | -0.118 pp | 0/5 | -0.168 pp | 34.077% | 503 |
| Installer rare-grouped plus frequency | 81.494% | -0.130 pp | 0/5 | -0.253 pp | 34.337% | 477 |
| Installer frequency | 81.486% | -0.139 pp | 1/5 | -0.410 pp | 34.453% | 302 |

Funder frequency changed the five fold accuracies by -0.105, 0.000, +0.189,
+0.168 and -0.095 points. Both components improved slightly in their standalone
means, but their changed errors did not combine into a stable vote gain:
XGBoost moved from 81.014% to 81.056%, while Random Forest moved from 80.591%
to 80.715%.

The identity representations expanded the fold-1 matrix by roughly 175 to 375
columns without improving the complete vote. Installer frequency was also
consistently weaker than funder frequency. The evidence therefore does not
support adding either organisation identity or combining both fields.

## LGA-disjoint sensitivity

Funder frequency received one bounded robustness check because it was the best
frozen-fold challenger and organisation prevalence may proxy geography. The
same LGA-disjoint folds used by the geography screen prevent any LGA from
appearing in both training and validation.

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Accepted feature policy | 72.101% | — | — | — | 3.497% |
| Funder frequency | **72.350%** | **+0.250 pp** | **3/5** | -0.274 pp | 2.959% |

The gain is mostly associated with Random Forest: its grouped mean rises from
72.192% to 72.766%, while XGBoost rises from 71.694% to 71.736%. The fold
changes are -0.221, +0.430, +1.111, +0.202 and -0.274 points.

This is useful robustness evidence, but not a second promotion opportunity.
The primary frozen-fold gain remains well below the declared threshold; the
grouped result also breaches the worst-fold limit and reduces the already very
low repair recall by 0.538 points. Funder prevalence may help some unfamiliar
areas, but the evidence is not stable enough to alter the competition recipe.

## Stop rule and next step

The accepted feature policy remains frozen. Do not tune more rare thresholds,
frequency transformations or aliases against these same folds in this
iteration. Revisit organisation identities only with a materially different
validation design or representation established in advance.

The next data-focused feature-family loop should test numeric state plus
magnitude treatments for `amount_tsh`, `population`, `gps_height` and
`num_private`. That directly addresses the audit's sentinel spikes, skewed
positive values and fold-fitted imputation question without returning to
oversampling.

Executable screening lives in
[`scripts/run_high_cardinality_screen.py`](../scripts/run_high_cardinality_screen.py),
with reusable transformations in
[`src/high_cardinality_features.py`](../src/high_cardinality_features.py) and
evaluation in
[`src/high_cardinality_evaluation.py`](../src/high_cardinality_evaluation.py).
Runtime models and CSV summaries remain ignored local artefacts under
`.runtime/high-cardinality-screen/`.

# Cross-fitted high-cardinality target-encoding screen

## Decision

Do not replace the accepted feature policy with high-cardinality target
encoding. All three complete 55% XGBoost / 45% Random Forest candidates trail
the accepted 81.625% ensemble by at least 0.514 percentage points and reduce
repair recall materially.

Organisation-target-encoded XGBoost does provide a small complementary signal.
A fixed 20% contribution to the accepted ensemble reaches 81.673%, a gain of
0.048 points with four fold wins and a worst-fold change of -0.095 points. This
is stable enough to record but below the required +0.10-point promotion gain.
No local-test or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application in this experiment |
| --- | --- |
| 1. Define the goal and scope | Add smoothed class evidence from deferred identity fields without allowing a row's target to encode itself. |
| 2. Gather the data | Reuse the frozen development partition, raw deferred fields and accepted model evidence. |
| 3. Explore the data | Audit normalised level counts, singleton support, competition unseen exposure and leakage-safe univariate OOF accuracy. |
| 4. Clean and preprocess the data | Normalise case and whitespace, preserve missing and literal sentinel states, and disambiguate ward and subvillage with LGA. |
| 5. Select and engineer features | Add five predeclared identities as empirical-Bayes multiclass target encodings while retaining the accepted 29-feature frame. |
| 6. Define the machine-learning task | Three-class supervised classification with each identity represented by three conditional-class estimates. |
| 7. Partition the data | Keep the five frozen outer folds. Inside every model fit, generate training encodings from a separate five-fold stratified cross-fit; transform outer validation from mappings fitted on outer training only. |
| 8. Select and train candidate methods | Evaluate location, organisation and combined policies using the accepted component specifications and fixed vote. |
| 9. Evaluate and interpret the results | Apply the existing promotion gate, inspect component crosses and run one bounded low-weight organisation-XGBoost contribution curve. |
| 10. Deploy and iterate | Retain the accepted recipe. Move from category-wide rates to cross-fitted local spatial outcome features. |

## Field audit and fixed register

The audit used only OOF encodings when labels participated. LGA-waterpoint-name
was excluded before model fitting because it has 36,276 levels, 33,016
singletons and only 55.676% univariate OOF accuracy.

| Candidate identity | Development levels | Singleton levels | Unseen competition rows | Univariate OOF accuracy |
| --- | ---: | ---: | ---: | ---: |
| LGA–ward | 2,182 | 60 | 21 | 65.404% |
| LGA–subvillage | 21,808 | 11,855 | 3,802 | 62.513% |
| Scheme name | 2,383 | 581 | 191 | 61.923% |
| Funder | 1,692 | 850 | 270 | 60.785% |
| Installer | 1,717 | 839 | 252 | 61.473% |

The three policies were fixed before model evaluation:

- **location identity:** LGA–ward, LGA–subvillage and scheme name;
- **organisation identity:** funder and installer;
- **combined identity:** all five fields.

Each field creates three numeric values, one for each class. Smoothing uses the
target encoder's empirical-Bayes estimate; there is no smoothing grid. The
combined first-fold matrix has 316 columns: the accepted 301 transformed
features plus fifteen encoded values.

## Leakage boundary

For outer-training rows, `fit_transform` creates identity values from a fixed
five-fold stratified inner cross-fit. For outer-validation rows, `transform`
uses mappings fitted on all outer-training rows. Unknown identities receive the
outer-training class mean. No outer-validation label participates in a mapping,
and no training row receives an encoding calculated from its own target.

The accelerated-tree evaluator previously had no supervised preprocessing and
therefore called `fit_transform(X)` without `y`. This screen generalised that
handoff to `fit_transform(X, y)` for XGBoost, LightGBM and sklearn tree refits.
Existing unsupervised preprocessors ignore the additional argument and retain
their prior behaviour. A focused regression test records the encoded target
received by full-data model refitting.

## Complete-policy results

| Policy | Mean accuracy | Change | Fold wins | Worst fold | Repair-recall change | XGBoost | Random Forest |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted ensemble | **81.625%** | — | — | — | — | 81.014% | 80.591% |
| Organisation identity | 81.111% | -0.514 pp | 1/5 | -0.894 pp | -3.735 pp | 80.787% | 80.730% |
| Combined identity | 80.817% | -0.808 pp | 0/5 | -1.231 pp | -5.936 pp | 80.644% | 80.463% |
| Location identity | 80.756% | -0.869 pp | 0/5 | -1.210 pp | -5.907 pp | 80.486% | 80.179% |

The target rates are informative alone, but adding them changes both tree
learners' decision surfaces in an unhelpful direction. Location identities are
especially sparse and the complete policies move probability mass away from
the already difficult repair class.

## Component and contribution checks

Six fixed component crosses paired one encoded component with its unchanged
accepted partner. The closest was organisation-target-encoded XGBoost plus the
accepted Random Forest at 81.608%, down 0.017 points with two fold wins. No
component cross passes the promotion gate.

Because that XGBoost cross was close, one bounded contribution curve preserved
the accepted 55:45 ratio in the remaining probability mass:

| Organisation-XGBoost contribution | Accuracy | Change | Fold wins | Worst fold | Repair-recall change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 81.599% | -0.025 pp | 2/5 | -0.084 pp | -0.087 pp |
| 10% | 81.639% | +0.015 pp | 2/5 | -0.126 pp | -0.289 pp |
| 15% | 81.660% | +0.036 pp | 3/5 | -0.063 pp | -0.492 pp |
| **20%** | **81.673%** | **+0.048 pp** | **4/5** | **-0.095 pp** | **-0.897 pp** |
| 25% | 81.633% | +0.008 pp | 3/5 | -0.200 pp | -1.390 pp |
| 30% | 81.648% | +0.023 pp | 4/5 | -0.263 pp | -1.737 pp |

The curve peaks at 20%; the predeclared 25% and 30% bracket stops further
weight exploration. An exploratory cross against the prior source-plus-class
components reached 81.675% at a 15% contribution, +0.051 points against the
accepted baseline with four wins. It confirms complementarity but still does
not meet the minimum mean-gain threshold and is not a promotion candidate.

## Interpretation and next experiment

Target encoding is not the missing percentage point in this form. Category-wide
class rates add a little diversity through organisation-aware XGBoost, but the
complete representation is substantially worse and the small blend gain has
already consumed a bounded weight check. Stop target-encoding field and weight
grids here.

The next higher-value experiment is local spatial outcome structure: for each
outer-training row, create class-rate features from geographically nearby
inner-training pumps; for validation, use neighbours from the complete
outer-training partition. This can distinguish local conditions within a broad
ward or LGA rather than assigning one category-wide rate. It must provide
explicit fallbacks for invalid coordinates and must never include a row as its
own neighbour.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_target_encoding_screen.py
```

Runtime evidence remains ignored under `.runtime/target-encoding-screen/`.

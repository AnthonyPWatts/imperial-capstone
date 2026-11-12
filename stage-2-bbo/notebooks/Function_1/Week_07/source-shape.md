# Function 1 — Week 7 source-shaped comparison

**Agreed Week 7 choice, 5 November 2026: `0.627916-0.628953`.**
The [selection record](../../../submissions/Week_07/README.md) establishes the
decision following this study, replacing the earlier log-magnitude suggestion.
Nothing has been submitted. Seven queries remain, including Week 7. The study was executed on
**5 November 2026**, using **only the 16 official observations through round 6**.

[Executed notebook](official/source_shape_comparison.ipynb) ·
[Complete numerical evidence](official/source-shape/results.json) ·
[Verification](official/source-shape/verification.json) ·
[Earlier three-target study](activity-hypothesis.md) · [Week-level review](report.md)

## What was fitted

The simplest hypothesis is a circular positive Gaussian component:

$$
s(x)=A\exp\left[-\frac{(x_1-c_1)^2+(x_2-c_2)^2}{2\sigma^2}\right].
$$

Its base-10 logarithm is a concave quadratic with four coefficients. Bounded
linear least squares estimates those coefficients, and the centre follows
analytically. There is no starting-location guess or candidate search for this
model's peak. The strongest five positive training readings supply one residual
degree of freedom. The fitted width is **0.01768**, and the peak's conditional
height is **0.7081**. That height is a prediction under this component assumption,
not a calibrated forecast of the actual signed function.

The five readings are `round-06`, `initial-003`, `initial-008`, `initial-002`
and `initial-010`. Their training log10 MAE is **0.0613 decades**, reported
separately from validation. Negative observations remain in the dataset and
every validation score; they are excluded only from positive-component fitting.
The component cannot represent a negative response.

Cohort discussion suggested the model family and subset approach. No peer
location, output, fitted parameter or CSV enters fitting, validation, query
selection or the source maps. The fitted location is inferred from the official
coordinates and responses. Since the strategy was introduced after seeing the
data and discussion, these are retrospective checks, not blind confirmation.

## Comparison with the existing fits

All 16 official targets receive leave-one-out predictions. Each source fold
removes its held-out observation before selecting its positive training subset.
The existing four GP families' **official-only** predictions are read from the
previous study; the saved official-only neural seed-mean forecasts are also
retained. No new peer-assisted result enters this comparison.

The signed-log target is
$h(y)=\operatorname{sign}(y)\log_{10}(1+|y|/10^{-200})$.
Lower error is better; MAE remains the declared selection metric, with RMSE
shown to expose large mistakes.

| Fit, official data only | Signed-log MAE | Signed-log RMSE |
| --- | ---: | ---: |
| Mean baseline | 135.523 | **146.950** |
| Median baseline | 165.078 | 174.736 |
| Best existing signed GP | 135.652 | 147.135 |
| Selected neural architecture, three-seed mean | 193.057 | 234.137 |
| Circular source, strongest five positives | **91.662** | 155.154 |

The source's signed-log MAE is **32.4% lower than the official-only GP's**.
Its RMSE is worse, and it necessarily misses all seven negative signs. This is
evidence for investigating a local component, not grounds for replacing the
whole signed function with a positive Gaussian.

The other targets tell a different part of the story. Log magnitude is
$g(y)=|h(y)|$; raw magnitude is $|y|$. Columns have different units and must
not be compared with each other.

| Fit | Log-magnitude MAE | Raw-magnitude MAE |
| --- | ---: | ---: |
| Mean baseline | 48.423 | 0.00106346 |
| Median baseline | 58.003 | 0.00056971 |
| Best existing official GP for that target | **21.557** | 0.00106494 |
| Circular source, strongest five positives | 33.276 | **0.00036056** |

The log-magnitude GP remains the better global predictor of log magnitude.
The source's raw **signed** MAE is **0.00081132**, worse than the signed median
baseline's **0.00056971**. Its favourable raw-magnitude error must not be
mistaken for equally good signed-score prediction.

## How stable is the source location?

| Full-data source assumption | Proposed input | Conditional peak height |
| --- | --- | ---: |
| Circular, five positives | `0.627916-0.628953` | 0.7081 |
| Circular, six positives | `0.628011-0.629003` | 0.7486 |
| Axis-aligned, five positives | `0.629738-0.627069` | 0.6136 |
| Axis-aligned, six positives | `0.628031-0.628707` | 0.6625 |

With one official observation omitted at a time, the default source's
rounded peak spans **x1 = 0.627916–0.628910** and
**x2 = 0.628253–0.629387**. Conditional peak heights span **0.439–0.877**.
These are deletion-sensitivity ranges, not confidence intervals. The
five-reading axis-aligned fit has five parameters and zero residual degrees
of freedom; its exact interpolation does not validate the extra flexibility.

The full-data five/six agreement has limits. Six-reading leave-one-out fits
can produce extreme extrapolations: circular raw-magnitude MAE reaches
**2.70e18**, despite a similar full-data peak. Keeping seven to nine positive
readings moves or destabilises the peak. The seven-reading axis-aligned fit
hits the curvature floor and proposes a boundary maximum. These results expose
dependence on the component assumption; weaker readings are not automatically
labelled noise.

Fitting every absolute magnitude also fails as a simple global source. The
circular envelope proposes `0.525264-0.531784`, has log-magnitude MAE **36.578**,
and extreme raw extrapolation error. It does not justify treating all activity
as one positive source.

## Check against earlier returned queries

The notebook rebuilds the default source before each of the six historical
returns, using only preceding observations. From the original ten rows it
already proposes `0.628315-0.629387`. For the latest input, it predicts
**1.0350e-14**, compared with the observed **2.1709e-14**, approximately a factor
of 2.1 apart.

This latest match is useful but selective success would mislead. The same
chronological checks miss the four negative returns' signs and several
magnitudes badly; all six are displayed. They are reconstructed temporal
holdouts, not forecasts recorded before those evaluations.

## Why test this query next?

The narrow positive-component hypothesis gives a specific, falsifiable test
and independently fits a consistent peak under the simpler five/six-reading
assumptions. Its strongest-five deletion checks keep that peak nearby. This
supports spending one of seven remaining opportunities on its predicted
maximum, while retaining the GP fits as competing explanations.

For a fair current comparison, the selected official-only GPs are refitted on
the same 16 observations. Their common seeded broad/local pool includes every
source model's proposed maximum. Their weight-1.96 choices are:

| Policy, official data only | Proposed input |
| --- | --- |
| Signed-log GP | `0.545831-0.555178` |
| Log-magnitude GP | `0.584887-0.590241` |
| Raw-magnitude GP | `0.452527-0.439455` |
| Circular positive source | **`0.627916-0.628953`** |

These differ from the earlier peer-assisted proposals because this comparison
uses only official training data. At the source point the signed GP predicts
**h = -17.610**, with latent SD **135.343**: it supplies no confident positive
confirmation. The nearest official observation is the negative `initial-005`
at Euclidean distance **0.05707**, returning **-0.00360606**. A negative or
near-zero return remains possible.

The recommended point satisfies the input bounds and six-decimal portal format,
and duplicates neither an official observation nor any of the six submitted
Function 1 inputs. No evaluation or submission was performed.

## Verification and reproduction

The new notebook completed **nine code cells**, saving four embedded charts
and four PNG files; all four figures were visually inspected. The source,
Function 1 target, prediction-review and cohort-review helper suites passed
**31 tests**. Independent readback checked **192 training-subset selections**,
recomputed all 16 default fold forecasts and 106 target-specific metrics,
and matched the saved official GP/neural predictions exactly.

Hashes confirmed **260 protected files unchanged**, including prior activity
evidence, observations and submission records. The original model notebook's
code and outputs are unchanged; its markdown now links the source study.
There is no new independent forward evaluation. From the Capstone root,
with the local observation snapshot restored, reproduce the source study with:

```powershell
.\run-python.ps1 stage-2-bbo/scripts/run_week_07_reviews.py --functions 1 --kind source
```

See the [reproduction guide](../../../docs/reproducing-the-analyses.md#function-1-official-only-source-comparison)
for the saved activity/model review prerequisites.

## Course lifecycle

1. **Define the goal and scope** — compare a small source hypothesis with
   existing fits and recommend one Function 1 experiment.
2. **Gather the data** — use the 16 official observations through round 6 and
   saved official-only predictions; peer data are not used.
3. **Explore the data** — retain nine positive and seven negative readings,
   their magnitudes, locations and chronological order.
4. **Clean and preprocess the data** — preserve full-precision observations;
   fit in log10 units and retain extreme predictions in the log domain.
5. **Select and engineer features** — use linear coordinates and constrained
   quadratic terms; reselect positive subsets inside every training fold.
6. **Define the machine-learning task** — regress a hypothetical positive
   component while evaluating against the signed and magnitude objectives.
7. **Partition the data** — use identical official leave-one-out targets and
   retrospective temporal holdouts; no independent final test set is available.
8. **Select and train candidate methods** — fit circular and axis-aligned
   sources, subset/envelope controls, and the selected official-only GPs.
9. **Evaluate and interpret the results** — compare target-specific MAE,
   signed-log RMSE, baselines, deletion stability and model failures.
10. **Deploy and iterate** — recommend the bounded source peak for discussion;
    await a future official return before judging the hypothesis.

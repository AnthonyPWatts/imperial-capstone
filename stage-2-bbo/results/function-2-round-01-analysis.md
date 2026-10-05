# Function 2: first returned result

Reviewed on **5 October 2026**, using the first emailed return and eleven
accumulated observations. The email, attachments and proposal were reconciled
during the [round 1 import](../submissions/round-01-results.json); no observations
have been added or changed by this analysis.

## The problem and its objective

The [original Function 2 brief](https://classroom.emeritus.org/courses/18642/modules/items/3237303)
describes two numerical inputs producing a noisy log-likelihood score, with
possible local optima. The stated objective is to maximise the supplied score.
Higher signed values are better; maximising magnitude, negating the score or
minimising it would change the objective. The noise level and true optimum
are not supplied. The raw scores are used throughout, with no extra log transform.

## What changed

| Measure | Result |
| --- | ---: |
| Evaluated input | `(0.774999, 0.949999)` |
| Returned output | `0.2834440433397594` |
| Observations | 10 → 11 |
| Rank of new output | 5th of 11 |
| Best observed output, unchanged | `0.6112052157614438` |
| Best observed location, approximately | `(0.702637, 0.926564)` |
| Difference from the best observed score | `-0.3277611724` |
| Improvement in the best observed score | 0 |

This is a middle-ranking response, not a new extreme. The supplied best remains
the incumbent, but in a noisy problem the largest recorded value need not
identify the location with the highest underlying mean. No location has been
evaluated twice, so repeat-evaluation noise remains unmeasured.

## What the query taught us

The query was chosen by the local two-input model's knowledge gradient: the
expected value of an answer for improving a later decision. It was not selected
as the best geometric coverage point or as a guaranteed improvement.

It lies only **0.076063** input-space units from the best observed location,
which is also its nearest initial neighbour. The changes were approximately
`+0.07236` in x1 and `+0.02343` in x2. A substantially lower output nearby weakens
the expectation of a broadly high response extending north-east from the
incumbent. It could reflect local variation, observation noise, or both.
Because both inputs changed, it does **not** isolate an x2 effect or reject an
x1-only explanation. A narrower promising band around x1 ≈ 0.7 remains plausible.

Average distance to the nearest observation fell from **0.169002303 to
0.167800765**, a **0.71096%** coverage improvement. That modest gain matches the
original estimate and is consistent with querying near an existing observation.
It does not measure information gained or model accuracy.

The original models' forecasts were recomputed using only the initial ten rows:

| Original-data model | Predicted mean at query | Predictive SD for a noisy observation | Actual minus prediction, in assumed SD |
| --- | ---: | ---: | ---: |
| x1-only | 0.383206 | 0.172905 | -0.58 |
| Local 2D, shared length scale | 0.536115 | 0.161246 | -1.57 |
| Flexible 2D, separate length scales | 0.360537 | 0.185524 | -0.42 |

The return is closer to the x1-only and flexible forecasts. It is still within
the displayed ±1.96 predictive-SD ranges of all three models, so this single
answer does not decisively rule out any of them. These are conditional model
ranges, not empirically calibrated coverage guarantees. All use an assumed
observation-noise SD of **0.05 output units**, not a noise estimate from repeats.

## Updated prediction checks

Each leave-one-out prediction refits the model on the other observations,
including output normalisation within that training fold. The original and
updated snapshots use identical code, families and main noise assumptions.

| Method | Initial RMSE | Updated RMSE | Updated MAE |
| --- | ---: | ---: | ---: |
| Training-mean baseline | 0.250405 | 0.236953 | 0.196003 |
| x1-only GP | **0.188355** | **0.167832** | **0.129154** |
| Local 2D GP | 0.265191 | 0.248998 | 0.210944 |
| Flexible 2D GP | 0.202645 | 0.178863 | 0.136745 |

The x1-only model retains the lowest prediction error: **29.17% lower RMSE**
than the updated mean baseline. The flexible model is **24.52% better** than
that baseline; the local model is **5.08% worse**. This is appreciably more
support for predictive structure than Function 1 offered.

The flexible fit again chooses a long x2 length scale at the permitted upper
bound, consistent with weak x2 variation under this fitted model. That boundary
fit and the small sample do not prove x2 is irrelevant. The local model instead
shortens its shared length scale from about **0.134 to 0.0626**, producing tighter
features around observations. This is its response to the evidence, not a
measurement of the true surface.

The initial and updated RMSE columns contain different observation sets. Their
decrease alone is not evidence of improved generalisation. Compare models on
the same rows; the eleven adaptive observations do not provide an independent
test set or statistical proof of a winning model.

## Conditional next-query choices

The next one-query knowledge-gradient calculation uses the existing three
model families and a 41 × 41 grid, supplemented by the coverage optimum,
earlier gap examples and portal-rounded observed locations. The full-precision
observations remain possible final decisions. All proposed query coordinates
obey six-decimal precision and `[0, 1)` bounds.

| Assumed model, noise SD 0.05 | Best tested next input | Conditional KG gain | Geometric coverage gain |
| --- | --- | ---: | ---: |
| x1-only | `(0.949999, 0.300000)` | 0.006754 | 7.88% |
| Local 2D | `(0.674999, 0.924999)` | 0.018537 | 0.32% |
| Flexible 2D | `(0.699999, 0.000000)` | 0.021346 | 0.89% |
| Pure coverage, without output modelling | `(0.162244, 0.811282)` | Model-dependent | 13.91% |

These KG values are expected improvements in a final fitted-mean decision on a
finite pool, **not** predicted improvements over the recorded incumbent.
Each row is conditional on a different model; the largest number is not a
model-independent reason to choose that row. Under the x1-only model, x2 makes
no statistical difference, so geometric coverage breaks the tie between x2 values.

The 25 × 25 grid check retains the same broad regions. Noise sensitivity is
more consequential: at SD 0.15 the x1-only choice moves towards `(0.725, 0.4)`,
and the local model switches from the upper incumbent to the lower high-response
region near `(0.675, 0.125)`. The flexible model stays near x1 = 0.7 at low x2
across SD 0.025, 0.05 and 0.15. Neither the noise assumptions nor these kernels
have been established as correct.

The practical conclusion is to retain **x1-led exploration and refinement** in
the shortlist, while testing whether the apparent promising band extends across
x2. The flexible model's lower-edge point is one concrete candidate; the x1-only
model's far-right point addresses a different uncertainty. The local model's
incumbent-neighbour point has weaker held-out support. No single next point is
settled or submitted by this review, and no updated twelve-round rollout was
run. One of the 13 planned rounds is complete; 12 remain.

## Local figures and reproduction

- [Returned location and original-data predictions](../../.runtime/function-2-round-01/result-and-predictions.png)
- [Local-model means before and after, with shared axes and colour scale](../../.runtime/function-2-round-01/local-model-before-after.png)
- [Numerical results, runtime versions and source/figure hashes](../../.runtime/function-2-round-01/analysis.json)

The surface plots show one fitted model, not the unknown function. Their
background reversion towards the sample mean is a modelling assumption. Figures
and detailed numerical records remain local with the restricted observations.

```powershell
.venv/Scripts/python.exe -X utf8 stage-2-bbo/scripts/analyse_function_2_round_01.py
.venv/Scripts/python.exe -m unittest discover -s stage-2-bbo/tests -p 'test_function_2*.py' -v
```

The analysis completed and **all seven focused Function 2 tests passed**.
Saved RMSE and MAE values were independently recomputed from their prediction
arrays. The new row's leave-one-out forecasts match fits on the initial data;
data and source hashes match the preserved records. Candidate bounds, precision
and non-duplication were checked, both figures were visually inspected, and
`git diff --check` passed. The local
[verification record](../../.runtime/function-2-round-01/verification.json)
records these checks. The updated twelve-round policy, actual noise level and
global optimum remain unverified.

## Course lifecycle checkpoint

| Step | Application in this review |
| --- | --- |
| 1. Define the goal and scope | Review the first Function 2 result and its implications for maximising a noisy score. |
| 2. Gather the data | Use the reconciled round 1 return and ten unchanged initial observations. |
| 3. Explore the data | Compare rank, nearby observations, coverage and forecast residuals. |
| 4. Clean and preprocess the data | Preserve all valid observations; keep GP normalisation within training folds. |
| 5. Select and engineer features | Compare the existing x1-only and two-input hypotheses; no new features. |
| 6. Define the machine-learning task | Sequential maximisation aided by noisy-response regression. |
| 7. Partition the data | Leave-one-out checks; no separate test set with eleven adaptive observations. |
| 8. Select and train candidate methods | Refit the existing three GP families and a training-mean baseline. |
| 9. Evaluate and interpret the results | No incumbent gain; x1-only prediction remains strongest; query choices depend on model and noise assumptions. |
| 10. Deploy and iterate | Record conditional next choices; deployment/submission is outside this review. |

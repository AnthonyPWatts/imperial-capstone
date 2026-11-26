# Week 10 review

The Week 9 results arrived on 25 November 2026 at 12:51 GMT. All eight inputs
match the recorded submission. The official total is **247 observations**, and
the existing best scores still lead on all eight functions. Four queries per
function remain, including Week 10.
[Round 9 results](../submissions/round-09-results.json) · [Current unsubmitted decision](../submissions/Week_10/README.md)

The pooled snapshot adds 243 published peer returns, including 26 newly
available returns. There are 490 rows across the eight functions. Six new
reported rows were excluded because a coordinate was 1.000000.
[Snapshot and provenance notes](../data/README.md#full-cohort-refresh-on-25-november-2026)

## Owner-first cohort advice

The [advisory review](../submissions/Week_10/advisory-review.md) links eight executed notebooks, conditional pairwise surfaces and a separate experimental input list. The owner GP generates the shortlist and supplies 80% of the ranking. Current and archived cohort layers contribute up to 10% each. The [current Week 10 decision](../submissions/Week_10/README.md) also compares the archive challengers outside the shortlist and records the course check.

## Weekly decision notebooks

These are the primary Week 10 reviews. **Expectations** replays the forecast
frozen before round 9. **Results** compares the official return with local
measurements and labelled peer and historical references. **Predictions**
checks the incumbent, compares the four recorded proposals and shows both prepared, unsubmitted routes before the adaptive strategy. Read the [cross-function decision story](../notebooks/capstone_decision_story.ipynb) for the full chronology.

| Function | Peak assessment | Prepared Week 10 plan |
| --- | --- | --- |
| [F1](../notebooks/Function_1/Week_10/weekly_review.ipynb) | A narrow local source is supported. A second source remains possible. | Use the diagonal source refinement where assessed reuse permits. The fallback uses the unconstrained official fit. |
| [F2](../notebooks/Function_2/Week_10/weekly_review.ipynb) | The x1 ≈ 0.70 band is supported. Published repeat evidence favours a competing region. | The revised batch tests the exact archive centre. The fallback stays near the established x1 band. |
| [F3](../notebooks/Function_3/Week_10/weekly_review.ipynb) | Multiple near-zero observations support a region. The zero ceiling is a hypothesis. | Both paths change only x2. The revised path tests 0.555 and the fallback tests 0.650. |
| [F4](../notebooks/Function_4/Week_10/weekly_review.ipynb) | A local hill is supported. The peak location and any disconnected hill remain open. | Use the x3 midpoint between the best two official scores. |
| [F5](../notebooks/Function_5/Week_10/weekly_review.ipynb) | You match the best verified portal-valid corner. Both GPs strongly favour it on the sampled set. | Probe the x4 slope at the checked spacing. Both paths use the same point. |
| [F6](../notebooks/Function_6/Week_10/weekly_review.ipynb) | The x5 = 0 face is a strong local region. Published scores identify a stronger interior region. | The revised path tests the archive interior. The official fallback uses the x5 = 0 face. |
| [F7](../notebooks/Function_7/Week_10/weekly_review.ipynb) | The boundary basin is promising. Published interior results beat it. | The revised path tests the x1/x3 archive basin. The fallback changes x3 on the official face. |
| [F8](../notebooks/Function_8/Week_10/weekly_review.ipynb) | The verified starter-only quadratic peaks at 10. The incumbent is 0.06323 below it. | Evaluate the fitted stationary point at six-decimal precision. |

Each decision fit uses only official observations. Two kernel families
report how often the incumbent is within a declared tolerance of the best
point on a local or broader candidate set. The figures depend on model
fit and search coverage. The notebooks include held-out forecast errors,
compare one-query expected improvement with an adaptive two-query
shortlist calculation, and discuss where these estimates are useful.

[Course collaboration and data-use review](../docs/capstone-data-policy.md)

## Observations and notebooks

| Function | Round 9 return | Official rank | Official exploration | Official predictions | Pooled exploration | Pooled predictions |
| --- | ---: | ---: | --- | --- | --- | --- |
| 1 | 1.99399319 | 3/19 | [Explore](../notebooks/Function_1/Week_10/official/exploration.ipynb) | [Predict](../notebooks/Function_1/Week_10/official/model_predictions.ipynb) | [Explore](../notebooks/Function_1/Week_10/full-cohort/observation_review.ipynb) | [Predict](../notebooks/Function_1/Week_10/full-cohort/model_predictions.ipynb) |
| 2 | 0.702966281 | 2/19 | [Explore](../notebooks/Function_2/Week_10/official/exploration-review.ipynb) | [Predict](../notebooks/Function_2/Week_10/official/model_predictions.ipynb) | [Explore](../notebooks/Function_2/Week_10/full-cohort/observation_review.ipynb) | [Predict](../notebooks/Function_2/Week_10/full-cohort/model_predictions.ipynb) |
| 3 | -0.00810935749 | 2/24 | [Explore](../notebooks/Function_3/Week_10/official/exploration-review.ipynb) | [Predict](../notebooks/Function_3/Week_10/official/model_predictions.ipynb) | [Explore](../notebooks/Function_3/Week_10/full-cohort/observation_review.ipynb) | [Predict](../notebooks/Function_3/Week_10/full-cohort/model_predictions.ipynb) |
| 4 | 0.507907947 | 4/39 | [Explore](../notebooks/Function_4/Week_10/official/observation_review.ipynb) | [Predict](../notebooks/Function_4/Week_10/official/model_predictions.ipynb) | [Explore](../notebooks/Function_4/Week_10/full-cohort/observation_review.ipynb) | [Predict](../notebooks/Function_4/Week_10/full-cohort/model_predictions.ipynb) |
| 5 | 8223.20909 | 6/29 | [Explore](../notebooks/Function_5/Week_10/observation_review.ipynb) | [Predict](../notebooks/Function_5/Week_10/model_predictions.ipynb) | [Explore](../notebooks/Function_5/Week_10/full-cohort/observation_review.ipynb) | [Predict](../notebooks/Function_5/Week_10/full-cohort/model_predictions.ipynb) |
| 6 | -0.241345937 | 3/29 | [Explore](../notebooks/Function_6/Week_10/observation_review.ipynb) | [Predict](../notebooks/Function_6/Week_10/model_predictions.ipynb) | [Explore](../notebooks/Function_6/Week_10/full-cohort/observation_review.ipynb) | [Predict](../notebooks/Function_6/Week_10/full-cohort/model_predictions.ipynb) |
| 7 | 2.54888783 | 5/39 | [Explore](../notebooks/Function_7/Week_10/observation_review.ipynb) | [Predict](../notebooks/Function_7/Week_10/model_predictions.ipynb) | [Explore](../notebooks/Function_7/Week_10/full-cohort/observation_review.ipynb) | [Predict](../notebooks/Function_7/Week_10/full-cohort/model_predictions.ipynb) |
| 8 | 9.87426737 | 3/49 | [Explore](../notebooks/Function_8/Week_10/observation_review.ipynb) | [Predict](../notebooks/Function_8/Week_10/model_predictions.ipynb) | [Explore](../notebooks/Function_8/Week_10/full-cohort/observation_review.ipynb) | [Predict](../notebooks/Function_8/Week_10/full-cohort/model_predictions.ipynb) |

Rank 1 denotes the highest score among your own starting data and confirmed
returns for that function. The exploration notebooks replay the Week 9 fits
at the exact delivered inputs and compare those forecasts with the new scores.

## Archived Week 9 forecasts at the returned inputs

| Function | Units | Actual | Archived GP mean | Archived neural mean |
| --- | --- | ---: | ---: | ---: |
| 1 | Signed-log height | 200.29972 | 198.1633 | 130.82299 |
| 2 | Raw score | 0.70296628 | 0.62178547 | 0.3237486 |
| 3 | Raw score | -0.0081093575 | -0.014328326 | -0.016313374 |
| 4 | Raw score | 0.50790795 | -0.060301592 | 0.42683399 |
| 5 | Raw score | 8223.2091 | 8309.6607 | 8303.713 |
| 6 | Raw score | -0.24134594 | -0.26789612 | -0.31625106 |
| 7 | Raw score | 2.5488878 | 2.6438182 | 2.7203799 |
| 8 | Raw score | 9.8742674 | 9.9966436 | 9.5203052 |

Some Week 9 decisions used local evidence or a different model. The table
reports the saved weekly GP and neural fits on a consistent basis.

## Pooled validation and tabular transformer experiment

Each pooled prediction notebook compares owner-only and pooled GP fits,
screens fixed alternatives, then uses two nested five-fold repeats to
assess model choice and shortlist ranking. Holdouts consist of official
rows grouped at portal precision. Any peer at a held-out portal point is
excluded from the training fold.

The transformer experiment uses one token per numeric coordinate and a
single frozen block with self-attention and a residual feed-forward layer.
A ridge readout is fitted within each training fold. The matched ablation
omits self-attention and keeps the featurisation and readout settings.
Three fixed seeds are averaged before scoring. This experiment uses the
same grouped official targets as the fixed surrogate screen.
[Feature-tokenisation reference](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)

| Function | Error metric | Pooled rows | Owner GP | Pooled GP | Frozen transformer | Attention-off |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | MAE | 52 | 119.57 | 86.4078 | 131.946 | 134.69 |
| 2 | RMSE | 50 | 0.185619 | 0.133634 | 0.179897 | 0.154847 |
| 3 | RMSE | 54 | 0.0682472 | 0.0195131 | 0.0580351 | 0.0660682 |
| 4 | RMSE | 69 | 1.50245 | 1.37765 | 1.28348 | 1.16382 |
| 5 | RMSE | 57 | 512.099 | 338.844 | 343.836 | 489.389 |
| 6 | RMSE | 60 | 0.323009 | 0.186099 | 0.145846 | 0.0944492 |
| 7 | RMSE | 70 | 0.293189 | 0.397654 | 0.346268 | 0.497415 |
| 8 | RMSE | 78 | 0.279568 | 0.278993 | 0.186379 | 0.0893239 |

Function 1 uses signed-log MAE. Functions 2–8 report RMSE in each function's
raw units. Compare errors within each row.

Pooled GPs had lower error than owner-only GPs on **7 of 8**
functions. The frozen transformer beat the pooled GP on **4 of 8**
(4, 6, 7, 8) and its ablation on **4 of 8** (1, 3, 5, 7). The fit is small and
the attention weights remain frozen, so the result applies to this
benchmark. The Week 10 queries use measured local evidence and checked
surrogate proposals.

[Reproducing the notebook runs](../docs/reproducing-the-analyses.md#week-10-reviews)

# Function 1 — Week 7 review

## Agreed Week 7 choice: test the positive-source hypothesis

**Agreed on 5 November 2026: `0.627916-0.628953`**, not submitted.
The [Week 7 selection record](../../../submissions/Week_07/README.md)
records the decision. The [source comparison](source-shape.md) and
[executed notebook](official/source_shape_comparison.ipynb) use only the
16 official observations. A four-parameter Gaussian fitted to the strongest
five positive training readings places its peak there; the six-reading and
axis-aligned controls place it nearby.

The source has lower held-out signed-log MAE than the official-only GP
(91.662 versus 135.652), but worse RMSE (155.154 versus 147.135). The GP
predicts log magnitude better; the source misses negative responses, and
larger subsets can destabilise it. This supports one specific experiment,
not a recovered global function. The model's conditional height of 0.7081
is not a calibrated signed-score forecast. Seven queries remain.

This follow-up supersedes the earlier log-magnitude recommendation below;
its original fitted evidence remains preserved.

## Earlier study: three target rules

**Earlier recommendation: `0.340635-0.362578` for one log-magnitude-led
exploration query**, recommended for discussion without final agreement or
submission. The [full study](activity-hypothesis.md) and
[executed notebook](official/activity_hypothesis.ipynb) compare signed score,
log magnitude and raw magnitude using all 16 official rows and six unchanged
peers. The log fit improves MAE by 65.0% against its own mean baseline and
beats a median benchmark; the raw fit loses to its median benchmark. All
three weight-1.96 proposals survive shared-pool expansion. The signed mean
at the recommendation is negative and highly uncertain, so this is an
activity-seeking experiment with a risk of a negative return.

The original signed/neural review below preserves its fitted evidence and
provisional signed-GP candidate as a comparator. The source follow-up above
supplies the current recommendation. Seven query opportunities remain.

## Original signed-score and neural comparison

Prepared on **5 November 2026**, using **16 official observations through round 6**. The latest result arrived on 2 November at 07:28 GMT. Seven queries remain, including Week 7. The model candidate is **provisional and unsubmitted**.

[Executed observation review](official/exploration.ipynb) · [Executed models](official/model_predictions.ipynb) · [Numerical review](official/model-review.json) · [Offline GP/neural explorer](official/figures/neural-network-explorer.html)

## What the return adds

The central-region hunch returned the highest signed output recorded so far: 2.1709129251221444e-14. It remains practically zero; no background level or source-detection threshold has been established. This is a numerical incumbent, not established source detection. The fixed coverage shortlist is complete. Assess the central region using the new official observation while keeping the signed-score objective separate from magnitude-based activity hypotheses.

The signed return is **2.17091293e-14**, rank **1/16** (1 = highest). The best observed output is **2.17091293e-14**.

## Forecasts recorded before the return

The final Week 6 selection record supplies these forecasts at the actually submitted coordinates. The observation notebook checks the saved record hash and exact input match; it does not refit the historical models. Units below are **signed-log height h**. The actual value is transformed with the same fixed signed-log scale; no inverse-transformed posterior mean is claimed.

| Quantity | Value |
| --- | ---: |
| Actual in prediction units | 186.336642 |
| Saved GP mean | 184.023849 |
| Saved GP latent SD | 18.5210311 |
| Actual minus saved GP mean | 2.31279379 |
| Saved neural seed mean | -219.54375 |
| Actual minus saved neural mean | 405.880392 |

GP latent SD and neural seed variation measure different things. A single forecast error does not establish calibration.

## Refreshed model evidence

The declared criterion is **LOO MAE** in **signed-log height h**. Each error uses the same official held-out targets, with complete fitting pipelines inside each fold. Neural architecture selection averages seed errors for seeds 11, 42 and 73; ensemble error scores their mean predictions.

| Fit | Held-out error |
| --- | ---: |
| Mean baseline, official only | 135.523206 |
| Best declared official-only GP: Gaussian process | 135.651902 |
| Selected GP: Gaussian process, official_plus_retained_peers | 109.584157 |
| Neural seed mean: relu-8-8, official only | 193.057461 |

The signed-log GP mean maximum remains central at `0.538665-0.520452`, close to the tested point. Weight 1.96 instead chooses the upper-region candidate because of its uncertainty bonus. The selected peer-assisted GP beats the mean baseline on official held-out targets, while the best official-only GP remains weaker than that baseline. The raw return still does not establish meaningful source activity. Magnitude is diagnostic and never replaces the signed competition objective.

Selection reuses these validation errors, so they are optimistic estimates of future performance. Retained-peer GPs have additional training rows and are labelled separately. These checks do not compare complete optimisation policies or establish a global maximum. The declared GP acquisition policy is retained; lower neural prediction error does not automatically replace it.

## Provisional candidate

| Quantity | Value |
| --- | --- |
| GP-UCB input | `0.709459-0.792240` |
| Target mean | 112.643497 |
| Latent posterior SD | 122.527757 |
| UCB, mean + 1.96 × SD | 352.797899 |
| Mean minus measured best in target units | -73.6931458 |
| Inside selected GP training hull | True |
| Nearest known input distance | 0.0630430452 |

The neural mean-led alternative is `0.997160-0.276876`, with target mean **199.226941** and seed SD **52.5234018**. Both candidates use the same eligible finite pool. No final Week 7 selection or submission is recorded.

The notebook checks weights 0, 1, 1.96 and 3 on a common seeded broad/local pool. Six-decimal bounds, duplicate exclusions and the existing spacing settings are retained. The coverage comparator is a finite-pool farthest-point proxy. UCB is a selection score, not a forecast or guarantee of improvement.

![Held-out prediction checks](official/figures/model-validation.png)

![Conditional responses through the incumbent](official/figures/model-response-incumbent.png)

## Reproducibility and limits

Both notebooks have saved executions from the Capstone-local Python interpreter. The new independent CSV exactly matches the round 6 accumulated data at float64 precision. Earlier snapshots and submitted inputs are preserved. Functions 1–4 reuse unchanged retained peers with a new audit; official progress excludes them. Fitting warnings remain in the numerical review. The offline explorer exports the new fitted GP and all declared neural architectures and seeds. It performs no training, course evaluations or submissions.

For 1 selected seed fit, the incumbent is near a ReLU kink. The anchor check is flagged and backpropagation is verified at a fixed nearby smooth probe; no validated derivative is claimed at the kink.

## Course lifecycle

1. **Define the goal and scope** — review round 6 and prepare evidence for the next weekly input; success means reconciled observations and reviewable candidate evidence.
2. **Gather the data** — use the retained 2 November email and original cumulative attachments; reuse audited peer rows only where already present.
3. **Explore the data** — inspect signed ranks, incumbent progress, full input profiles and coverage.
4. **Clean and preprocess the data** — validate finite values, dimensions, bounds and exact round counts; retain full precision and fit scalers inside held-out folds.
5. **Select and engineer features** — retain the ordered input coordinates and established feature-set comparisons; no speculative feature expansion.
6. **Define the machine-learning task** — continuous-output regression supports black-box maximisation.
7. **Partition the data** — use leave-one-out comparisons on the small adaptive dataset; no independent final test set is available.
8. **Select and train candidate methods** — retain established baseline/regression/GP families and three neural architectures with three fixed seeds.
9. **Evaluate and interpret the results** — keep historical forward forecasts separate from new retrospective errors; compare identical official targets and candidate/weight sensitivity.
10. **Deploy and iterate** — deployment is not applicable; save local models, explorers and provisional next-query evidence. Seven queries remain including Week 7, and no query is submitted.

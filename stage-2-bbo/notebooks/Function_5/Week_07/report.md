# Function 5 — Week 7 review

Prepared on **5 November 2026**, using **26 official observations through round 6**. The latest result arrived on 2 November at 07:28 GMT. Seven queries remain, including Week 7. The model candidate is **provisional and unsubmitted**.

[Executed observation review](observation_review.ipynb) · [Executed models](model_predictions.ipynb) · [Numerical review](model-review.json) · [Offline GP/neural explorer](figures/neural-network-explorer.html)

## What the return adds

The deliberate exploration query returns 1616.625747348229, sixth of 26 and above every starting observation, while remaining far below the incumbent 8662.405001248297. Both saved model means were negative, so the new region performed much better than forecast. The return adds information but does not establish a new optimum or resolve the unfinished local x3 check. Its comparison with the incumbent changes x1 and x2 together.

The signed return is **1616.62575**, rank **6/26** (1 = highest). The best observed output is **8662.405**.

## Forecasts recorded before the return

The final Week 6 selection record supplies these forecasts at the actually submitted coordinates. The observation notebook checks the saved record hash and exact input match; it does not refit the historical models. Units below are **output y**. Predictions and the actual return use the original signed output units.

| Quantity | Value |
| --- | ---: |
| Actual in prediction units | 1616.62575 |
| Saved GP mean | -685.995383 |
| Saved GP latent SD | 1583.19981 |
| Actual minus saved GP mean | 2302.62113 |
| Saved neural seed mean | -13.5148153 |
| Actual minus saved neural mean | 1630.14056 |

GP latent SD and neural seed variation measure different things. A single forecast error does not establish calibration.

## Refreshed model evidence

The declared criterion is **LOO RMSE** in **output y**. Each error uses the same official held-out targets, with complete fitting pipelines inside each fold. Neural architecture selection averages seed errors for seeds 11, 42 and 73; ensemble error scores their mean predictions.

| Fit | Held-out error |
| --- | ---: |
| Mean baseline, official only | 3146.78331 |
| Best declared official-only GP: Gaussian process | 570.712259 |
| Selected GP: Gaussian process, official | 570.712259 |
| Neural seed mean: tanh-8, official only | 381.042561 |

The new exploratory row raises held-out error for both model families. The neural ensemble retains lower RMSE (381.043 versus 570.712 for the raw-target GP); the log-target GP comparator has RMSE 498.457. The declared GP-UCB policy and 0.02 spacing safeguard again shortlist the isolated x3 reduction at the incumbent corner. All four tested weights choose the same eligible point. Its mean is below the incumbent. This remains a candidate for the unfinished local check rather than an agreed query or proof of an optimum.

Selection reuses these validation errors, so they are optimistic estimates of future performance. Retained-peer GPs have additional training rows and are labelled separately. These checks do not compare complete optimisation policies or establish a global maximum. The declared GP acquisition policy is retained; lower neural prediction error does not automatically replace it.

## Provisional candidate

| Quantity | Value |
| --- | --- |
| GP-UCB input | `0.999999-0.999999-0.979593-0.999999` |
| Target mean | 8368.77903 |
| Latent posterior SD | 159.853272 |
| UCB, mean + 1.96 × SD | 8682.09144 |
| Mean minus measured best in target units | -293.625975 |
| Inside selected GP training hull | False |
| Nearest known input distance | 0.020406 |

The neural mean-led alternative is `0.976212-0.999999-0.999999-0.999999`, with target mean **8361.98732** and seed SD **2.57662546**. Both candidates use the same eligible finite pool. No final Week 7 selection or submission is recorded.

The notebook checks weights 0, 1, 1.96 and 3 on a common seeded broad/local pool. Six-decimal bounds, duplicate exclusions and the existing spacing settings are retained. The coverage comparator is a finite-pool farthest-point proxy. UCB is a selection score, not a forecast or guarantee of improvement.

![Held-out prediction checks](figures/model-validation.png)

![Conditional responses through the incumbent](figures/model-response-incumbent.png)

## Reproducibility and limits

Both notebooks have saved executions from the Capstone-local Python interpreter. The new independent CSV exactly matches the round 6 accumulated data at float64 precision. Earlier snapshots and submitted inputs are preserved. Functions 1–4 reuse unchanged retained peers with a new audit; official progress excludes them. Fitting warnings remain in the numerical review. The offline explorer exports the new fitted GP and all declared neural architectures and seeds. It performs no training, course evaluations or submissions.

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

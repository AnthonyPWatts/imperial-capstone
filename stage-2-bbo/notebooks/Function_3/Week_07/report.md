# Function 3 — Week 7 review

Prepared on **5 November 2026**, using **21 official observations through round 6**. The latest result arrived on 2 November at 07:28 GMT. Seven queries remain, including Week 7. The model candidate is **provisional and unsubmitted**.

[Executed observation review](official/exploration-review.ipynb) · [Executed models](official/model_predictions.ipynb) · [Numerical review](official/model-review.json) · [Offline GP/neural explorer](official/figures/neural-network-explorer.html)

## What the return adds

The return ranks 19th of 21 and is below both saved GP and neural mean predictions. The two recent low-x3 queries have both disappointed; this weakens the case for repeating that particular extrapolation without a fresh review. The queries changed all three inputs, so they do not isolate an x3 effect. Retained-peer performance must still be checked on official targets. The course description does not specify noisy outputs for this function.

The signed return is **-0.121416802**, rank **19/21** (1 = highest). The best observed output is **-0.0348353134**.

## Forecasts recorded before the return

The final Week 6 selection record supplies these forecasts at the actually submitted coordinates. The observation notebook checks the saved record hash and exact input match; it does not refit the historical models. Units below are **output y**. Predictions and the actual return use the original signed output units.

| Quantity | Value |
| --- | ---: |
| Actual in prediction units | -0.121416802 |
| Saved GP mean | -0.0517369697 |
| Saved GP latent SD | 0.0649374432 |
| Actual minus saved GP mean | -0.0696798318 |
| Saved neural seed mean | -0.0137746368 |
| Actual minus saved neural mean | -0.107642165 |

GP latent SD and neural seed variation measure different things. A single forecast error does not establish calibration.

## Refreshed model evidence

The declared criterion is **LOO RMSE** in **output y**. Each error uses the same official held-out targets, with complete fitting pipelines inside each fold. Neural architecture selection averages seed errors for seeds 11, 42 and 73; ensemble error scores their mean predictions.

| Fit | Held-out error |
| --- | ---: |
| Mean baseline, official only | 0.0779444049 |
| Best declared official-only GP: GP: separate length scales | 0.0770914319 |
| Selected GP: Gaussian process, official_plus_retained_peers | 0.0512737995 |
| Neural seed mean: tanh-8, official only | 0.120022712 |

The refreshed peer-assisted GP still has lower declared error than the official-only alternatives, but the latest forward miss remains explicit. The weight-1.96 candidate moves away from the very-low-x3 boundary to x3=0.478802 and lies inside the selected fit's training hull. Its mean is below the incumbent; uncertainty drives the selection. Retrospective validation does not erase the two disappointing recent returns.

Selection reuses these validation errors, so they are optimistic estimates of future performance. Retained-peer GPs have additional training rows and are labelled separately. These checks do not compare complete optimisation policies or establish a global maximum. The declared GP acquisition policy is retained; lower neural prediction error does not automatically replace it.

## Provisional candidate

| Quantity | Value |
| --- | --- |
| GP-UCB input | `0.434867-0.709252-0.478802` |
| Target mean | -0.0633180605 |
| Latent posterior SD | 0.0663518657 |
| UCB, mean + 1.96 × SD | 0.0667315963 |
| Mean minus measured best in target units | -0.0284827471 |
| Inside selected GP training hull | True |
| Nearest known input distance | 0.179123583 |

The neural mean-led alternative is `0.533798-0.992595-0.724669`, with target mean **0.000302658944** and seed SD **0.00782553897**. Both candidates use the same eligible finite pool. No final Week 7 selection or submission is recorded.

The notebook checks weights 0, 1, 1.96 and 3 on a common seeded broad/local pool. Six-decimal bounds, duplicate exclusions and the existing spacing settings are retained. The coverage comparator is a finite-pool farthest-point proxy. UCB is a selection score, not a forecast or guarantee of improvement.

![Held-out prediction checks](official/figures/model-validation.png)

![Conditional responses through the incumbent](official/figures/model-response-incumbent.png)

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

# Function 4 — Week 7 review

Prepared on **5 November 2026**, using **36 official observations through round 6**. The latest result arrived on 2 November at 07:28 GMT. Seven queries remain, including Week 7. The model candidate is **provisional and unsubmitted**.

[Executed observation review](official/observation_review.ipynb) · [Executed models](official/model_predictions.ipynb) · [Numerical review](official/model-review.json) · [Offline GP/neural explorer](official/figures/neural-network-explorer.html)

## What the return adds

The local test returns the second-highest observed output, 0.48990613858950427, below the round 5 incumbent of 0.6501304054949952. Two nearby positive returns support further examination of this region. The saved GP mean was negative (-0.4087963505982497), so the result also exposes substantial local underprediction; a good relative candidate rank does not establish an accurate or calibrated local forecast.

The signed return is **0.489906139**, rank **2/36** (1 = highest). The best observed output is **0.650130405**.

## Forecasts recorded before the return

The final Week 6 selection record supplies these forecasts at the actually submitted coordinates. The observation notebook checks the saved record hash and exact input match; it does not refit the historical models. Units below are **output y**. Predictions and the actual return use the original signed output units.

| Quantity | Value |
| --- | ---: |
| Actual in prediction units | 0.489906139 |
| Saved GP mean | -0.408796351 |
| Saved GP latent SD | 0.421991855 |
| Actual minus saved GP mean | 0.898702489 |
| Saved neural seed mean | 0.229630537 |
| Actual minus saved neural mean | 0.260275602 |

GP latent SD and neural seed variation measure different things. A single forecast error does not establish calibration.

## Refreshed model evidence

The declared criterion is **LOO RMSE** in **output y**. Each error uses the same official held-out targets, with complete fitting pipelines inside each fold. Neural architecture selection averages seed errors for seeds 11, 42 and 73; ensemble error scores their mean predictions.

| Fit | Held-out error |
| --- | ---: |
| Mean baseline, official only | 9.1952855 |
| Best declared official-only GP: Gaussian process | 1.52441017 |
| Selected GP: Gaussian process, official | 1.52441017 |
| Neural seed mean: relu-8-8, official only | 2.30351136 |

The official GP remains stronger than the neural comparator on held-out error. It still predicts negative means in the promising positive region. Even the provisional point's UCB is below the measured incumbent, so its selection cannot be presented as demonstrated predicted improvement. Review this local forecast mismatch before agreeing the next query.

Selection reuses these validation errors, so they are optimistic estimates of future performance. Retained-peer GPs have additional training rows and are labelled separately. These checks do not compare complete optimisation policies or establish a global maximum. The declared GP acquisition policy is retained; lower neural prediction error does not automatically replace it.

## Provisional candidate

| Quantity | Value |
| --- | --- |
| GP-UCB input | `0.372732-0.416315-0.443160-0.432391` |
| Target mean | -0.390906225 |
| Latent posterior SD | 0.487548024 |
| UCB, mean + 1.96 × SD | 0.564687902 |
| Mean minus measured best in target units | -1.04103663 |
| Inside selected GP training hull | True |
| Nearest known input distance | 0.0455586561 |

The neural mean-led alternative is `0.461139-0.413208-0.377270-0.390194`, with target mean **0.346304697** and seed SD **0.18160062**. Both candidates use the same eligible finite pool. No final Week 7 selection or submission is recorded.

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

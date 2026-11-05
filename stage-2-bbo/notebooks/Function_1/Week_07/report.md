# Function 1 — Week 7 review

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

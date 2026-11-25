# Function 4 — Week 9 review

The 23 November round 8 return was **0.638271373**, rank **3/38**. The best observed score is **0.666089599**.

The single-coordinate probe relative to the high peer setting returned the third-best official score. Its observed 0.638271 was well above the archived weekly GP mean of −0.111061 at the same point.

[Observation review](official/observation_review.ipynb) · [Model predictions](official/model_predictions.ipynb)

## Archived fit comparison

At `0.407459-0.418058-0.377000-0.420348`, the frozen Week 8 weekly GP mean was **-0.11106153** and the neural seed mean was **0.106936834**. The observed score in these same raw output y units was **0.638271373**.

## Refreshed models

The selected GP (Gaussian process, official) had **1.5269439** LOO RMSE. The relu-8-8 ensemble had **2.42572311**, and the mean baseline had **9.58493044** on the same official targets.

Adding the retained peers was tested on the same official targets. The official-only GP had the lower error for this week.

### Provisional inputs

| Candidate | Input | GP mean | GP latent SD | Neural mean |
| --- | --- | ---: | ---: | ---: |
| GP UCB | `0.391921-0.432079-0.420229-0.427165` | -0.060301592 | 0.400089209 | 0.426833994 |
| Neural mean | `0.434136-0.419322-0.362094-0.405590` | -0.116297406 | 0.337217276 | 0.436365605 |
| Saved Week 8 follow-up | `0.407459-0.406060-0.384000-0.420348` | 0.00201605491 | 0.327823122 | 0.451780217 |

Five queries remain per function. Review the candidates alongside the measured results and uncertainty before choosing the next input.

[Numerical review](official/model-review.json) · [Offline explorer](official/figures/neural-network-explorer.html)

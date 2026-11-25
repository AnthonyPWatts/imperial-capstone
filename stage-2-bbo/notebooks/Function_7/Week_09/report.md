# Function 7 — Week 9 review

The 23 November round 8 return was **2.41389007**, rank **6/38**. The best observed score is **2.73937512**.

The return on the x1 = 0 face ranked sixth. Its output was 0.32549 below the incumbent, with differences across several of the remaining coordinates.

[Observation review](observation_review.ipynb) · [Model predictions](model_predictions.ipynb)

## Archived fit comparison

At `0.000000-0.192025-0.266625-0.295966-0.243731-0.645462`, the frozen Week 8 weekly GP mean was **2.55172174** and the neural seed mean was **2.60734456**. The observed score in these same raw output y units was **2.41389007**.

## Refreshed models

The selected GP (Gaussian process, official) had **0.285916953** LOO RMSE. The tanh-8 ensemble had **0.365961407**, and the mean baseline had **0.953776769** on the same official targets.

The neural ensemble predicts a higher score for its own candidate. The selected GP has the lower held-out error overall. The held-out and conditional-response plots show the different comparisons.

### Provisional inputs

| Candidate | Input | GP mean | GP latent SD | Neural mean |
| --- | --- | ---: | ---: | ---: |
| GP UCB | `0.000000-0.062967-0.325730-0.392674-0.349371-0.775387` | 2.50214338 | 0.22620949 | 2.69388586 |
| Neural mean | `0.020514-0.094706-0.276932-0.221175-0.288729-0.949122` | 2.13479766 | 0.32357388 | 2.94351943 |
| Saved Week 8 follow-up | `0.000000-0.192554-0.245000-0.306265-0.311144-0.649736` | 2.45961397 | 0.094612595 | 2.53204058 |

Five queries remain per function. Review the candidates alongside the measured results and uncertainty before choosing the next input.

[Numerical review](model-review.json) · [Offline explorer](figures/neural-network-explorer.html)

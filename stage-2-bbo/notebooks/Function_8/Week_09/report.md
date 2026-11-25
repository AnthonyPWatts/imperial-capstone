# Function 8 — Week 9 review

The 23 November round 8 return was **9.93677342**, rank **1/48**. The best observed score is **9.93677342**.

The round 8 query increased the measured best by 0.04350. The archived weekly GP at the submitted input predicted 9.934895, within 0.00188 of the observed score.

[Observation review](observation_review.ipynb) · [Model predictions](model_predictions.ipynb)

## Archived fit comparison

At `0.042902-0.215002-0.170186-0.000000-0.946758-0.586941-0.251883-0.480634`, the frozen Week 8 weekly GP mean was **9.93489481** and the neural seed mean was **9.7363731**. The observed score in these same raw output y units was **9.93677342**.

## Refreshed models

The selected GP (Gaussian process, official) had **0.296346971** LOO RMSE. The tanh-16 ensemble had **0.37871984**, and the mean baseline had **1.09062147** on the same official targets.

The saved follow-up has a GP mean above the measured best and a neural mean below it. The broad UCB pick gains its score chiefly from higher latent uncertainty. Both comparisons are visible in the model notebook.

### Provisional inputs

| Candidate | Input | GP mean | GP latent SD | Neural mean |
| --- | --- | ---: | ---: | ---: |
| GP UCB | `0.000000-0.999999-0.000000-0.999999-0.999999-0.000000-0.000000-0.000000` | 9.00451863 | 0.735872884 | 8.57652425 |
| Neural mean | `0.000000-0.065002-0.320186-0.000000-0.999999-0.736941-0.101883-0.630634` | 9.80283286 | 0.1598086 | 9.93922623 |
| Saved Week 8 follow-up | `0.132099-0.186226-0.025472-0.110284-0.888628-0.372450-0.217039-0.569179` | 10.0384712 | 0.13907072 | 9.60732696 |

Five queries remain per function. Review the candidates alongside the measured results and uncertainty before choosing the next input.

[Numerical review](model-review.json) · [Offline explorer](figures/neural-network-explorer.html)

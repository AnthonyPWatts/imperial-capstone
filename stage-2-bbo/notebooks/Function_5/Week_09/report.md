# Function 5 — Week 9 review

The 23 November round 8 return was **8099.17996**, rank **6/28**. The best observed score is **8662.405**.

Lowering x2 alone from the upper-corner incumbent reduced the score by 563.225. The upper corner remains the best observed input. The changes in x2, x3 and x4 now each have a measured comparison with the incumbent.

[Observation review](observation_review.ipynb) · [Model predictions](model_predictions.ipynb)

## Archived fit comparison

At `0.999999-0.969053-0.999999-0.999999`, the frozen Week 8 weekly GP mean was **8284.83603** and the neural seed mean was **8330.07253**. The observed score in these same raw output y units was **8099.17996**.

## Refreshed models

The selected GP (Gaussian process, official) had **529.24327** LOO RMSE. The tanh-8 ensemble had **380.86264**, and the mean baseline had **3494.24057** on the same official targets.

The neural ensemble has the lower held-out error. Both model means at their provisional picks remain below the measured corner score, and the candidate search preserves the 0.02 spacing guard.

### Provisional inputs

| Candidate | Input | GP mean | GP latent SD | Neural mean |
| --- | --- | ---: | ---: | ---: |
| GP UCB | `0.976212-0.999999-0.999999-0.999999` | 8309.66074 | 140.191201 | 8303.71298 |
| Neural mean | `0.976212-0.999999-0.999999-0.999999` | 8309.66074 | 140.191201 | 8303.71298 |

Five queries remain per function. Review the candidates alongside the measured results and uncertainty before choosing the next input.

[Numerical review](model-review.json) · [Offline explorer](figures/neural-network-explorer.html)

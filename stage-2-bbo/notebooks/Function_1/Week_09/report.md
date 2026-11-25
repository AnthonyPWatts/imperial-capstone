# Function 1 — Week 9 review

The 23 November round 8 return was **1.9973557**, rank **1/18**. The best observed score is **1.9973557**.

The new high is only 0.00004134 above the round 7 high. Both readings came from inputs about 0.000385 apart, giving two measured points for examining the local peak.

[Observation review](official/exploration.ipynb) · [Model predictions](official/model_predictions.ipynb)

## Archived fit comparison

At `0.628208-0.629204`, the frozen Week 8 weekly GP mean was **197.213528** and the neural seed mean was **86.9053801**. The observed score in these same signed-log height h units was **200.300455**.

## Refreshed models

The selected GP (Gaussian process, official plus retained peers) had **98.3953555** LOO MAE. The relu-8-8 ensemble had **165.799434**, and the mean baseline had **138.70595** on the same official targets.

Keep the signed-log GP comparison in its fitted units. The six positive-source shape fits separately predict peak heights from 1.461 to 2.092 in raw score units, despite using both recent high positives. This sensitivity limits what the fitted peak locations establish.

### Provisional inputs

| Candidate | Input | GP mean | GP latent SD | Neural mean |
| --- | --- | ---: | ---: | ---: |
| GP UCB | `0.650268-0.605937` | 127.26493 | 122.333169 | 118.328378 |
| Neural mean | `0.997160-0.276876` | 1.8865086 | 148.842388 | 213.058801 |
| Saved Week 8 follow-up | `0.627720-0.629627` | 198.014987 | 11.2993104 | 130.64167 |

Five queries remain per function. Review the candidates alongside the measured results and uncertainty before choosing the next input.

[Numerical review](official/model-review.json) · [Offline explorer](official/figures/neural-network-explorer.html)

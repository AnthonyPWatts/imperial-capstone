# Function 2 — Week 9 review

The 23 November round 8 return was **0.531874267**, rank **4/18**. The best observed score is **0.723057546**.

The lower-x1 edge probe placed fourth. The separate-length-scale official GP used in the final decision recorded a nominal mean of 0.737960 and predictive observation SD of 0.03825. The return was 0.2061 below that mean, about 5.39 of those modelled SDs. [Week 8 decision](../Week_08/official/model_predictions.ipynb)

[Observation review](official/exploration-review.ipynb) · [Model predictions](official/model_predictions.ipynb)

## Archived fit comparison

At `0.689813-0.000000`, the frozen Week 8 weekly GP mean was **0.732454447** and the neural seed mean was **0.615044821**. The observed score in these same raw output y units was **0.531874267**.

## Refreshed models

The selected GP (GP: x1 only, official plus retained peers) had **0.134494275** LOO RMSE. The relu-8-8 ensemble had **0.202727433**, and the mean baseline had **0.242844761** on the same official targets.

The selected x1-only GP leaves x2 unresolved. Geometric coverage breaks that tie for the provisional UCB pick. The nominal Week 8 forecast above came from the separate-length-scale decision fit.

### Provisional inputs

| Candidate | Input | GP mean | GP latent SD | Neural mean |
| --- | --- | ---: | ---: | ---: |
| GP UCB | `0.071663-0.585937` | 0.270057252 | 0.23208761 | -0.0667295509 |
| Neural mean | `0.668181-0.000000` | 0.525016458 | 0.0222473515 | 0.599500956 |
| Saved Week 8 follow-up | `0.724000-0.494000` | 0.499530404 | 0.0195101855 | 0.305700414 |

Five queries remain per function. Review the candidates alongside the measured results and uncertainty before choosing the next input.

[Numerical review](official/model-review.json) · [Offline explorer](official/figures/neural-network-explorer.html)

# Function 3 — Week 9 review

The 23 November round 8 return was **-0.00495200082**, rank **1/23**. The best observed score is **-0.00495200082**.

The x2 probe gave the first improvement over the starting best, from −0.03483531 to −0.00495200. It varied x2 relative to the high peer setting in the Week 8 decision.

[Observation review](official/exploration-review.ipynb) · [Model predictions](official/model_predictions.ipynb)

## Archived fit comparison

At `0.520000-0.610000-0.450000`, the frozen Week 8 weekly GP mean was **-0.0386144742** and the neural seed mean was **-0.0615397043**. The observed score in these same raw output y units was **-0.00495200082**.

## Refreshed models

The selected GP (Gaussian process, official plus retained peers) had **0.0460325126** LOO RMSE. The tanh-16 ensemble had **0.104684589**, and the mean baseline had **0.0769478277** on the same official targets.

The latest result is better than both archived weekly means at the submitted input. The conditional surfaces and the saved follow-up show how the new fits score nearby alternatives.

### Provisional inputs

| Candidate | Input | GP mean | GP latent SD | Neural mean |
| --- | --- | ---: | ---: | ---: |
| GP UCB | `0.405878-0.697808-0.476376` | -0.0284031095 | 0.0603787984 | -0.0352728214 |
| Neural mean | `0.413275-0.989623-0.717766` | -0.0778006709 | 0.0666216974 | 0.00785689466 |
| Saved Week 8 follow-up | `0.495000-0.550000-0.450000` | -0.0251886803 | 0.0341246368 | -0.0173404998 |

Five queries remain per function. Review the candidates alongside the measured results and uncertainty before choosing the next input.

[Numerical review](official/model-review.json) · [Offline explorer](official/figures/neural-network-explorer.html)

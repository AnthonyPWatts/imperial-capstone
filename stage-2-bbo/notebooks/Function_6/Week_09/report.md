# Function 6 — Week 9 review

The 23 November round 8 return was **-0.228584024**, rank **2/28**. The best observed score is **-0.21395954**.

The return on the x5 = 0 face was 0.01462 below the round 7 incumbent. Both of the two strongest official observations lie on that face.

[Observation review](observation_review.ipynb) · [Model predictions](model_predictions.ipynb)

## Archived fit comparison

At `0.458028-0.356761-0.615581-0.686191-0.000000`, the frozen Week 8 weekly GP mean was **-0.293258731** and the neural seed mean was **-0.342923121**. The observed score in these same raw output y units was **-0.228584024**.

## Refreshed models

The selected GP (Gaussian process, official) had **0.329611475** LOO RMSE. The relu-8-8 ensemble had **0.289304971**, and the mean baseline had **0.617181008** on the same official targets.

The neural ensemble has the lower held-out error. The model notebook shows the exact held-out targets and the current scores for the saved x5 = 0 follow-up.

### Provisional inputs

| Candidate | Input | GP mean | GP latent SD | Neural mean |
| --- | --- | ---: | ---: | ---: |
| GP UCB | `0.424535-0.332747-0.976123-0.806605-0.002468` | -0.571698241 | 0.271615286 | -0.298414751 |
| Neural mean | `0.444302-0.306456-0.662828-0.716525-0.000000` | -0.269954023 | 0.0486384176 | -0.227534339 |
| Saved Week 8 follow-up | `0.425361-0.365167-0.650014-0.709746-0.000000` | -0.259793441 | 0.0579102838 | -0.293145127 |

Five queries remain per function. Review the candidates alongside the measured results and uncertainty before choosing the next input.

[Numerical review](model-review.json) · [Offline explorer](figures/neural-network-explorer.html)

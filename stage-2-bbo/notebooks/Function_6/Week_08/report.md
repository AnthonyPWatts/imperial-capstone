# Function 6 — Week 8 review

The round 7 return received on **12 November 2026 at 03:39 GMT** was **-0.213959540335**, rank **1/27**. The returned score improved the previous best of -0.328192387858.

[Observation review](observation_review.ipynb) · [Model predictions](model_predictions.ipynb)

The observation notebook checks the pre-return forecast against the exact returned input. Six queries remain per function, including Week 8.

## Saved forecast

The saved GP and neural means were **-0.36390761** and **-0.34129561** in output y. The actual return in those units was **-0.21395954**.

## Model comparison

The selected GP scored **0.337966** on LOO RMSE (official), and the neural seed mean scored **0.287374**. Both scores use the same official held-out targets. The GP and neural candidate scores are provisional.

[Numerical model review](model-review.json) · [Offline explorer](figures/neural-network-explorer.html)

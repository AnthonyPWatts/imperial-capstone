# Function 8 — Week 8 review

The round 7 return received on **12 November 2026 at 03:39 GMT** was **9.89327362302**, rank **1/47**. The returned score improved the previous best of 9.78294275768.

[Observation review](observation_review.ipynb) · [Model predictions](model_predictions.ipynb)

The observation notebook checks the pre-return forecast against the exact returned input. Six queries remain per function, including Week 8.

## Saved forecast

The saved GP and neural means were **9.87885584** and **9.79205801** in output y. The actual return in those units was **9.89327362**.

## Model comparison

The selected GP scored **0.303117** on LOO RMSE (official), and the neural seed mean scored **0.340707**. Both scores use the same official held-out targets. The GP and neural candidate scores are provisional.

[Numerical model review](model-review.json) · [Offline explorer](figures/neural-network-explorer.html)

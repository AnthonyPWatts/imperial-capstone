# Function 7 — Week 8 review

The round 7 return received on **12 November 2026 at 03:39 GMT** was **2.55901786001**, rank **3/37**. The returned score fell short of the previous best of 2.739375118.

[Observation review](observation_review.ipynb) · [Model predictions](model_predictions.ipynb)

The observation notebook checks the pre-return forecast against the exact returned input. Six queries remain per function, including Week 8.

## Saved forecast

The saved GP and neural means were **2.6426902** and **2.7765171** in output y. The actual return in those units was **2.55901786**.

## Model comparison

The selected GP scored **0.283123** on LOO RMSE (official), and the neural seed mean scored **0.380992**. Both scores use the same official held-out targets. The GP and neural candidate scores are provisional.

[Numerical model review](model-review.json) · [Offline explorer](figures/neural-network-explorer.html)

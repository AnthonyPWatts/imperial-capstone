# Function 5 — Week 8 review

The round 7 return received on **12 November 2026 at 03:39 GMT** was **8283.06154433**, rank **5/27**. The returned score fell short of the previous best of 8662.40500125.

[Observation review](observation_review.ipynb) · [Model predictions](model_predictions.ipynb)

The observation notebook checks the pre-return forecast against the exact returned input. Six queries remain per function, including Week 8.

## Saved forecast

The saved GP and neural means were **8368.77903** and **8351.00501** in output y. The actual return in those units was **8283.06154**.

## Model comparison

The selected GP scored **546.525** on LOO RMSE (official), and the neural seed mean scored **383.843**. Both scores use the same official held-out targets. The GP and neural candidate scores are provisional.

[Numerical model review](model-review.json) · [Offline explorer](figures/neural-network-explorer.html)

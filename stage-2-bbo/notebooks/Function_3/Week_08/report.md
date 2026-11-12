# Function 3 — Week 8 review

The round 7 return received on **12 November 2026 at 03:39 GMT** was **-0.0658738372473**, rank **10/22**. The returned score fell short of the previous best of -0.0348353133501.

[Observation review](official/exploration-review.ipynb) · [Model predictions](official/model_predictions.ipynb)

The observation notebook checks the pre-return forecast against the exact returned input. Six queries remain per function, including Week 8.

## Saved forecast

The saved GP and neural means were **-0.0328870248** and **-0.0549041778** in output y. The actual return in those units was **-0.0658738372**.

## Model comparison

The selected GP scored **0.0488603** on LOO RMSE (official plus retained peers), and the neural seed mean scored **0.105518**. Both scores use the same official held-out targets. The GP and neural candidate scores are provisional.

[Numerical model review](official/model-review.json) · [Offline explorer](official/figures/neural-network-explorer.html)

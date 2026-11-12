# Function 4 — Week 8 review

The round 7 return received on **12 November 2026 at 03:39 GMT** was **0.666089598934**, rank **1/37**. The returned score improved the previous best of 0.650130405495.

[Observation review](official/observation_review.ipynb) · [Model predictions](official/model_predictions.ipynb)

The observation notebook checks the pre-return forecast against the exact returned input. Six queries remain per function, including Week 8.

## Saved forecast

The saved GP and neural means were **-0.28422225** and **0.270912112** in output y. The actual return in those units was **0.666089599**.

## Model comparison

The selected GP scored **1.53822** on LOO RMSE (official), and the neural seed mean scored **2.70712**. Both scores use the same official held-out targets. The GP and neural candidate scores are provisional.

[Numerical model review](official/model-review.json) · [Offline explorer](official/figures/neural-network-explorer.html)

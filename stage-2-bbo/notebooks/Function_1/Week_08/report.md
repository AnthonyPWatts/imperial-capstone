# Function 1 — Week 8 review

The round 7 return received on **12 November 2026 at 03:39 GMT** was **1.99731436342**, rank **1/17**. The returned score improved the previous best of 2.17091292512e-14. The test at the previously fitted positive-source peak returned 1.99731436. The fitted component had a conditional peak height of about 0.7081. The observation establishes a high positive score at this input.

[Observation review](official/exploration.ipynb) · [Model predictions](official/model_predictions.ipynb)

The observation notebook checks the pre-return forecast against the exact returned input. Six queries remain per function, including Week 8.

## Saved forecast

The saved GP and neural means were **-218.183691** and **21.3957944** in signed-log height h. The actual return in those units was **200.300446**.

## Model comparison

The selected GP scored **126.51** on LOO MAE (official plus retained peers), and the neural seed mean scored **195.976**. Both scores use the same official held-out targets. The GP and neural candidate scores are provisional.

[Numerical model review](official/model-review.json) · [Offline explorer](official/figures/neural-network-explorer.html)

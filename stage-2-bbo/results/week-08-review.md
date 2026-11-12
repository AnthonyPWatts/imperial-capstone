# Week 8 review — round 7 returns and refreshed models

The results email arrived on **12 November 2026 at 03:39 GMT**. All eight inputs match the prepared Week 7 batch. The accumulated data contain **231 official observations**. Six query opportunities remain per function, including Week 8.

Functions **1, 4, 6 and 8** set new bests. Function 1 returned **1.997314363**, rank 1 of 17, at the query selected from the positive Gaussian source hypothesis. That hypothesis gave a conditional peak height of about **0.7081** at the chosen input.

## Measured results and notebooks

| Function | Return | Rank | Best observed | Observation review | Model predictions |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 1.997314363 | 1/17 | 1.997314363 | [Review](../notebooks/Function_1/Week_08/official/exploration.ipynb) | [Models](../notebooks/Function_1/Week_08/official/model_predictions.ipynb) |
| 2 | 0.5117797787 | 4/17 | 0.7230575456 | [Review](../notebooks/Function_2/Week_08/official/exploration-review.ipynb) | [Models](../notebooks/Function_2/Week_08/official/model_predictions.ipynb) |
| 3 | -0.06587383725 | 10/22 | -0.03483531335 | [Review](../notebooks/Function_3/Week_08/official/exploration-review.ipynb) | [Models](../notebooks/Function_3/Week_08/official/model_predictions.ipynb) |
| 4 | 0.6660895989 | 1/37 | 0.6660895989 | [Review](../notebooks/Function_4/Week_08/official/observation_review.ipynb) | [Models](../notebooks/Function_4/Week_08/official/model_predictions.ipynb) |
| 5 | 8283.061544 | 5/27 | 8662.405001 | [Review](../notebooks/Function_5/Week_08/observation_review.ipynb) | [Models](../notebooks/Function_5/Week_08/model_predictions.ipynb) |
| 6 | -0.2139595403 | 1/27 | -0.2139595403 | [Review](../notebooks/Function_6/Week_08/observation_review.ipynb) | [Models](../notebooks/Function_6/Week_08/model_predictions.ipynb) |
| 7 | 2.55901786 | 3/37 | 2.739375118 | [Review](../notebooks/Function_7/Week_08/observation_review.ipynb) | [Models](../notebooks/Function_7/Week_08/model_predictions.ipynb) |
| 8 | 9.893273623 | 1/47 | 9.893273623 | [Review](../notebooks/Function_8/Week_08/observation_review.ipynb) | [Models](../notebooks/Function_8/Week_08/model_predictions.ipynb) |

## Forecasts recorded before the return

The observation notebooks check the saved Week 7 selection hash and the exact returned input. Function 1’s GP and neural values use signed-log units. The conditional source height above uses raw units.

| Function | Units | Actual | Saved GP mean | Saved neural mean |
| --- | --- | ---: | ---: | ---: |
| 1 | signed-log height h | 200.300446 | -218.183691 | 21.3957944 |
| 2 | output y | 0.511779779 | 0.703454647 | 0.667512206 |
| 3 | output y | -0.0658738372 | -0.0328870248 | -0.0549041778 |
| 4 | output y | 0.666089599 | -0.28422225 | 0.270912112 |
| 5 | output y | 8283.06154 | 8368.77903 | 8351.00501 |
| 6 | output y | -0.21395954 | -0.36390761 | -0.34129561 |
| 7 | output y | 2.55901786 | 2.6426902 | 2.7765171 |
| 8 | output y | 9.89327362 | 9.87885584 | 9.79205801 |

## Refreshed model evidence

Each pair scores the same official held-out targets. Peer-assisted GP fits use additional retained peer rows. Lower held-out error describes these retrospective fits.

| Function | Criterion | GP training | GP error | Neural seed-mean error |
| --- | --- | --- | ---: | ---: |
| 1 | LOO MAE | official plus retained peers | 126.51 | 195.976 |
| 2 | LOO RMSE | official plus retained peers | 0.115234 | 0.212174 |
| 3 | LOO RMSE | official plus retained peers | 0.0488603 | 0.105518 |
| 4 | LOO RMSE | official | 1.53822 | 2.70712 |
| 5 | LOO RMSE | official | 546.525 | 383.843 |
| 6 | LOO RMSE | official | 0.337966 | 0.287374 |
| 7 | LOO RMSE | official | 0.283123 | 0.380992 |
| 8 | LOO RMSE | official | 0.303117 | 0.340707 |

The GP had lower declared held-out error for six functions. Neural ensembles led on Functions 5 and 6. These estimates reuse validation for model selection on small adaptive datasets.

## Verification

All **16 notebooks** and **184 code cells** ran with the Capstone-local kernel, saving 52 embedded PNGs and 52 figure files. Readback matched the official CSVs to the accumulated NumPy arrays, retained the existing peer rows, and reproduced the import bytes on a second run.

[Round 7 results](../submissions/round-07-results.json) · [Week 7 selection](../submissions/Week_07/README.md) · [Week 8 snapshot instructions](../data/README.md#week-8-notebook-snapshots)

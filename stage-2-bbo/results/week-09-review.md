# Week 9 review — round 8 observations and refreshed models

The results email arrived on **23 November 2026 at 14:55 GMT**. All eight
returned inputs match the submitted Week 8 batch. The official dataset has
**239 observations**, and five queries remain per function, including Week 9.
Functions **1, 3 and 8** set new bests. Function 3 improved on its starting
best for the first time. [Round 8 record](../submissions/round-08-results.json)

[Full-cohort review and both provisional input lists](week-09-cohort-review.md)

[Official-only alternatives and exact query comparison](week-09-official-alternatives.md)

## Observations and notebooks

| Function | Return | Rank | Best observed | Exploration | Modelling | Notes |
| --- | ---: | --- | ---: | --- | --- | --- |
| 1 | 1.9973557 | 1/18 | 1.9973557 | [Observations](../notebooks/Function_1/Week_09/official/exploration.ipynb) | [Models](../notebooks/Function_1/Week_09/official/model_predictions.ipynb) | [Report](../notebooks/Function_1/Week_09/report.md) |
| 2 | 0.531874267 | 4/18 | 0.723057546 | [Observations](../notebooks/Function_2/Week_09/official/exploration-review.ipynb) | [Models](../notebooks/Function_2/Week_09/official/model_predictions.ipynb) | [Report](../notebooks/Function_2/Week_09/report.md) |
| 3 | -0.00495200082 | 1/23 | -0.00495200082 | [Observations](../notebooks/Function_3/Week_09/official/exploration-review.ipynb) | [Models](../notebooks/Function_3/Week_09/official/model_predictions.ipynb) | [Report](../notebooks/Function_3/Week_09/report.md) |
| 4 | 0.638271373 | 3/38 | 0.666089599 | [Observations](../notebooks/Function_4/Week_09/official/observation_review.ipynb) | [Models](../notebooks/Function_4/Week_09/official/model_predictions.ipynb) | [Report](../notebooks/Function_4/Week_09/report.md) |
| 5 | 8099.17996 | 6/28 | 8662.405 | [Observations](../notebooks/Function_5/Week_09/observation_review.ipynb) | [Models](../notebooks/Function_5/Week_09/model_predictions.ipynb) | [Report](../notebooks/Function_5/Week_09/report.md) |
| 6 | -0.228584024 | 2/28 | -0.21395954 | [Observations](../notebooks/Function_6/Week_09/observation_review.ipynb) | [Models](../notebooks/Function_6/Week_09/model_predictions.ipynb) | [Report](../notebooks/Function_6/Week_09/report.md) |
| 7 | 2.41389007 | 6/38 | 2.73937512 | [Observations](../notebooks/Function_7/Week_09/observation_review.ipynb) | [Models](../notebooks/Function_7/Week_09/model_predictions.ipynb) | [Report](../notebooks/Function_7/Week_09/report.md) |
| 8 | 9.93677342 | 1/48 | 9.93677342 | [Observations](../notebooks/Function_8/Week_09/observation_review.ipynb) | [Models](../notebooks/Function_8/Week_09/model_predictions.ipynb) | [Report](../notebooks/Function_8/Week_09/report.md) |

Rank 1 is the highest observed score for that function. Function 1's two
highest positives came from inputs about 0.000385 apart. Function 6 placed
second with another strong return on the x5 = 0 face. Function 5's isolated
x2 reduction cost about 563.23 points relative to the measured upper corner.

## Archived fits at the submitted inputs

The observation notebooks evaluate the saved Week 8 weekly fits at the exact
coordinates returned in round 8. They use the parameters exported before that
return. Several final choices used separate decision fits or broader peer
evidence. These replays give a like-for-like historical comparison across all
eight functions.

| Function | Units | Actual | Archived GP mean | Archived neural mean |
| --- | --- | ---: | ---: | ---: |
| 1 | Signed-log h | 200.300455 | 197.213528 | 86.9053801 |
| 2 | Raw y | 0.531874267 | 0.732454447 | 0.615044821 |
| 3 | Raw y | -0.00495200082 | -0.0386144742 | -0.0615397043 |
| 4 | Raw y | 0.638271373 | -0.11106153 | 0.106936834 |
| 5 | Raw y | 8099.17996 | 8284.83603 | 8330.07253 |
| 6 | Raw y | -0.228584024 | -0.293258731 | -0.342923121 |
| 7 | Raw y | 2.41389007 | 2.55172174 | 2.60734456 |
| 8 | Raw y | 9.93677342 | 9.93489481 | 9.7363731 |

The nominal Function 2 decision GP expected **0.737960**, with predictive
observation SD **0.03825**, against a return of **0.531874**. The archived
weekly GP for Function 8 predicted **9.934895**, close to the observed
**9.936773**. [Function 2 decision](../notebooks/Function_2/Week_08/official/model_predictions.ipynb)

## Refreshed models

All methods use the same official leave-one-out targets within each function.
Functions 1–4 compare the GP with and without the same retained peers carried
forward from Week 8. The neural column scores the selected architecture's
three-seed mean. Function 1 uses MAE in signed-log units. The others use RMSE
in raw score units.

| Function | Metric | Selected GP training | Mean baseline | GP error | Neural error |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | LOO MAE | Official plus retained peers | 138.70595 | 98.3953555 | 165.799434 |
| 2 | LOO RMSE | Official plus retained peers | 0.242844761 | 0.134494275 | 0.202727433 |
| 3 | LOO RMSE | Official plus retained peers | 0.0769478277 | 0.0460325126 | 0.104684589 |
| 4 | LOO RMSE | Official | 9.58493044 | 1.5269439 | 2.42572311 |
| 5 | LOO RMSE | Official | 3494.24057 | 529.24327 | 380.86264 |
| 6 | LOO RMSE | Official | 0.617181008 | 0.329611475 | 0.289304971 |
| 7 | LOO RMSE | Official | 0.953776769 | 0.285916953 | 0.365961407 |
| 8 | LOO RMSE | Official | 1.09062147 | 0.296346971 | 0.37871984 |

The selected GP had lower held-out error for six functions. Neural ensembles
led for Functions 5 and 6. The estimates support comparison of these fits on
the available targets, with model selection reusing the same folds.

The model notebooks score provisional GP and neural inputs and revisit the
possible follow-ups recorded in the Week 8 lookaheads. Function 1 also refits
six positive-source shapes. Their predicted raw peaks span 1.461 to 2.092.
Function 8's saved follow-up is an example of model disagreement. Its GP
mean is 10.0385 and its neural mean is 9.60733, against the measured best of
9.93677.

## Verification

The Capstone kernel executed all **16 notebooks** and
**192 code cells**. Readback verified the **52 PNG figures**,
**12 interactive HTML files**, full-precision comparison records and
**22 retained peer rows**. The Stage 2 suite passed **149 tests**.
[Reproduction guide](../docs/reproducing-the-analyses.md#week-9-reviews) ·
[Week 9 snapshots](../data/README.md#week-9-notebook-snapshots)

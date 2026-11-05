# Week 7 review: round 6 returns and refreshed models

Results received on **2 November 2026 at 07:28 GMT**, ingested and reviewed on **5 November 2026**. All eight submitted inputs and all earlier cumulative returns match exactly. The official total is **223 observations**. Seven queries remain per function, including Week 7; new candidates are provisional and unsubmitted.

Functions **1, 2, 6, 7 and 8** improve their numerical incumbents. Function 1 remains practically zero. Function 4 ranks second; Function 5's exploration beats every starting observation but falls far short of its best; Function 3 ranks 19th of 21.

| Function | Return | Rank | Observation review | Models | Interpretation |
| --- | ---: | ---: | --- | --- | --- |
| 1 | 2.17091293e-14 | 1/16 | [Notebook](../notebooks/Function_1/Week_07/official/exploration.ipynb) | [Notebook](../notebooks/Function_1/Week_07/official/model_predictions.ipynb) | [Report](../notebooks/Function_1/Week_07/report.md) |
| 2 | 0.723057546 | 1/16 | [Notebook](../notebooks/Function_2/Week_07/official/exploration-review.ipynb) | [Notebook](../notebooks/Function_2/Week_07/official/model_predictions.ipynb) | [Report](../notebooks/Function_2/Week_07/report.md) |
| 3 | -0.121416802 | 19/21 | [Notebook](../notebooks/Function_3/Week_07/official/exploration-review.ipynb) | [Notebook](../notebooks/Function_3/Week_07/official/model_predictions.ipynb) | [Report](../notebooks/Function_3/Week_07/report.md) |
| 4 | 0.489906139 | 2/36 | [Notebook](../notebooks/Function_4/Week_07/official/observation_review.ipynb) | [Notebook](../notebooks/Function_4/Week_07/official/model_predictions.ipynb) | [Report](../notebooks/Function_4/Week_07/report.md) |
| 5 | 1616.62575 | 6/26 | [Notebook](../notebooks/Function_5/Week_07/observation_review.ipynb) | [Notebook](../notebooks/Function_5/Week_07/model_predictions.ipynb) | [Report](../notebooks/Function_5/Week_07/report.md) |
| 6 | -0.328192388 | 1/26 | [Notebook](../notebooks/Function_6/Week_07/observation_review.ipynb) | [Notebook](../notebooks/Function_6/Week_07/model_predictions.ipynb) | [Report](../notebooks/Function_6/Week_07/report.md) |
| 7 | 2.73937512 | 1/36 | [Notebook](../notebooks/Function_7/Week_07/observation_review.ipynb) | [Notebook](../notebooks/Function_7/Week_07/model_predictions.ipynb) | [Report](../notebooks/Function_7/Week_07/report.md) |
| 8 | 9.78294276 | 1/46 | [Notebook](../notebooks/Function_8/Week_07/observation_review.ipynb) | [Notebook](../notebooks/Function_8/Week_07/model_predictions.ipynb) | [Report](../notebooks/Function_8/Week_07/report.md) |

## What the returns change

Function 6's neural-led trial produces its first improvement since round 1. Function 8 first exceeds its initial best. Function 4's second nearby positive result exposes persistent GP underprediction. Function 5's new exploration score substantially exceeds both saved negative model means, while Function 3 misses both means badly.

The refreshed Function 2 comparison now selects official-only training. Function 3's provisional GP candidate moves away from the disappointing low-x3 boundary. Function 5's shortlist returns to the unfinished isolated x3 check. These are model-review outcomes, not final query decisions.

## Forecasts retained before refitting

The final Week 6 selection supplies predictions at the actual submitted inputs, including the changed choices for Functions 1, 5 and 6. Exact file hashes and coordinates are checked. Function 1 uses signed-log units; its raw return remains 2.1709129251221444e-14.

| Function | Units | Actual | Saved GP mean | Saved neural mean |
| --- | --- | ---: | ---: | ---: |
| 1 | signed-log height h | 186.336642 | 184.023849 | -219.54375 |
| 2 | output y | 0.723057546 | 0.588435236 | 0.500716539 |
| 3 | output y | -0.121416802 | -0.0517369697 | -0.0137746368 |
| 4 | output y | 0.489906139 | -0.408796351 | 0.229630537 |
| 5 | output y | 1616.62575 | -685.995383 | -13.5148153 |
| 6 | output y | -0.328192388 | -0.394067823 | -0.36349738 |
| 7 | output y | 2.73937512 | 2.48135293 | 2.44391511 |
| 8 | output y | 9.78294276 | 9.81224155 | 9.61440882 |

## Refreshed predictive comparisons

| Function | Criterion | Selected GP training | GP error | Neural seed-mean error | Lower error |
| --- | --- | --- | ---: | ---: | --- |
| 1 | LOO MAE | official_plus_retained_peers | 109.584157 | 193.057461 | Selected GP |
| 2 | LOO RMSE | official | 0.116089118 | 0.239601869 | Selected GP |
| 3 | LOO RMSE | official_plus_retained_peers | 0.0512737995 | 0.120022712 | Selected GP |
| 4 | LOO RMSE | official | 1.52441017 | 2.30351136 | Selected GP |
| 5 | LOO RMSE | official | 570.712259 | 381.042561 | Neural |
| 6 | LOO RMSE | official | 0.33445885 | 0.310913468 | Neural |
| 7 | LOO RMSE | official | 0.276568257 | 0.39998438 | Selected GP |
| 8 | LOO RMSE | official | 0.307339018 | 0.35355053 | Selected GP |

Selected GPs lead for six functions; neural ensembles lead for Functions 5 and 6. The same official held-out target rows are used in each comparison, but peer-assisted GPs have additional training data. Selection uses these errors, so they remain optimistic estimates on small, adaptively sampled datasets. The GP query policy is retained pending a separate final selection; neural seed variation is not GP posterior uncertainty.

## Verification

All **sixteen notebooks, 184 code cells**, were executed with the Capstone-local interpreter, saving **52 embedded PNGs and 64 figure files**. The focused round-import, weekly-observation, exploration, plotting, prediction and cohort helper suites passed **56 tests**. Readback confirmed exact CSV/NumPy agreement, unchanged hashes for **181 historical notebook/data/result files**, retained selection-source hashes and repeat-import idempotency. Observation and model charts were visually inspected. The new explorers' JavaScript GP/neural calculations matched Python at four inputs per function; browser controls were not manually smoke-tested. The local `data/week-07-verification-2026-11-05.json` retains the detailed checks. No course evaluation, portal submission or publication was performed.

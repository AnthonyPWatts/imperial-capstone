# Week 6 model prediction review

The subsequent [final selection review](../submissions/Week_06/README.md) has
finalised the [eight-line input file](../submissions/Week_06/submissions.txt).
Submission was confirmed by the project owner on **31 October 2026**; results
are pending and **seven future queries remain per function**.
Functions 2–4, 7 and 8 retain their notebook proposals. Function 1 now follows
the agreed central-region hunch at `0.537942-0.522217`, its saved signed-mean
maximum with no uncertainty bonus. None of its outputs is regarded as clearly
non-zero in practical or source-detection terms. Function 5 tests a
global information/coverage gap, while Function 6 uses the saved neural
candidate for one focused, GP-cross-checked query. The model evidence below
retains its original provisional decisions. The exact submitted inputs and
owner confirmation are recorded separately from the model predictions.

All eight model notebooks use the **215 confirmed official observations through round 5**, received on **30 October 2026**. Each has a bounded nine-configuration neural comparison and a standalone GP/neural input explorer. Functions 1–4 also compare unchanged, audited peer training rows from 20 October on the same official held-out targets.

The selected GP has lower declared held-out error for **six functions**. The neural seed ensemble has lower RMSE for **Functions 5 and 6**: the Function 5 difference is substantial, while Function 6's is modest. Neural models therefore remain useful comparators. These small adaptive datasets do not establish a universal winner or an independent optimisation-policy result.

Function 1 has completed its fixed coverage plan and now proposes a signed-score GP-UCB query. Its monotone signed-log transform preserves output ordering; magnitude remains a separate diagnostic. Its predictive improvement over the mean baseline is modest and its proposal uncertainty remains broad. Means and SDs are reported in signed-log units, with no raw posterior mean inferred.

## Identical official held-out targets

Errors use the units and metric in each row. Do not compare error magnitudes across different functions. Compare **OfficialGP** and **Neural** for model families trained on identical official rows. **QueryGP** separately shows the selected policy fit, including additional retained peer training where chosen. Neural architecture selection averages all three seeds; the Neural column then scores their mean held-out predictions. All columns score the same official targets. Selection optimism and adaptive sampling apply to both families.

| Function | Metric | Units | Baseline | OfficialGP | Neural | QueryGP | Smaller | Scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LOO MAE | signed-log height h | 132.332 | 123.174 | 136.574 | 119.636 | Official-only GP | official plus retained peers |
| 2 | LOO RMSE | output y | 0.209025 | 0.123352 | 0.234339 | 0.111379 | Official-only GP | official plus retained peers |
| 3 | LOO RMSE | output y | 0.07986 | 0.0739004 | 0.106502 | 0.0486971 | Official-only GP | official plus retained peers |
| 4 | LOO RMSE | output y | 8.95281 | 1.51557 | 2.48624 | 1.51557 | Official-only GP | official |
| 5 | LOO RMSE | output y | 3214.24 | 329.432 | 136.779 | 329.432 | Neural seed ensemble | official |
| 6 | LOO RMSE | output y | 0.541677 | 0.339648 | 0.326054 | 0.339648 | Neural seed ensemble | official |
| 7 | LOO RMSE | output y | 0.796164 | 0.266487 | 0.410558 | 0.266487 | Official-only GP | official |
| 8 | LOO RMSE | output y | 1.01724 | 0.313369 | 0.368572 | 0.313369 | Official-only GP | official |

## Per-function evidence

- **Function 1:** [report](../notebooks/Function_1/Week_06/report.md), [executed notebook](../notebooks/Function_1/Week_06/official/model_predictions.ipynb), [offline explorer](../notebooks/Function_1/Week_06/official/figures/neural-network-explorer.html).
- **Function 2:** [report](../notebooks/Function_2/Week_06/report.md), [executed notebook](../notebooks/Function_2/Week_06/official/model_predictions.ipynb), [offline explorer](../notebooks/Function_2/Week_06/official/figures/neural-network-explorer.html).
- **Function 3:** [report](../notebooks/Function_3/Week_06/report.md), [executed notebook](../notebooks/Function_3/Week_06/official/model_predictions.ipynb), [offline explorer](../notebooks/Function_3/Week_06/official/figures/neural-network-explorer.html).
- **Function 4:** [report](../notebooks/Function_4/Week_06/report.md), [executed notebook](../notebooks/Function_4/Week_06/official/model_predictions.ipynb), [offline explorer](../notebooks/Function_4/Week_06/official/figures/neural-network-explorer.html).
- **Function 5:** [report](../notebooks/Function_5/Week_06/report.md), [executed notebook](../notebooks/Function_5/Week_06/model_predictions.ipynb), [offline explorer](../notebooks/Function_5/Week_06/figures/neural-network-explorer.html).
- **Function 6:** [report](../notebooks/Function_6/Week_06/report.md), [executed notebook](../notebooks/Function_6/Week_06/model_predictions.ipynb), [offline explorer](../notebooks/Function_6/Week_06/figures/neural-network-explorer.html).
- **Function 7:** [report](../notebooks/Function_7/Week_06/report.md), [executed notebook](../notebooks/Function_7/Week_06/model_predictions.ipynb), [offline explorer](../notebooks/Function_7/Week_06/figures/neural-network-explorer.html).
- **Function 8:** [report](../notebooks/Function_8/Week_06/report.md), [executed notebook](../notebooks/Function_8/Week_06/model_predictions.ipynb), [offline explorer](../notebooks/Function_8/Week_06/figures/neural-network-explorer.html).

## Query status and limits

Every notebook saves a provisional input; none submits it. Eight weekly opportunities were available including Week 6 at preparation; seven remain after the owner-confirmed submission. The GP acquisition policy retains UCB with weight 1.96 and explicit weight sensitivity. Neural alternatives use mean predictions across seeds; seed variation is not calibrated predictive uncertainty. Function 5 retains its 0.02 spacing safeguard and log-target median comparator.

The new numerical and explorer helpers reuse the established modelling libraries and add no dependencies. Earlier snapshots, observation reviews and submissions remain intact. The [notebook guide](../notebooks/README.md) links the measured reviews and model notebooks; the [execution guide](../docs/reproducing-the-analyses.md) describes the environment and checks.

## Verification

All eight prediction notebooks executed with the capstone-local interpreter:
**96 code cells**, **27 embedded PNG charts** and **eight offline explorers**.
The new prediction/export tests and existing model, surrogate, cohort and
Function 1 helper tests passed **51 tests**. Readback verified saved notebook
schemas, complete executions, numerical records, proposal bounds/precision,
duplicate exclusions and unchanged historical notebooks, snapshots and submissions.

JavaScript inference matched Python for all **72 neural fits and eight GPs**
at 13 inputs per function, including boundaries, the incumbent and proposed
query. Browser smoke tests covered Function 1's reset, input, architecture and
curve-axis controls and Function 8's eighth input and alternate architecture.
Representative validation, conditional-response and acquisition figures were
visually inspected. Other per-function browser interactions were not manually
tested; their prediction engines passed the same numerical parity checks.

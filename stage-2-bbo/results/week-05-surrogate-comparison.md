# Week 5 neural-network and surrogate comparison

**Study date: 29 October 2026.** Eight executed notebooks compare numerical-score
surrogates and good/bad classifiers using **207 official observations through
round 4**. These retrospective studies follow the fifth-query submission. Its
results are pending, its input file is unchanged and none of the hypothetical
queries has been evaluated by the course portal.

The [final Week 5 discussion draft](../docs/week-05-discussion-draft.md) records
the posted reflection and the interpretation of the weak F3 classifier and
suspected F5 boundary maximum, with their limitations.

## Regression: compare every seed

The NN column reports the architecture with the lowest mean leave-one-out RMSE
across seeds 11, 42 and 73. The spread is one seed standard deviation, not a
confidence interval. Seed 42 is the fixed representative for gradients and plots.
The reference is the lowest-RMSE non-neural method in the declared matrix.
Function 1 uses signed-log score units; other rows use original score units.
Compare methods within each row rather than raw errors between functions.

| Function | NN architecture | NN mean RMSE ± seed SD | Best non-neural method | Non-neural RMSE | NN / non-neural error |
| --- | --- | ---: | --- | ---: | ---: |
| [1](../notebooks/Function_1/Week_05/official/surrogate_comparison.ipynb) | tanh-8 | 160.17 ± 7.04 | RBF SVR | 132.696 | 1.207 |
| [2](../notebooks/Function_2/Week_05/official/surrogate_comparison.ipynb) | relu-8-8 | 0.24831 ± 0.0193 | GP x1 only | 0.145783 | 1.703 |
| [3](../notebooks/Function_3/Week_05/official/surrogate_comparison.ipynb) | tanh-16 | 0.113431 ± 0.00244 | GP separate length scales | 0.0757028 | 1.498 |
| [4](../notebooks/Function_4/Week_05/official/surrogate_comparison.ipynb) | relu-8-8 | 2.7748 ± 0.232 | GP shared length scale | 1.46679 | 1.892 |
| [5](../notebooks/Function_5/Week_05/surrogate_comparison.ipynb) | tanh-8 | 126.903 ± 5.97 | GP shared length scale | 334.684 | 0.379 |
| [6](../notebooks/Function_6/Week_05/surrogate_comparison.ipynb) | relu-8-8 | 0.376118 ± 0.0269 | RBF SVR | 0.285722 | 1.316 |
| [7](../notebooks/Function_7/Week_05/surrogate_comparison.ipynb) | tanh-8 | 0.453814 ± 0.078 | GP shared length scale | 0.257475 | 1.763 |
| [8](../notebooks/Function_8/Week_05/surrogate_comparison.ipynb) | tanh-16 | 0.380952 ± 0.0254 | GP separate length scales | 0.205328 | 1.855 |

The selected neural architecture's mean RMSE beats the best declared non-neural alternative for **Function 5**.

Some individual seeds rank better than their architecture average. Those
outcomes remain visible rather than being promoted as stable model improvements.

## Classification: good/bad boundaries

Each fold labels a test result good if it reaches the training outcomes' 75th
percentile, estimated without that held-out outcome. Full-fit boundary plots
use a separately recorded full-data threshold. Balanced accuracy gives the
two classes equal weight; the notebooks also retain recall, precision,
false negatives/positives and probability metrics where available.

| Function | NN architecture | NN mean balanced accuracy ± seed SD | Logistic regression | RBF SVM |
| --- | --- | ---: | ---: | ---: |
| 1 | relu-8-8 | 0.825 ± 0.000 | 0.950 | 0.825 |
| 2 | relu-8-8 | 0.775 ± 0.000 | 0.775 | 0.775 |
| 3 | relu-8-8 | 0.567 ± 0.051 | 0.464 | 0.500 |
| 4 | tanh-16 | 0.878 ± 0.012 | 0.531 | 0.904 |
| 5 | relu-8-8 | 0.929 ± 0.000 | 0.929 | 0.929 |
| 6 | tanh-8 | 0.796 ± 0.016 | 0.778 | 0.833 |
| 7 | relu-8-8 | 0.816 ± 0.012 | 0.809 | 0.758 |
| 8 | tanh-8 | 0.830 ± 0.024 | 0.776 | 0.625 |

The classification comparison can favour a network even where a GP gives better
numerical predictions. Classification discards score magnitude and remains a
secondary framing of maximisation. A flexible fitted boundary is not a known
true boundary, and a probability near 0.5 can reflect noisy ambiguity as well
as lack of information.

## Inputs, support vectors and seven reflection sections

All studies identify actual SVM support vectors and check neural input gradients
against finite differences. The input ranking below averages absolute regression
gradients across observed locations and the selected architecture's three seeds.
It measures sampled model sensitivity, rather than causal or global importance.

| Function | Largest sampled NN sensitivities | RBF support vectors / observations | Full report |
| --- | --- | ---: | --- |
| 1 | x1, x2 | 9 / 14 | [Seven reflection sections and figures](../notebooks/Function_1/Week_05/official/surrogate-study/report.md) |
| 2 | x1, x2 | 10 / 14 | [Seven reflection sections and figures](../notebooks/Function_2/Week_05/official/surrogate-study/report.md) |
| 3 | x3, x2, x1 | 14 / 19 | [Seven reflection sections and figures](../notebooks/Function_3/Week_05/official/surrogate-study/report.md) |
| 4 | x2, x1, x3 | 20 / 34 | [Seven reflection sections and figures](../notebooks/Function_4/Week_05/official/surrogate-study/report.md) |
| 5 | x2, x1, x3 | 14 / 24 | [Seven reflection sections and figures](../notebooks/Function_5/Week_05/surrogate-study/report.md) |
| 6 | x2, x1, x4 | 16 / 24 | [Seven reflection sections and figures](../notebooks/Function_6/Week_05/surrogate-study/report.md) |
| 7 | x2, x1, x5 | 22 / 34 | [Seven reflection sections and figures](../notebooks/Function_7/Week_05/surrogate-study/report.md) |
| 8 | x3, x1, x7 | 32 / 44 | [Seven reflection sections and figures](../notebooks/Function_8/Week_05/surrogate-study/report.md) |

The bounded experiment ran **144 final neural fits** and **6,250 estimator fits** including validation and other methods.

## Limits and reproducibility

The declared neural configurations are tanh-8, tanh-16 and ReLU-8-8 with three
initialisations each, L-BFGS, L2 alpha 0.1 and a maximum of 400 iterations.
Regression also includes the mean baseline, linear/ridge methods, kernel ridge,
RBF SVR and GPs with shared/separate length scales. Function 2 retains its x1-only
GP hypothesis. Input and regression target scaling are fitted inside folds.

Convergence and kernel-bound warnings remain in the saved results. These are
warning-message counts, not failed-fit counts: one GP fit can warn about several
bounded coordinates. Iteration-limited networks may behave differently with a
larger training budget; this study does not prove that an architecture is
intrinsically unsuitable. Neural seed variation is optimisation sensitivity,
not calibrated posterior uncertainty.

The same validation outcomes are reused to compare architectures and reference
models. There is no independent final test set or statistical guarantee for the
selected configuration. The observed inputs were collected adaptively, and
higher-dimensional boundary views are conditional slices with other inputs
held fixed. Off-slice observations are not projected onto them.

Run from the Capstone root:

```powershell
.\run-python.ps1 stage-2-bbo/scripts/run_surrogate_studies.py
```

Use `--functions 3 8` for a targeted rerun. Each study saves an executed notebook,
seven PNG figures, a source-bound result record and a report covering all seven
reflection criteria. The runner refreshes this index once all eight results are
available. See the [execution guide](../docs/reproducing-the-analyses.md#week-5-discussion-surrogate-comparisons).

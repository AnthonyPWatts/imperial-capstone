# Week 7 prepared inputs

**All eight choices are agreed and ready for portal submission as of
10 November 2026.** Functions 1 and 2 were agreed on 5 November. The
selection uses 223 official observations through round 6. Seven query
opportunities remain per function, including this batch.

The portal file contains one six-decimal input per line in Function 1–8
order. [Portal inputs](submissions.txt) · [Structured decisions](proposals.json) ·
[Executed selection review](selection_review.ipynb) · [Verification](verification.json)

| Function | Agreed portal input | Selection |
| --- | --- | --- |
| 1 | `0.627916-0.628953` | Positive source peak |
| 2 | `0.716870-0.000000` | Local GP-UCB at weight 1.96 |
| 3 | `0.495233-0.568579-0.295191` | Previously considered mean-led point near the incumbent |
| 4 | `0.419096-0.406060-0.364200-0.426466` | Isolate x3 using the retained positive peer result |
| 5 | `0.999999-0.999999-0.979593-0.999999` | Complete the upper-corner x3 check |
| 6 | `0.414821-0.260324-0.607614-0.728412-0.000000` | Local neural candidate with a GP cross-check |
| 7 | `0.000000-0.135660-0.364055-0.203076-0.342712-0.663441` | Local GP-UCB at weight 1.96 |
| 8 | `0.000000-0.126300-0.212455-0.132451-0.899004-0.732961-0.231231-0.808273` | Highest local GP mean after screening with both fits |

## Function 1: test the positive-source hypothesis

Use **`0.627916-0.628953`** for Week 7. This tests the analytic peak of a
circular positive Gaussian fitted to the strongest five positive readings
from the 16 official observations through round 6. No peer location or output
enters fitting or query selection. The six-reading and axis-aligned full-data
controls put their peaks nearby.

The source's signed-log held-out MAE is **91.662**, versus **135.652** for
the best official-only GP; its RMSE is worse (**155.154 versus 147.135**).
The log-magnitude GP remains better at predicting that target. The source
cannot represent negative responses, and larger positive subsets can
destabilise its extrapolations. Its conditional predicted height of **0.7081**
is not a calibrated forecast of the actual signed score.

The selected point passes dimension, bounds, six-decimal format and duplicate
checks against the official observations and all six prior submission files.
Agreement records the input choice, with no portal submission performed.

[Source comparison](../../notebooks/Function_1/Week_07/source-shape.md) ·
[Executed source notebook](../../notebooks/Function_1/Week_07/official/source_shape_comparison.ipynb) ·
[Week 7 review](../../results/week-07-review.md)

## Function 2: refine the successful lower-edge region

Use **`0.716870-0.000000`**, with GP-UCB uncertainty weight **1.96**.
The separate-length-scale GP is fitted to the 16 official observations
through round 6. Its held-out RMSE is **0.116089**, versus **0.240371** for
the mean baseline and **0.239602** for the selected neural seed mean.

This mainly changes x1 from the latest successful `0.700520-0.000114`,
which returned **0.723058**, while retaining x2 near its lower boundary.
At the agreed point the GP mean is **0.703455**, latent SD **0.037765**,
and weight-1.96 UCB **0.777474**. The mean remains below the measured best;
UCB is a selection score, not a forecast of improvement. The point is
outside the sampled hull and **0.01635** from the nearest known input.

Cohort discussion supports investigating the x1 neighbourhood around 0.70,
but does not resolve x2's influence or distinguish nearby six-decimal
choices. One report describes a new best near the upper x2 boundary;
another describes different returns at near-repeated inputs. These are
contextual self-reports, not additional official observations or a measured
noise level. They do not enter the selected official-only fit.

Dimension, bounds, formatting and duplicate checks passed against the
official/retained-peer snapshot and all six earlier submission files.
The input, weight and caveats are agreed; nothing has been submitted.

[Function 2 review](../../notebooks/Function_2/Week_07/report.md) ·
[Executed models](../../notebooks/Function_2/Week_07/official/model_predictions.ipynb) ·
[Numerical evidence](../../notebooks/Function_2/Week_07/official/model-review.json)

## Functions 3–8

- **Function 3** revisits the Week 6 mean-led option near the measured best
  after the exploratory low-x3 test. The refreshed peer-assisted GP predicts
  **−0.03289**, against **−0.03484** observed at the incumbent. The neural
  mean is **−0.05490**. [Week 7 review](../../notebooks/Function_3/Week_07/report.md)
- **Function 4** keeps x1, x2 and x4 at the official incumbent and sets x3
  to **0.364200**, the x3 value of the retained peer's **0.712146** return.
  The saved GP predicts **−0.2842** at the agreed point and the neural mean
  is **0.2709**. The test is motivated by nearby positive measurements and
  isolates the change from our incumbent. [Week 7 review](../../notebooks/Function_4/Week_07/report.md)
- **Function 5** reduces x3 alone from the upper-corner incumbent. All four
  tested GP weights choose the same input. The GP and neural means are
  **8368.78** and **8351.01**, below the observed **8662.41**. The nearest
  known point is **0.020406** away, above the **0.02** minimum.
  [Week 7 review](../../notebooks/Function_5/Week_07/report.md)
- **Function 6** follows the successful neural-led trial with a small step on
  the x5=0 face. The neural mean is **−0.34130** and the GP mean is
  **−0.36391**, both below **−0.32819** at the incumbent. Both fits rank this
  point above the more exploratory GP-UCB candidate.
  [Week 7 review](../../notebooks/Function_6/Week_07/report.md)
- **Function 7** retains the weight-1.96 local GP-UCB point, mainly lowering
  x4. The GP and neural means are **2.64269** and **2.77652**, respectively,
  against the measured **2.73938**. [Week 7 review](../../notebooks/Function_7/Week_07/report.md)
- **Function 8** uses the existing ±0.15 per-coordinate box around the
  new incumbent. Of **4,352** local candidates, **67** have both fitted
  means above **9.78294**. The highest GP mean among those candidates is
  **9.87886**, with neural mean **9.79206**. The predicted gains are small
  relative to held-out errors. [Week 7 review](../../notebooks/Function_8/Week_07/report.md)

The selection notebook records forecasts at the exact prepared points. The
Function 1 source height uses raw output units, while its GP and neural
cross-checks use signed-log units. Source hashes, all 245 official and
retained peer inputs, and the six earlier submission files were checked.

From the capstone root, verify or reproduce the portal input file with:

```powershell
.\run-python.ps1 stage-2-bbo/scripts/review_week_07_queries.py
```

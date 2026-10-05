# Function 1: first returned result

Reviewed on **5 October 2026**. The Week 1 results email arrived at **00:38 BST**
(4 October, 23:38 UTC). All eight returned points match the saved first-query
proposals; the email body and both attachments agree. The complete return is
recorded in [round-01-results.json](../submissions/round-01-results.json).
This analysis covers Function 1 only.

## What changed

| Measure | Result |
| --- | ---: |
| Evaluated input | `(0.468429, 0.435440)` |
| Returned output | `-0.00550934555381189` |
| Observations | 10 → 11 |
| Previous minimum | `-0.0036060626443634764` |
| Best observed output, unchanged | `7.710875114502849e-16` |
| Best observed location, approximately | `(0.731024, 0.733000)` |
| Improvement in the best observed output | 0 |

The result is the lowest of all eleven observations. It is `0.00190328` below
the previous minimum, or **52.78% larger in negative magnitude**. Because this
is a maximisation task, that is a worse response. A percentage relative to the
near-zero positive incumbent would be misleading.

There are now **two sizeable negative observations**, rather than just one;
the other nine still have absolute values at most `7.71e-16`. Both negative
results are retained. Nothing in the return justifies clipping them, treating
them as corrupt, or replacing very small non-zero observations with zero.
The sign count changes from seven positive/three negative to seven positive/four
negative.

## What the new location tells us

The new point fills part of the previously unobserved central area. It lies
`0.305889` units from the old minimum near `(0.650114, 0.681526)`. Its nearest
initial neighbour is a different, almost-zero observation, `0.293669` units away.
Two sizeable negative responses at distinct locations broaden the evidence for
negative behaviour. They do **not** establish a continuous valley, its width,
the sign between the points, or the location of a positive peak. Noise and
repeatability also remain unmeasured.

The query was chosen to improve geometric coverage without using outputs.
Mean distance from a uniformly selected location in the unit square to its
nearest observation fell from **0.152412231 to 0.140923767**, a **7.53776%**
reduction. The achieved coverage agrees with the original calculation. This
is a reduction in sampling gaps, not a measured gain in predictive accuracy
or a probability of finding the optimum.

Local figures, reproduced by the command below:

- [Locations and signed output magnitudes](../../.runtime/function-1-round-01/result-and-scale.png)
- [Coverage before and after, using shared axes and colour limits](../../.runtime/function-1-round-01/coverage-before-after.png)

These figures stay local with the restricted observations. In the magnitude
plot, a large negative magnitude remains a poor output; output rank 1 is best.

## Do the existing models now earn more trust?

The existing additive, local and separate-length Gaussian-process families were
refitted under the same assumptions, including an assumed noise standard
deviation of 10% of the training outputs' standard deviation. Each leave-one-out
check fitted only the remaining observations. Both the original ten-row and
updated eleven-row checks were recomputed with the same current code.

| Prediction method | Initial leave-one-out RMSE | Updated leave-one-out RMSE |
| --- | ---: | ---: |
| Training-mean baseline | 0.00120202 | **0.00198452** |
| Additive GP | 0.00121296 | 0.00208218 |
| Local GP | 0.00122922 | 0.00205155 |
| Separate-length GP | 0.00123546 | 0.00208306 |
| Constant zero reference | 0.00114034 | 0.00198532 |

The two columns use different observation sets. Their absolute increase alone
does not show that adding data worsened the models; compare each model with
the baseline on the same rows.

All three models remain worse than the training-mean baseline on updated RMSE
(roughly 3.4–5.0% worse), and also on updated mean absolute error. The two
negative values dominate squared error, so these scores say little about the
models' ability to locate an unseen positive peak. Zero is a useful descriptive
reference for this scale pattern; its inclusion is exploratory, not an
independently selected model or a claim that tiny values are literally zero.

Retrospective fits using only the initial ten observations predict about
`-0.000253` to `-0.000361` at the evaluated input. The actual result is about
4.75–4.78 assumed predictive standard deviations lower, including the assumed
observation noise. This is evidence of a poor forecast under those models,
**not a calibrated significance test**. Their fitted length scales also hit
the configured lower bound. Neither noise nor the kernel assumptions are
validated by this small dataset. No wider kernel search or updated multi-round
knowledge-gradient simulation was run.

## Implication for the next query

There is still insufficient evidence to switch confidently from coverage to
exploitation of a fitted response surface. Continuing the declared early-round
coverage strategy gives **`(0.112627, 0.828222)`**, portal string
**`0.112627-0.828222`**. This is the next point in the earlier two-point plan,
now independently recomputed with the first returned location fixed.

It would reduce mean sampling distance from `0.140923767` to `0.130673391`:
**7.27370%** relative to the current eleven observations. The candidate has valid
bounds and precision and does not duplicate any observation. It is a
conditional coverage candidate, not a predicted maximiser or a submitted query.
One of the 13 planned query rounds has now been evaluated; 12 remain.

## Course lifecycle checkpoint

| Step | Application in this review |
| --- | --- |
| 1. Define the goal and scope | Maximise Function 1; review the first returned evaluation. |
| 2. Gather the data | Preserve the complete email return and reconcile it to the proposals. |
| 3. Explore the data | Compare ranks, signs, scale and spatial coverage before and after. |
| 4. Clean and preprocess the data | Validate dimensions, finite values, bounds and pairing; retain extremes and original precision. GP normalisation stays within each training fold. |
| 5. Select and engineer features | Retain both supplied coordinates; feature engineering is not justified. |
| 6. Define the machine-learning task | Sequential maximisation, with regression surrogates as decision aids. |
| 7. Partition the data | Leave-one-out checks; a separate hold-out set is not used with only eleven adaptive observations. |
| 8. Select and train candidate methods | Reuse the three existing GP families and simple prediction baselines. |
| 9. Evaluate and interpret the results | No incumbent gain; better coverage; no held-out advantage for the tested GPs. |
| 10. Deploy and iterate | Update local observations and record a conditional next candidate; deployment/submission is not part of this review. |

## Reproduction and verification

From the capstone repository root, using its existing environment:

```powershell
.venv/Scripts/python.exe stage-2-bbo/scripts/import_round_results.py --round 1
.venv/Scripts/python.exe stage-2-bbo/scripts/analyse_function_1_round_01.py
.venv/Scripts/python.exe -m unittest discover -s stage-2-bbo/tests -p test_round_results.py -v
```

The full Stage 2 suite also passed: **36 tests**, using
`.venv/Scripts/python.exe -m unittest discover -s stage-2-bbo/tests -v`.
All 16 initial-array hashes and the starter ZIP hash match the September intake
manifest. CSV values round-trip exactly to the NumPy arrays for all eight
functions; a second import produces byte-identical datasets and results records.
`git diff --check` passed. The local
[intake verification](../../.runtime/function-1-round-01/intake-verification.json)
records the data checks.

The import rebuilds all eight accumulated datasets from the unchanged initial
arrays and unique round sources, so rerunning it does not append observations
twice. Original arrays, source evidence and accumulated datasets remain ignored
by Git. The original proposal record remains an unchanged historical snapshot;
the new results record confirms evaluation without inventing a submission date.

Numerical checks, runtime versions, source hashes and figure hashes are saved in
the local [analysis.json](../../.runtime/function-1-round-01/analysis.json).
The model and coverage commands completed successfully, and both figures were
visually inspected. This is a bounded descriptive/model review; it does not
validate the unknown function globally or demonstrate an optimal next query.

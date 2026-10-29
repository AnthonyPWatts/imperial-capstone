# Week 5 submitted inputs

**Prepared on 28 October 2026. Submission confirmed by the project owner on
29 October 2026. Status: submitted; awaiting results.** The exact portal
submission timestamp was not supplied.

[`submissions.txt`](submissions.txt) contains exactly eight lines, ordered from
Function 1 to Function 8, preserving the submitted inputs. The file contains
no labels or extra spaces. Eight future weekly query opportunities remain
per function after this fifth submission. Results through round 4 remain the
latest confirmed feedback; the official datasets still contain 207 observations.

The selections use the 207 official observations through round 4, received on
23 October. Functions 2 and 3 additionally use the retained, audited 20 October
peer snapshot. These peer reports remain self-reported evidence and do not
change the official progress history or query allowance. No source repository
was refreshed. The original official and cohort analyses remain intact.

## Selected inputs

| Function | Portal input | Basis |
| --- | --- | --- |
| 1 | `0.912140-0.861783` | official fixed coverage plan |
| 2 | `0.636151-0.495605` | official plus retained cohort |
| 3 | `0.658460-0.558234-0.006012` | official plus retained cohort |
| 4 | `0.419096-0.406060-0.408855-0.426466` | official fit with saved common candidate review |
| 5 | `0.999999-0.999999-0.999999-0.979636` | official only |
| 6 | `0.310003-0.303475-0.617318-0.965962-0.334839` | official only |
| 7 | `0.000000-0.257992-0.509505-0.318385-0.371190-0.725014` | official only |
| 8 | `0.000000-0.000000-0.000000-0.000000-0.000000-0.000000-0.000000-0.999999` | official only |

## Selection review

- **Function 1:** Retain the final point in the predetermined exploration shortlist. See the [Week 5 report](../../notebooks/Function_1/Week_05/report.md).
- **Function 2:** Adopt the pooled x1-only GP proposal: paired official-target RMSE falls from 0.145783 to 0.121879 (16.4%), with lower MAE as well. The x2 setting remains a geometric tie-break. See the [Week 5 report](../../notebooks/Function_2/Week_05/report.md).
- **Function 3:** Adopt the pooled shared-length-scale GP proposal: its RMSE on the same 19 official targets is 0.047850, versus 0.075703 for the official selected GP (36.8% lower). MAE is slightly higher than the official selected GP; retain the declared RMSE criterion. The new point is extrapolative. See the [Week 5 report](../../notebooks/Function_3/Week_05/report.md).
- **Function 4:** Retain the official GP: adding peers raises paired RMSE from 1.466791 to 1.543057 and MAE from 0.988323 to 1.122493. Adopt the already saved common-pool winner under that official fit: UCB 0.152263 versus 0.139098 at the initial official proposal. See the [Week 5 report](../../notebooks/Function_4/Week_05/report.md).
- **Function 5:** Retain raw-output GP selection, including the log-target comparator. Apply an explicit 0.02 minimum Euclidean input distance to avoid a near-repeat after two non-improving corner refinements; test x4 while retaining the other three upper-bound settings. See the [Week 5 report](../../notebooks/Function_5/Week_05/report.md).
- **Function 6:** Retain the established GP/UCB choice from the full-space candidate pool; its mean is below the incumbent and the query is exploratory. See the [Week 5 report](../../notebooks/Function_6/Week_05/report.md).
- **Function 7:** Refine the region of the new round-4 incumbent with the GP/UCB local candidate. Its higher uncertainty, rather than a higher predicted mean, supports selection. See the [Week 5 report](../../notebooks/Function_7/Week_05/report.md).
- **Function 8:** Select an unmeasured corner by GP/UCB after comparing full-space, incumbent-local, latest-return-local and geometric coverage alternatives. The point is extrapolative and weight-sensitive. See the [Week 5 report](../../notebooks/Function_8/Week_05/report.md).

## Verification and limits

The four new model notebooks were executed with the Capstone-local Python,
with all code cells completed and no saved errors. Models were refitted inside
each leave-one-out fold; the GP had the lowest held-out RMSE among the methods
tested for each of Functions 5–8. Its RMSE reductions against the mean baseline
were 88.7%, 35.6%, 64.7% and 68.8%, respectively. Function 5 also retains the
log-target GP comparison: it has lower MAE but higher RMSE than the raw-output GP.

The existing helper checks passed: `test_model_review_helpers.py` (12 tests)
and `test_cohort_review_helpers.py` (10 tests), each run with
`.\run-python.ps1 -m unittest discover -s stage-2-bbo/tests -p <test-file> -v`.
Readback also verified all four notebook schemas and completed executions,
their 12 embedded charts and 12 PNG files, 38 local report links and source hashes.

Dimensions, finite coordinates, `[0, 1)` bounds, six-decimal formatting, function
order, absence of spaces, the eight-line count and duplicates were checked.
Each point was compared with the initial and returned official observations,
the retained cohort observations where available, and all four earlier
submission files. Function 5 also passes its declared 0.02 spacing constraint.

[`proposals.json`](proposals.json) preserves the chosen inputs, model evidence,
selection rationale, source-file SHA-256 hashes, prepared-file hash and checks.
The original notebooks retain their provisional proposal histories; this review
records which alternatives were chosen for submission. The confirmation date,
source and pending-result status are recorded separately from the preparation
and validation evidence.

These are unevaluated inputs. GP posterior SD and UCB are model-dependent;
UCB is not a returned output or a calibrated guarantee. The selected model means
for Functions 2 and 4–8 are below their respective official incumbents. Function
3's mean is only 0.001551 above its incumbent, compared with model SD 0.065885;
its selection is also driven largely by exploration. Its peer-informed fit improves
RMSE but has slightly higher MAE than the official selected fit. All four tested
exploration weights choose different points for Functions 2, 6, 7 and 8; Function
5's weight choices agree after its separately declared spacing rule. The analysis
workflow prepared the inputs; the project owner later confirmed portal submission.
No Week 5 results have been recorded.

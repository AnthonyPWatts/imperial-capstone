# First-query proposals

Eight proposed first inputs, before any portal evaluation.

**Start by reducing ignorance, finish by exploiting what we have learned.**

| Function | Portal input |
| --- | --- |
| 1 | `0.468429-0.435440` |
| 2 | `0.774999-0.949999` |
| 3 | `0.335102-0.257540-0.459461` |
| 4 | `0.377878-0.477308-0.239218-0.435290` |
| 5 | `0.375048-0.953644-0.999999-0.999999` |
| 6 | `0.344331-0.260596-0.575965-0.745346-0.066311` |
| 7 | `0.057896-0.491672-0.247422-0.318118-0.420428-0.730970` |
| 8 | `0.000000-0.000000-0.000000-0.000000-0.999999-0.000000-0.000000-0.000000` |

## Interpretation

Function 1 retains its coverage proposal after an inconclusive raw-output KG check. Function 2 retains the finer-grid point from its existing study. Functions 3–8 use the conservative three-model KG comparison with noise, candidate-pool and thirteen-query checks. The larger-pool check revises Function 5's proposed point.

Prediction is strongest in Functions 4, 6 and 8, and weakest in Functions 1, 3 and 7. These differences reflect function behaviour and sample size as well as dimension. Numerical KG values are conditional expected reductions in decision shortfall, in original output units, not probabilities or guarantees. Six decimals express portal precision.

## Function 1

Retain average coverage. Raw-output KG is dominated by one negative extreme and lacks held-out support.

Estimated full-space coverage gain: 7.5379%. Best fitted-model held-out RMSE reduction versus the training-mean baseline: -0.9%.

## Function 2

Upper-right query from the finer local-model KG search; model uncertainty is more important than geometric coverage alone.

Estimated full-space coverage gain: 0.7110%. Best fitted-model held-out RMSE reduction versus the training-mean baseline: 24.8%.

## Function 3

Test whether the relatively good region around ranks 3 and 4 extends to a nearby combination; model support remains weak.

Estimated full-space coverage gain: 1.9120%. Best fitted-model held-out RMSE reduction versus the training-mean baseline: -3.6%.

## Function 4

Test a combination around the better observed settings. Held-out prediction strongly supports using fitted structure.

Estimated full-space coverage gain: 1.0766%. Best fitted-model held-out RMSE reduction versus the training-mean baseline: 82.7%.

## Function 5

Probe high x2, x3 and x4 near the large positive response. The larger-pool check changes the original proposal.

Estimated full-space coverage gain: 0.2792%. Best fitted-model held-out RMSE reduction versus the training-mean baseline: 28.2%.

## Function 6

Retain relatively high x4 and low x5 while testing a different combination of the other three inputs.

Estimated full-space coverage gain: 1.4850%. Best fitted-model held-out RMSE reduction versus the training-mean baseline: 59.3%.

## Function 7

Increase x4 by 0.1 at the best observation, keeping the other coordinates. A controlled local hypothesis test with weak model support.

Estimated full-space coverage gain: 0.0918%. Best fitted-model held-out RMSE reduction versus the training-mean baseline: 7.5%.

## Function 8

Test an extreme combination with low x1 and x3. Strong additive predictive structure supports modelling, while the conservative local-model choice is stable to noise.

Estimated full-space coverage gain: 0.0346%. Best fitted-model held-out RMSE reduction versus the training-mean baseline: 92.8%.

## Reproduction and reading copies

The individual notebooks retain the computational rationale. Run `stage-2-bbo/scripts/run_initial_analysis.py --function N` and the same command with `--decision-risk` using the repository's Python environment. Then run `stage-2-bbo/scripts/build_first_query_review.py` to rebuild the local pack.

The standalone index and portable ZIP are generated under `.runtime/first-query-review/` and `.runtime/capstone-first-query-pack.zip`. The proposal JSON records the base commit, verified report manifests, source hashes, settings and runtime versions. No submission or returned observation is recorded.

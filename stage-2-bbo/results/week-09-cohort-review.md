# Week 9 full-cohort review

The **23 November 2026** refresh adds **23** usable published peer returns.
The pool contains **456 observations**, comprising 239 official observations
through round 8 and 217 peer returns from 6 repositories.
Five query opportunities remain per function, including Week 9.
[Round 8 record](../submissions/round-08-results.json) ·
[Week 9 official review](week-09-review.md) ·
[Cohort surrogate study](week-09-cohort-surrogate-study.md)

## Provisional next inputs

| Function | My data only | Full cohort | Comparison |
| --- | --- | --- | --- |
| 1 | `0.615332-0.598353` | `0.759431-0.729743` | [Models](../notebooks/Function_1/Week_09/full-cohort/model_predictions.ipynb) |
| 2 | `0.000000-0.591796` | `0.515738-0.999999` | [Models](../notebooks/Function_2/Week_09/full-cohort/model_predictions.ipynb) |
| 3 | `0.152252-0.992806-0.410288` | `0.053258-0.998179-0.654576` | [Models](../notebooks/Function_3/Week_09/full-cohort/model_predictions.ipynb) |
| 4 | `0.391921-0.432079-0.420229-0.427165` | `0.386254-0.379280-0.428728-0.434290` | [Models](../notebooks/Function_4/Week_09/full-cohort/model_predictions.ipynb) |
| 5 | `0.976212-0.999999-0.999999-0.999999` | `0.999999-0.999999-0.999999-0.959224` | [Models](../notebooks/Function_5/Week_09/full-cohort/model_predictions.ipynb) |
| 6 | `0.424535-0.332747-0.976123-0.806605-0.002468` | `0.353915-0.525746-0.547870-0.672992-0.031418` | [Models](../notebooks/Function_6/Week_09/full-cohort/model_predictions.ipynb) |
| 7 | `0.000000-0.062967-0.325730-0.392674-0.349371-0.775387` | `0.141643-0.155568-0.443582-0.342683-0.360855-0.561684` | [Models](../notebooks/Function_7/Week_09/full-cohort/model_predictions.ipynb) |
| 8 | `0.000000-0.999999-0.000000-0.999999-0.999999-0.000000-0.000000-0.000000` | `0.000000-0.999999-0.000000-0.999999-0.000000-0.000000-0.000000-0.000000` | [Models](../notebooks/Function_8/Week_09/full-cohort/model_predictions.ipynb) |

The owner-only column reproduces the saved Week 9 proposals from the fits
trained on your observations. The full-cohort column refits the same declared
GP families on the pooled data and scores a fresh eligible candidate pool.
Both use upper confidence bound (UCB) with weight 1.96 and the weekly spacing
guard. The notebooks score both points with both fitted models and retain
the mean, latent SD and weight sensitivity. All sixteen points are distinct
from the known inputs in the current snapshot at portal precision.

Function 2's selected GP uses x1 alone in both scopes. The search breaks the
tie in x2 geometrically. Functions 5 and 8 have high measured incumbents and
the UCB picks can earn their rank through uncertainty about the score.

## Repository review and measured evidence

The course's profile-sharing thread contained **29 replies** and identified
**25 peer repositories**. GitHub showed four updated default branches and
18 unchanged ones. Two repositories were empty and one returned HTTP 404.
One self-referential link supplied no repository identifier. The refreshed
links and pinned commits are in the local intake audit. Three updated
repositories supplied the 23 additions. The other update changed modelling
files and left its published query histories at the previously reviewed blob
IDs.

The intake retained 194 previously accepted returns and matched the newly
published starter arrays to the official files by Git blob ID. It excluded
10 matching duplicates, 32 queries awaiting outputs and
13 records that failed the numeric input or output checks. Distinct
reported scores at the same portal input are retained and recorded in
the audit. Peer results are self-reported, with commit-pinned sources.
[Local intake audit](../data/cohort/2026-11-23/dataset-audit.json)

| Function | Official | Peer | New peer | Best official | Best peer | Data review |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 18 | 29 | 3 | 1.9973557 | 1.97963203 | [Exploration](../notebooks/Function_1/Week_09/full-cohort/observation_review.ipynb) |
| 2 | 18 | 27 | 3 | 0.723057546 | 0.736237833 | [Exploration](../notebooks/Function_2/Week_09/full-cohort/observation_review.ipynb) |
| 3 | 23 | 28 | 3 | -0.00495200082 | -0.000140306768 | [Exploration](../notebooks/Function_3/Week_09/full-cohort/observation_review.ipynb) |
| 4 | 38 | 27 | 3 | 0.666089599 | 0.712145931 | [Exploration](../notebooks/Function_4/Week_09/full-cohort/observation_review.ipynb) |
| 5 | 28 | 24 | 2 | 8662.405 | 6947.4309 | [Exploration](../notebooks/Function_5/Week_09/full-cohort/observation_review.ipynb) |
| 6 | 28 | 28 | 3 | -0.21395954 | -0.204982395 | [Exploration](../notebooks/Function_6/Week_09/full-cohort/observation_review.ipynb) |
| 7 | 38 | 28 | 3 | 2.73937512 | 2.60786287 | [Exploration](../notebooks/Function_7/Week_09/full-cohort/observation_review.ipynb) |
| 8 | 48 | 26 | 3 | 9.93677342 | 9.9465876 | [Exploration](../notebooks/Function_8/Week_09/full-cohort/observation_review.ipynb) |

Charts show the pooled score distribution and strongest input profiles. The
progress line follows the official round sequence.

## Held-out comparisons

Each GP family is compared on the same official leave-one-out targets. In
the full-cohort fit, a fold also withholds any peer measurement recorded at
the same six-decimal input. Function 1 uses MAE on signed-log targets.
Functions 2–8 use RMSE in the original score units.

| Function | Criterion | My data only | Full cohort | Matching peer rows withheld across folds | Lower error |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | LOO MAE | 127.03292 | 92.435946 | 1 | Full cohort |
| 2 | LOO RMSE | 0.17471391 | 0.12662465 | 2 | Full cohort |
| 3 | LOO RMSE | 0.070143927 | 0.019450225 | 1 | Full cohort |
| 4 | LOO RMSE | 1.5269439 | 1.3055844 | 1 | Full cohort |
| 5 | LOO RMSE | 529.24327 | 348.30559 | 1 | Full cohort |
| 6 | LOO RMSE | 0.32961147 | 0.20047093 | 1 | Full cohort |
| 7 | LOO RMSE | 0.28591695 | 0.39276528 | 1 | My data only |
| 8 | LOO RMSE | 0.29634697 | 0.28914023 | 1 | Full cohort |

The full-cohort fit had lower held-out error for seven functions. The largest
proportional drop was for Function 3, from 0.07014 to 0.01945. Function 7
favoured the owner-only fit, with RMSE 0.28592 versus 0.39277. These estimates
compare fits on the available official targets and inform the model choice.

## Verification

The Capstone kernel executed **16 notebooks** and **104 code cells**.
Readback checked **32 saved charts**, the dataset hashes, exact
six-decimal query strings and both held-out error calculations. The Stage 2
test suite passed **150 tests**.
[Reproduction instructions](../docs/reproducing-the-analyses.md#week-9-full-cohort-review)

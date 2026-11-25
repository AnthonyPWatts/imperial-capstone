# Week 9 official-only alternatives and query points

Seven Week 9 model notebooks refit alternatives supported by the official-only
validation. The new fits and model selection use the student's official
observations through round 8. The original owner GP and the full-cohort GP
coordinates come from the saved Week 9 comparison. [Full-cohort review](week-09-cohort-review.md)

The Matérn GP ranks candidates by the weekly upper confidence bound, the
predicted mean plus 1.96 times the latent standard deviation. Distance-weighted
nearest neighbours, the blend, SVR and kernel ridge rank candidates by
predicted mean. Each alternative uses the weekly broad and near-incumbent
Sobol search with official inputs as the reference for duplicate exclusion
and spacing. The Function 5 minimum distance is 0.02.

## Exact provisional inputs

| Function and alternative | My data only alternative | Original My data only GP | Full-cohort GP |
| --- | --- | --- | --- |
| [1](../notebooks/Function_1/Week_09/official/model_predictions.ipynb#my-data-only-alternatives) · Distance-weighted 5-NN | `0.627434-0.628714` | `0.615332-0.598353` | `0.759431-0.729743` |
| [2](../notebooks/Function_2/Week_09/official/model_predictions.ipynb#my-data-only-alternatives) · Matérn 3/2 GP | `0.920677-0.470459` | `0.000000-0.591796` | `0.515738-0.999999` |
| [3](../notebooks/Function_3/Week_09/official/model_predictions.ipynb) | — | `0.152252-0.992806-0.410288` | `0.053258-0.998179-0.654576` |
| [4](../notebooks/Function_4/Week_09/official/model_predictions.ipynb#my-data-only-alternatives) · Matérn 3/2 GP | `0.427496-0.489395-0.426743-0.422611` | `0.391921-0.432079-0.420229-0.427165` | `0.386254-0.379280-0.428728-0.434290` |
| [5](../notebooks/Function_5/Week_09/model_predictions.ipynb#my-data-only-alternatives) · Tuned GP blend | `0.976212-0.999999-0.999999-0.999999` | `0.976212-0.999999-0.999999-0.999999` | `0.999999-0.999999-0.999999-0.959224` |
| [6](../notebooks/Function_6/Week_09/model_predictions.ipynb#my-data-only-alternatives) · SVR (C = 3) | `0.445816-0.323600-0.551644-0.755918-0.072632` | `0.424535-0.332747-0.976123-0.806605-0.002468` | `0.353915-0.525746-0.547870-0.672992-0.031418` |
| 6 · Matérn 3/2 GP | `0.443519-0.397098-0.735830-0.825106-0.000000` | `0.424535-0.332747-0.976123-0.806605-0.002468` | `0.353915-0.525746-0.547870-0.672992-0.031418` |
| [7](../notebooks/Function_7/Week_09/model_predictions.ipynb#my-data-only-alternatives) · RBF kernel ridge | `0.000000-0.145572-0.392685-0.242554-0.346193-0.742781` | `0.000000-0.062967-0.325730-0.392674-0.349371-0.775387` | `0.141643-0.155568-0.443582-0.342683-0.360855-0.561684` |
| [8](../notebooks/Function_8/Week_09/model_predictions.ipynb#my-data-only-alternatives) · Matérn 3/2 GP | `0.000000-0.365002-0.020186-0.150000-0.796758-0.436941-0.101883-0.630634` | `0.000000-0.999999-0.000000-0.999999-0.999999-0.000000-0.000000-0.000000` | `0.000000-0.999999-0.000000-0.999999-0.000000-0.000000-0.000000-0.000000` |

## What changed

- **Function 1.** The 5-NN point is about 0.000538 units from the nearest
  official observation. It is a refinement of the region that produced the
  two highest measured returns. Both saved GP proposals go farther afield.
- **Function 2.** The modelled coordinate moves from `x1 = 0` to
  `x1 = 0.920677` under Matérn. Both kernels use x1 alone. The x2 setting
  follows the geometric tie-break over official observations.
- **Function 4.** The Matérn point moves mainly in x1 and x2. Its predicted
  mean is lower than Matérn's mean at the original point. The larger latent
  standard deviation accounts for the higher UCB.
- **Function 5.** The blend and original owner GP pick the same input.
  The final blend uses 25% target-warped GP and 75% three-seed tanh-8
  mean. Its x1 is reduced from the upper boundary. The full-cohort GP
  instead reduces x4.
- **Function 6.** The SVR point moves x3 to 0.551644, close to the full-cohort
  value of 0.547870. The separate Matérn candidate sets x5 to zero, and
  its x3 value sits between those of the two saved GPs.
- **Function 7.** Kernel ridge retains the original x1 = 0 face. Its
  improvement comes from the full set of observations. On the eight later
  returns, the original GP has lower error.
- **Function 8.** Matérn moves away from the two GP corners towards observed
  inputs. Its nearest official observation is 0.399 units away, compared
  with 1.114 for the original owner GP and 1.128 for the full-cohort GP.

## Official-only validation

Each cell below compares the original GP family with the named alternative
on the same two five-fold repeats. For Function 5, four inner grouped folds
choose the blend components and GP weight. The final refit uses all official
rows and the declared inner split seed 42. Later-return error uses the eight
official queries returned after the initial design. Best picks count how
often the highest predicted mean identified the best measured point in a
held-out shortlist. The last two columns give Euclidean distances between
the proposed points in the unit input box.

| Function and alternative | Error | Mean outer error | Later-return error | Best picks | Distance from original GP | Distance from full-cohort GP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 · Distance-weighted 5-NN | MAE | 127.02 → 108.58 | 103.77 → 106.78 | 6/10 → 8/10 | 0.033 | 0.166 |
| 2 · Matérn 3/2 GP | RMSE | 0.1606 → 0.1416 | 0.1849 → 0.1231 | 9/10 → 9/10 | 0.929 | 0.667 |
| 4 · Matérn 3/2 GP | RMSE | 2.394 → 1.707 | 0.7493 → 0.5653 | 7/10 → 8/10 | 0.068 | 0.118 |
| 5 · Tuned GP blend | RMSE | 588.60 → 382.31 | 844.43 → 578.37 | 9/10 → 9/10 | 0.000 | 0.047 |
| 6 · SVR (C = 3) | RMSE | 0.3341 → 0.2386 | 0.2672 → 0.1232 | 3/10 → 5/10 | 0.434 | 0.241 |
| 6 · Matérn 3/2 GP | RMSE | 0.3341 → 0.2755 | 0.2672 → 0.2292 | 3/10 → 4/10 | 0.250 | 0.290 |
| 7 · RBF kernel ridge | RMSE | 0.3233 → 0.2324 | 0.1149 → 0.1543 | 8/10 → 7/10 | 0.187 | 0.256 |
| 8 · Matérn 3/2 GP | RMSE | 0.3437 → 0.3119 | 0.3697 → 0.2724 | 8/10 → 9/10 | 1.329 | 1.536 |

Function 1 errors use signed-log height. The other errors use original score
units. The shortlists contain measured inputs, so these checks describe
retrospective ranking and prediction. The input strings above are provisional.

## Reproduction

The seven executed model notebooks save their official validation, fit
choices and scores at the archived points in `alternative-query-review.json`.
The ordinary Week 9 model runner regenerates the results. [Execution guide](../docs/reproducing-the-analyses.md#week-9-reviews)

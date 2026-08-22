# Class-membership probabilities

## Conclusion

The accepted model already provides the class memberships intended by this
idea. Every row receives three non-negative probabilities summing to one:

$$
P(\text{functional}\mid x),\quad
P(\text{repair}\mid x),\quad
P(\text{non-functional}\mid x).
$$

The submitted label is merely the class with maximum probability. Retaining the
full vector exposes ambiguity that the hard label discards.

The accepted 55% XGBoost / 45% Random Forest vote was refitted on all 59,400
labelled rows and used to create a validated **14,850-row competition membership
table**. This is an analytical export, not a DrivenData submission. It is stored
locally under the ignored runtime directory at
`.runtime/class-membership-analysis/accepted-competition-memberships.csv`.

The development out-of-fold evidence is reasonably encouraging: 81.625%
accuracy, 0.46851 log loss, 0.26669 multiclass Brier score and 0.708% top-label
expected calibration error. These remain model estimates, not guarantees of
individual pump condition.

## Course-aligned lifecycle

| Step | Application in this experiment |
| --- | --- |
| 1. Define the goal and scope | Retain and interpret the accepted classifier's complete three-class probability vector instead of immediately reducing it to one label. |
| 2. Gather the data | Reuse accepted out-of-fold development probabilities, then refit the accepted ensemble on all labelled rows and score the unlabelled competition rows. |
| 3. Explore the data | Compare membership shares, calibration, confidence, entropy, winning margins, runner-up classes and ambiguous cohorts. |
| 4. Clean and preprocess the data | Reuse the accepted fold-fitted preprocessing for development evidence and the same preprocessing fitted on all labelled rows for competition inference. |
| 5. Select and engineer features | Keep the accepted feature policy unchanged. |
| 6. Define the machine-learning task | Supervised three-class probabilistic classification; the hard label remains the maximum-probability decision. |
| 7. Partition the data | Use aligned out-of-fold probabilities for unbiased development diagnostics; competition rows remain unlabelled inference data. |
| 8. Select and train candidate methods | No new selection round: refit the already accepted XGBoost and Random Forest components at their frozen 55:45 weights. |
| 9. Evaluate and interpret the results | Evaluate probability quality only where labels exist; compare unlabelled competition membership distributions without claiming competition accuracy. |
| 10. Deploy and iterate | Export the validated membership table. Move the next bounded modelling loop to training-only outlier filtering. |

## Exported row contract

Each competition row contains:

- `id`;
- membership for functional, repair and non-functional, summing to one;
- maximum-probability `predicted_class` and its `confidence`;
- `runner_up_class` and its membership;
- `winning_margin`, the difference between the first and second memberships;
- `normalised_entropy`, from zero for concentrated membership to one for an
  even three-way distribution;
- flags for no majority membership, a margin below ten points and runner-up
  membership of at least 25%.

For example, a vector `(0.58, 0.05, 0.37)` is still predicted functional, but
the 37% non-functional membership is retained rather than discarded. This is
the practical distinction between a probability membership table and the hard
competition CSV.

## Are the probabilities usable as memberships?

The accepted out-of-fold probabilities provide a leakage-safe check because
every development row was predicted by models that did not train on it.

| Probability metric | Development result |
| --- | ---: |
| Accuracy after maximum-probability decision | 81.625% |
| Log loss | 0.46851 |
| Multiclass Brier score | 0.26669 |
| Top-label expected calibration error | 0.708% |
| Mean winning confidence | 80.925% |
| Mean winning margin | 65.147% |
| XGBoost/Random-Forest hard disagreement | 9.588% |

Average class membership closely reproduces observed prevalence:

| Class | Observed development share | Mean OOF membership | Difference | One-v-rest ECE |
| --- | ---: | ---: | ---: | ---: |
| Functional | 54.308% | 54.474% | +0.166 pp | 1.269% |
| Functional needs repair | 7.269% | 7.142% | -0.126 pp | 1.133% |
| Non-functional | 38.424% | 38.384% | -0.040 pp | 1.369% |

This aggregate agreement is useful but does not mean every 70% value is exact.
Calibration is estimated from finite, repeatedly consulted folds, and classwise
reliability varies across probability bands.

## What the hard labels hide

The repair class makes the distinction clearest.

| Repair measure | Development | Competition |
| --- | ---: | ---: |
| Mean repair membership | 7.142% | 7.178% |
| Hard repair prediction share | 4.158% | 3.953% |
| Rows with repair membership at least 25% | 8.611% | 8.909% |
| Rows with repair membership at least 50% | 3.239% | 2.997% |

The competition export therefore does not say only 3.95% of pumps have any
repair membership. It says repair is the largest membership for 587 rows, while
1,323 rows have at least 25% repair membership. Mean repair mass remains close
to the labelled prevalence even though maximum-probability decisions suppress
the difficult middle class.

On actual development repair rows, mean repair membership is only 33.766%.
That is consistent with the known difficulty: many true repair cases share
stronger observable evidence with functional or non-functional pumps.

## Ambiguity cohorts

The thresholds are descriptive and fixed before inspecting competition rows:

- **no majority:** winning membership below 50%;
- **close membership:** winning margin below ten percentage points;
- **substantial second membership:** runner-up membership at least 25%.

| Cohort | Development rows | Development share | Development accuracy | Competition rows | Competition share |
| --- | ---: | ---: | ---: | ---: | ---: |
| No majority | 2,155 | 4.535% | 46.821% | 692 | 4.660% |
| Close membership | 2,810 | 5.913% | 50.107% | 875 | 5.892% |
| Substantial second membership | 12,775 | 26.883% | 59.734% | 4,098 | 27.596% |

These cohorts behave as expected: the full development accuracy is 81.625%,
but close decisions are correct only about half the time. They are natural
review or triage candidates; changing their labels without a separately tested
decision rule would not be justified.

## Development-to-competition comparison

The unlabelled competition membership distribution is strikingly close to the
out-of-fold development distribution:

| Measure | Development | Competition |
| --- | ---: | ---: |
| Mean confidence | 80.925% | 80.925% |
| Mean winning margin | 65.147% | 65.174% |
| Mean normalised entropy | 42.325% | 42.108% |
| Close-decision share | 5.913% | 5.892% |
| No-majority share | 4.535% | 4.660% |
| Mean functional membership | 54.474% | 54.299% |
| Mean repair membership | 7.142% | 7.178% |
| Mean non-functional membership | 38.384% | 38.522% |

This is evidence against a large aggregate membership shift. It cannot detect
all covariate or conditional shifts, and competition accuracy remains unknown.

## Decision and next loop

Use the accepted probability vector as the class-membership representation. No
new fuzzy-label training or model family is required. Keep maximum probability
for the competition submission unless an alternative decision policy has its
own objective and fold-safe evaluation.

The next useful modelling change should be **outlier filtering**, not another
probability threshold, blend weight or global membership transformation. Start
with a bounded, training-only screen that distinguishes:

1. physically impossible or sentinel numeric states;
2. multivariate observations far outside the ordinary training support;
3. geographically or categorically isolated rows;
4. label-conflict candidates whose near neighbours have inconsistent outcomes.

Any filter must be learned inside each training fold, must never remove
validation rows from scoring, and must report both coverage and class-specific
removal rates. This probability export can help describe uncertain rows, but
model uncertainty must not itself be treated as proof that a row is a data
outlier.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_class_membership_analysis.py --force
```

Omit `--force` to reuse the ignored full-data component probabilities. The
stable diagnostics and export validation live in `src/probability_diagnostics.py`.

# Expanded ensemble voter screen

## Conclusion

Adding more voters does not improve the accepted two-model ensemble. The
closest bounded candidate adds a 5% MLP contribution and reaches **81.601%**
mean development-fold accuracy, 0.023 percentage points below the accepted
81.625%. CatBoost and LightGBM additions also reduce accuracy.

All three additions slightly improve log loss and Brier score, but the
competition selects hard labels by maximum probability and scores accuracy.
That calibration improvement is not an accuracy win.

No candidate passed the existing promotion gate. All results use persisted
out-of-fold probabilities; no model was refitted, the local test remained
closed and no competition prediction was generated.

## Course-aligned lifecycle

| Step | Application in this experiment |
| --- | --- |
| 1. Define the goal and scope | Test whether materially different third voters improve the accepted soft vote's frozen-fold accuracy. |
| 2. Gather the data | Reuse aligned OOF probabilities from models already trained on the five development folds. |
| 3. Explore the data | Compare standalone strength, hard-label disagreement, unique correct decisions and probability correlation. |
| 4. Clean and preprocess the data | Not applicable to this no-refit screen; every cached component was previously fitted with its own fold-safe preprocessing. |
| 5. Select and engineer features | Keep each voter's established representation: native categorical CatBoost, one-hot LightGBM and scaled one-hot MLP. |
| 6. Define the machine-learning task | Three-class classification under nominal accuracy. |
| 7. Partition the data | Reuse the untouched local test and five frozen development folds. |
| 8. Select and train candidate methods | No training is needed; combine three predeclared voters at fixed low weights while preserving 55:45 inside the accepted remainder. |
| 9. Evaluate and interpret the results | Apply the existing gate and compare accuracy, recall, disagreement, log loss and Brier score. |
| 10. Deploy and iterate | Not applicable: no candidate passed the gate. Retain two voters and stop this loop. |

## Candidate choice

The screen does not search arbitrary subsets or tune weights against the same
five folds. It adds one representative from each distinct hypothesis:

- **CatBoost d8 at 10%**: a native-categorical learner with a representation
  unlike the accepted one-hot models;
- **bagged LightGBM with 63 leaves at 15%**: the strongest prior third-voter
  complement from the accelerated model-family screen;
- **two-layer MLP at 5%**: the highest hard-label diversity among plausible
  cached voters, but weak enough to require a small contribution.

The remaining probability mass always preserves the accepted 55:45
XGBoost-to-Random-Forest ratio. This produces final weights of 49.5/40.5/10,
46.75/38.25/15 and 52.25/42.75/5 respectively.

This is an exploratory reuse of heavily used development folds. The repository
already contains 609 accelerated-model combinations, an earlier equal three-
and four-model round, a nested stack and 84 ANN blends. The bounded screen is a
useful stop decision, not independent confirmation for another submission.

## Is each voter actually complementary?

| Added voter | Standalone accuracy | Disagreement with accepted | Voter only correct | Accepted only correct | Probability correlation |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost d8 | 80.284% | 7.058% | 2.557% | 3.897% | 0.9762 |
| Bagged LightGBM 63 | 80.749% | 5.612% | 2.111% | 2.986% | 0.9845 |
| MLP 128–64 | 78.590% | 10.871% | 3.441% | 6.475% | 0.9432 |

LightGBM is reasonably strong but mostly repeats the accepted probability
signal. CatBoost changes the categorical treatment, but the accepted ensemble
still corrects more of its mistakes than the reverse. The MLP supplies the
most diversity, yet its errors are too numerous: the accepted model is uniquely
correct almost twice as often.

Extra Trees was not promoted to the final bounded screen. Its 78.72% standalone
accuracy and high disagreement initially look useful, but its OOF log loss is
about 1.32; uncalibrated soft voting would allow poor probabilities to move
otherwise correct decisions.

## Results

The accepted baseline has 81.6246% accuracy, 34.8588% repair recall, log loss
0.46851 and multiclass Brier score 0.26669.

| Added voter and final weights | Accuracy | Change | Fold wins | Worst fold | Repair-recall change | Log-loss change | Brier change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% MLP; 52.25% XGB / 42.75% RF | **81.6014%** | -0.0231 pp | 2/5 | -0.1368 pp | -0.3473 pp | -0.00063 | -0.00024 |
| 15% LightGBM; 46.75% XGB / 38.25% RF | 81.5783% | -0.0463 pp | 1/5 | -0.1684 pp | -0.8396 pp | -0.00108 | -0.00036 |
| 10% CatBoost; 49.5% XGB / 40.5% RF | 81.5446% | -0.0800 pp | 1/5 | -0.1684 pp | -0.5501 pp | -0.00088 | -0.00035 |

Every candidate misses both the required +0.10-point accuracy improvement and
the three-fold-win requirement. The negative log-loss and Brier changes mean
that the added voters slightly improve the probability distribution on
average, even as enough argmax decisions cross the wrong boundary to reduce
accuracy.

If calibrated risk estimates or expected operational cost replace competition
accuracy as the objective, these probability improvements could justify a new,
separately specified experiment. They do not justify changing the current
competition model.

## Decision

Retain the 55% child-weight-1 depth-8 XGBoost plus 45% Random Forest vote. More
voters are only useful when they contribute sufficiently strong, differently
wrong information. Voter count itself is not a source of performance.

Stop expanding this ensemble with the current cached families. A genuinely new
data source, representation or model family would be a better reason to reopen
the question than another weight grid over correlated tree learners.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_expanded_ensemble_screen.py
```

The ignored runtime directory contains candidate and component CSV summaries
plus serialised evaluations. Stable logic and the fixed recipes are in
`src/expanded_ensemble_evaluation.py`.

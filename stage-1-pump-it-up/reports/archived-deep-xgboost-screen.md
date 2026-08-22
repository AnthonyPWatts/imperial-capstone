# Archived deep XGBoost screen

## Decision

Retain the prepared archive synthesis and do not promote the strongest single
seed. The official-archive depth-17 XGBoost is a materially stronger component:
with top-50 identity indicators it reaches **81.4710–81.4731%**, compared with
81.0880% for top-50 depth-8 XGBoost and 81.0143% for accepted XGBoost.

Replacing the archive synthesis's main 33% XGBoost component with the first
predeclared deep seed reaches **81.8771%**, a promising 0.0842-percentage-point
gain with three fold wins, a 0.0210-point worst-fold loss and 0.6945 points
higher repair recall. It still misses the 0.10-point promotion gate.

A bounded equal average of three predeclared deep seeds lowers the archive
result to **81.8582%**. The original solution's variance-reduction mechanism
therefore does not supply the missing evidence. No local-test or competition
prediction is generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test the final deep stochastic learner paired with the positive top-common representation, without reopening a broad XGBoost grid. |
| 2. Gather the data | Reuse supplied predictors, frozen development rows, top-50 identity preprocessing and cached archive components. |
| 3. Explore the data | Compare the archived final specification with the repository's existing depth-6 to depth-11 candidate range. |
| 4. Clean and preprocess the data | Learn all 300 common-identity indicators inside each model-fitting fold and retain accepted preprocessing. |
| 5. Select and engineer features | Reuse the fixed top-50 representation; add no further fields or transforms. |
| 6. Define the machine-learning task | Retain unchanged three-class probabilistic classification and accuracy. |
| 7. Partition the data | Reuse the five frozen folds; fit each fixed 600-tree model on four folds and score the fifth. |
| 8. Select and train candidate methods | Fit depth 17, learning rate 0.02, child weight 1, 80% row sampling and 50% column sampling for exactly 600 trees. |
| 9. Evaluate and interpret the results | Test one direct archive substitution, one accepted/deep bag and, after meaningful signal, one equal three-seed deep bag. |
| 10. Deploy and iterate | Stop before local or competition use because neither the single seed nor the predeclared variance reduction passes the gate. |

## External motivation and experiment boundary

The archived 0.8264 entry's
[final training script](https://raw.githubusercontent.com/drivendataorg/pump-it-up/master/benedekrozemberczki/5_xgbooster_predictions.R)
uses depth 17, learning rate 0.02, child weight 1, 80% row sampling, 50%
column sampling and 600 trees, then generates 30 stochastic models. Its
[aggregation script](https://github.com/drivendataorg/pump-it-up/blob/master/benedekrozemberczki/6_prediction_aggregation.R)
averages their class probabilities.

This adaptation preserves the model settings but uses current XGBoost defaults
explicitly for zero gamma, unit L2 regularisation and zero L1 regularisation.
One deterministic seed was evaluated first. Its 0.0842-point archive gain was
large enough to trigger a bounded variance check, but not local confirmation.
Exactly two more seeds were then fitted and all three were averaged equally.
No seed was selected retrospectively.

## Results

| Candidate | Accuracy | Change vs archive | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing archive synthesis | 81.7929% | — | — | — | — |
| Accepted XGBoost | 81.0143% | — | — | — | — |
| Top-50 depth-8 XGBoost | 81.0880% | — | — | — | — |
| Deep seed 20260822 | 81.4710% | — | — | — | — |
| Deep seed 20260823 | 81.4646% | — | — | — | — |
| Deep seed 20260824 | **81.4731%** | — | — | — | — |
| Three-seed deep bag | 81.4646% | — | — | — | — |
| Equal accepted/deep archive bag | 81.8266% | +0.0337 pp | 3/5 | -0.0316 pp | +0.4631 pp |
| Three-seed deep archive substitution | 81.8582% | +0.0652 pp | 3/5 | -0.0947 pp | +0.4919 pp |
| Single-seed deep archive substitution | **81.8771%** | **+0.0842 pp** | 3/5 | -0.0210 pp | +0.6945 pp |

The deep model disagrees with accepted XGBoost on 4.89% of development rows.
It is uniquely correct on 2.4684%, versus 2.0118% for accepted XGBoost, so the
component improvement is substantive. Different deep seeds disagree on roughly
1.45–1.50% of rows, but averaging them does not improve accuracy or archive
placement. This suggests the first seed's extra archive corrections include
favourable variance, despite all three component means being tightly grouped.

The strongest result is 40 net rows above the archive incumbent and only eight
rows short of the mean-accuracy gate. That is precisely where discipline is
most important: selecting that seed, adding more seeds until an average crosses
the line, tuning depth/rounds or opening local test would adapt to a heavily
reused validation set. Retain the architecture as a near-miss for genuinely new
confirmation data, but do not create a third submission from it now.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_archived_deep_xgboost_screen.py
```

Runtime evidence remains ignored under `.runtime/archived-deep-xgboost-screen/`.

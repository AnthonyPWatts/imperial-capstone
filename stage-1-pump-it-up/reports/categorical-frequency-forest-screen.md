# All-categorical occurrence-frequency forest screen

## Decision

Retain the promoted 44:36:20 complete-identity vote. Replacing every
categorical source value with fold-fitted occurrence support creates a useful
but insufficiently strong Random Forest voter. An equal bag of the accepted and
frequency forests raises mean accuracy from 81.7403% to **81.7656%**, a
**+0.0253 percentage-point** change with **3/5** fold wins. This is below the
predeclared +0.10-point promotion gate.

No frequency transform, rare threshold, forest parameter or blend weight is
tuned. No local-test or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test a materially different representation found in the official competition solution archive. |
| 2. Gather the data | Reuse the supplied source predictors, frozen development rows and cached promoted components. |
| 3. Explore the data | Compare the archived count representation with the narrower six-identity frequency experiment already completed. |
| 4. Clean and preprocess the data | Apply accepted numeric cleaning and normalise categorical case, blanks and whitespace. |
| 5. Select and engineer features | Replace all 29 categorical/date fields with training-partition occurrence counts; support below five and unseen values share -1. |
| 6. Define the machine-learning task | Unchanged three-class classification and accuracy metric. |
| 7. Partition the data | Learn every count only inside each frozen outer-training fold; keep local test closed. |
| 8. Select and train candidate methods | Fit one accepted-spec Random Forest, then test a direct replacement and one equal within-family representation bag. |
| 9. Evaluate and interpret the results | Compare accuracy, fold stability, repair recall and hard-prediction diversity with the promoted vote. |
| 10. Deploy and iterate | Reject promotion, retain the evidence as a diverse near-miss and continue the external audit. |

## External lead and adaptation

DrivenData's [official Pump It Up solution archive](https://github.com/drivendataorg/pump-it-up)
records a 0.8265 public score for the `madRid` entry. Its
[submitted R code](https://github.com/drivendataorg/pump-it-up/blob/master/madRid/PumpItUp_DataDriven_2nd_place_madRid_team.R)
replaces every factor with its occurrence count, merges counts below five and
fits a random forest. The original code calculates counts after concatenating
the labelled and competition feature frames. This experiment deliberately uses
only each outer-training partition so that outer-validation feature values do
not influence fitted preprocessing.

This differs from the earlier six-identity screen in two important ways: it
covers every categorical hierarchy plus the exact recording date, and it
removes raw one-hot categories rather than appending six counts to them.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Equal Random Forest representation bag | **81.7656%** | **+0.0253 pp** | 3/5 | -0.0842 pp | -0.6372 pp | No |
| Promoted complete-identity vote | 81.7403% | — | — | — | — | incumbent |
| Frequency-forest replacement vote | 81.7066% | -0.0337 pp | 2/5 | -0.2946 pp | -1.0134 pp | No |
| Frequency-only Random Forest | 80.6103% | -1.1301 pp | 0/5 | -1.4625 pp | +1.7947 pp | No |
| Accepted Random Forest | 80.5913% | -1.1490 pp | 0/5 | -1.2311 pp | +2.9236 pp | No |

The frequency forest is 0.0190 points stronger than the accepted forest and
their hard predictions disagree on 6.4646% of development rows. Each is alone
correct on about 2.95% of rows, so the diversity is balanced rather than one
model simply dominating. The 18% frequency-forest allocation captures only a
small net gain, however. That is evidence against a blend-weight search on the
heavily reused folds.

The representation produces 43 engineered numeric features and 48 transformed
columns in fold one after missing-value indicators. This is compact compared
with one-hot preprocessing, but compactness alone does not improve the primary
metric.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_categorical_frequency_forest_screen.py
```

Runtime evidence remains ignored under
`.runtime/categorical-frequency-forest-screen/`.

# Top-common identity XGBoost screen

## Decision

Retain the prepared archive synthesis. Adding 300 fold-fitted indicators for
the 50 most common values of each deferred identity strengthens accepted
XGBoost from **81.0143%** to **81.0880%**, but its direct archive substitution
weakens. An equal accepted/top-common XGBoost bag reaches **81.8140%**, only
0.0210 percentage points above the 81.7929% archive incumbent.

The ten-row gain wins three folds but loses its worst fold by 0.0631 points and
reduces repair recall by 0.2607 points. It does not meet the predeclared
0.10-point accuracy gate, so no local-test or competition evidence is used.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test one compact sparse-identity representation absent from the existing native, frequency, target-encoding, hashing and text experiments. |
| 2. Gather the data | Reuse the six supplied deferred identity fields and the frozen development partition. |
| 3. Explore the data | Audit the official solution archive and identify the top-common indicator policy used by the archived 0.8264 entry. |
| 4. Clean and preprocess the data | Preserve exact loaded strings and an explicit missing state; learn every common-value list inside the current training fold. |
| 5. Select and engineer features | Add exactly 50 indicators for each of `funder`, `installer`, `wpt_name`, `subvillage`, `ward` and `scheme_name`. |
| 6. Define the machine-learning task | Retain unchanged three-class probabilistic classification and accuracy. |
| 7. Partition the data | Reuse the five frozen outer folds, with preprocessing and early stopping confined to each outer-training partition. |
| 8. Select and train candidate methods | Fit one accepted-spec depth-8, child-1 XGBoost candidate; do not screen common-value counts. |
| 9. Evaluate and interpret the results | Compare standalone accuracy, one archive substitution and one equal within-XGBoost representation bag. |
| 10. Deploy and iterate | Stop before local or competition use because the ten-row gain misses the promotion threshold. |

## External motivation and adaptation

DrivenData's [official solution archive](https://github.com/drivendataorg/pump-it-up)
records the `benedekrozemberczki` entry at 0.8264. Its
[keyword extractor](https://github.com/drivendataorg/pump-it-up/blob/master/benedekrozemberczki/1_keyword_extractor.R)
ranks common strings, and its
[cleaning code](https://raw.githubusercontent.com/drivendataorg/pump-it-up/master/benedekrozemberczki/2_data_cleaner.R)
creates indicators for the first 50 values of the sparse identity fields.

The adaptation keeps the current accepted features and model settings. It adds
only that missing representation, learns the six lists from each model-fitting
partition rather than the complete labelled data, and maps unselected or
unseen values to all-zero identity indicators. Lexical ordering breaks equal-
count ties deterministically.

The resulting first-fold matrix has 601 columns, including 300 new indicators.
Those indicators cover 40.81% of the possible row-by-identity positions in the
outer-training data.

## Results

| Candidate | Accuracy | Change vs archive | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing archive synthesis | 81.7929% | — | — | — | — |
| Accepted XGBoost | 81.0143% | — | — | — | — |
| Top-50 identity XGBoost | 81.0880% | — | — | — | — |
| Top-50 XGBoost archive substitution | 81.7761% | -0.0168 pp | 3/5 | -0.2104 pp | -0.6082 pp |
| Equal accepted/top-50 XGBoost archive bag | **81.8140%** | **+0.0210 pp** | 3/5 | -0.0631 pp | -0.2607 pp |

Top-common XGBoost disagrees with accepted XGBoost on 3.77% of development
rows. Its right-only correct share is 1.7572%, slightly above the accepted
model's 1.6835%, explaining the standalone improvement. Direct substitution
disturbs the existing probability balance, while equal representation
averaging recovers ten net archive predictions.

This is useful confirmation that common exact identities contain generalisable
signal. It is not enough evidence to tune 25/50/100 cut-offs, normalisation or
blend weights after extensive reuse of these folds.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_top_common_identity_screen.py
```

Runtime evidence remains ignored under `.runtime/top-common-identity-screen/`.

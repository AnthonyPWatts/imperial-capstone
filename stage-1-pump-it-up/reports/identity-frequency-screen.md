# Deferred-identity occurrence-frequency screen

## Decision

Retain the native-identity CatBoost representation without adding occurrence
frequencies to the accepted tree components. Six fold-fitted
`log1p(training count)` fields lower the promoted vote from 81.7403% to
**81.5846%**, a **-0.1557 percentage-point** change with only **1/5** fold wins.
The worst fold loses 0.4104 points and repair recall loses 1.5637 points.

No count scaling, frequency subsets or blend weights are tuned. No local-test
or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether identity support and rarity add information beyond native CatBoost identities. |
| 2. Gather the data | Reuse the six deferred source fields: funder, installer, waterpoint name, subvillage, ward and scheme name. |
| 3. Explore the data | Build on the audit that identified these fields as high-cardinality identities. |
| 4. Clean and preprocess the data | Apply the existing identity normalisation and map unseen values to zero. |
| 5. Select and engineer features | Add one numeric `log1p` occurrence count per identity; exclude raw identities from one-hot encoding. |
| 6. Define the machine-learning task | Unchanged three-class classification. |
| 7. Partition the data | Learn every count mapping only from the current outer-training fold. |
| 8. Select and train candidate methods | Refit the accepted XGBoost and Random Forest specifications. |
| 9. Evaluate and interpret the results | Compare both components, their 55:45 vote and the 44:36:20 identity vote. |
| 10. Deploy and iterate | Reject the frequency policy and retain the promoted native-identity vote. |

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Promoted identity vote | **81.7403%** | — | — | — | — | incumbent |
| Frequency identity vote | 81.5846% | -0.1557 pp | 1/5 | -0.4104 pp | -1.5637 pp | No |
| Frequency 55:45 vote | 81.4478% | -0.2925 pp | 0/5 | -0.6629 pp | -0.7820 pp | No |
| Accepted XGBoost | 81.0143% | -0.7260 pp | 0/5 | -0.8733 pp | -2.4034 pp | No |
| Frequency Random Forest | 80.9386% | -0.8018 pp | 0/5 | -1.1364 pp | +1.5922 pp | No |
| Accepted Random Forest | 80.5913% | -1.1490 pp | 0/5 | -1.2311 pp | +2.9236 pp | No |
| Frequency XGBoost | 80.5471% | -1.1932 pp | 0/5 | -1.4205 pp | -3.7929 pp | No |

The six fields expand the engineered frame from 29 to 35 features and the
first fold's transformed matrix to 307 columns. Occurrence support helps Random
Forest relative to its accepted form, but substantially weakens XGBoost. The
native CatBoost identity model already handles these strings more effectively;
adding a lossy rarity summary to both global trees reduces ensemble accuracy.

## Leakage controls

- Count mappings are fitted separately inside each outer-training fold.
- Validation-only and unseen identities map to zero.
- Counts contain no target labels and raw identity values are not one-hot
  encoded.
- The representation, tree specifications and 44:36:20 blend were fixed before
  the run.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_identity_frequency_screen.py
```

Runtime evidence remains ignored under `.runtime/identity-frequency-screen/`.

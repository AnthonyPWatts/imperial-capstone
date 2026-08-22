# Target-free deferred-identity hash screen

## Decision

Retain accepted XGBoost without an identity hash block. A fixed 4,096-column
shared hash lowers XGBoost by 0.3430 percentage points and lowers the promoted
vote from 81.7403% to **81.5783%**, a **-0.1620-point** change with only **1/5**
fold win.

Hash dimensions, subsets and ensemble weights are not tuned. No local-test or
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test the remaining audited target-free sparse-name representation. |
| 2. Gather the data | Reuse funder, installer, waterpoint name, subvillage, ward and scheme name. |
| 3. Explore the data | Build on native-category, occurrence-frequency and character-text results. |
| 4. Clean and preprocess the data | Apply conservative normalisation with field-prefixed missing and blank states. |
| 5. Select and engineer features | Hash six tokens per row into one fixed 4,096-column non-negative sparse block. |
| 6. Define the machine-learning task | Unchanged three-class classification. |
| 7. Partition the data | Reuse the five frozen folds; the hash itself contains no learned statistics. |
| 8. Select and train candidate methods | Refit only accepted depth-8 child-weight-1 XGBoost. |
| 9. Evaluate and interpret the results | Substitute hashed XGBoost in the unchanged 44:36:20 identity vote. |
| 10. Deploy and iterate | Reject hashing and close sparse-name representations. |

## Representation and controls

Each row contributes one field-prefixed token for each identity, for example
`funder=government` and `wpt_name=pump a`. `FeatureHasher` uses a shared 4,096-
column address space with `alternate_sign=False`. Rows therefore contain no
more than six additional non-zero values. The representation uses neither
targets nor fit-partition counts, and unseen identities require no special
mapping.

The first outer-training fold expands from about 301 accepted transformed
columns to 4,397 columns.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Promoted identity vote | **81.7403%** | — | — | — | — | incumbent |
| Hashed-XGBoost identity vote | 81.5783% | -0.1620 pp | 1/5 | -0.5892 pp | -1.2163 pp | No |
| Accepted XGBoost | 81.0143% | -0.7260 pp | 0/5 | -0.8733 pp | -2.4034 pp | No |
| Hashed XGBoost | 80.6713% | -1.0690 pp | 0/5 | -1.5888 pp | -3.3585 pp | No |

Hashing avoids unseen-category failures but forces unrelated identities to
share buckets and offers no ordered category statistics. In this data, native
CatBoost identities remain the stronger way to use the sparse names.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_identity_hash_screen.py
```

Runtime evidence remains ignored under `.runtime/identity-hash-screen/`.

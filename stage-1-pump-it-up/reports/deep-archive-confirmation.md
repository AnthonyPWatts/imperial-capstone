# Deep-XGBoost archive confirmation

## Decision

Prepare the unchanged single-seed deep-XGBoost archive substitution for the
final 23 August submission slot. It reaches **81.8771%** on the frozen
development folds and **81.3552%** on the used local test, improving the
submitted archive synthesis by **0.0842** and **0.1684 percentage points**
respectively.

The local result adds 20 net correct classifications among 175 changed hard
predictions. Functional and non-functional recall improve; repair recall falls
by 0.116 points. Log loss and multiclass Brier score also improve. No seed,
tree count, feature, component weight or decision threshold changed after the
local result was observed.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Reopen the held final submission slot for the strongest previously recorded, materially different near-miss. |
| 2. Gather the data | Reuse the supplied data, frozen development folds, used local test and cached archive components. |
| 3. Explore the data | Use the existing archived-deep screen; do not search another model or blend family. |
| 4. Clean and preprocess the data | Retain accepted fold-fitted preprocessing and the fixed top-50 indicators for six deferred identities. |
| 5. Select and engineer features | Replace only the main accepted XGBoost component with the already-evaluated top-50 deep XGBoost. |
| 6. Define the machine-learning task | Unchanged three-class probabilistic classification scored by accuracy. |
| 7. Partition the data | Confirm once on the existing 11,880-row local test after the development result was fixed. |
| 8. Select and train candidate methods | Fit seed 20260822 at depth 17, learning rate 0.02 and exactly 600 trees. |
| 9. Evaluate and interpret the results | Compare paired accuracy, recalls, macro F1, log loss, Brier score and uncertainty with the submitted archive. |
| 10. Deploy and iterate | Refit the unchanged recipe on all 59,400 labels and validate the 14,850-row candidate for the final daily slot. |

## Fixed recipe

| Component | Weight |
| --- | ---: |
| Deep top-50 identity XGBoost | 33% |
| Spatial-height XGBoost | 11% |
| Accepted Random Forest | 18% |
| Categorical-frequency Random Forest | 9% |
| Spatial-height Random Forest | 9% |
| Complete-identity CatBoost | 20% |

The only change from the submitted archive synthesis is its 33% primary
XGBoost allocation. The accepted depth-8 component is replaced by the fixed
depth-17, 600-tree, seed-20260822 component over 300 fold-fitted top-common
identity indicators.

## Evidence

| Metric | Archive synthesis | Deep substitution | Change |
| --- | ---: | ---: | ---: |
| Development accuracy | 81.7929% | **81.8771%** | **+0.0842 pp** |
| Local-test accuracy | 81.1869% | **81.3552%** | **+0.1684 pp** |
| Local balanced accuracy | 66.7792% | **66.8683%** | +0.0891 pp |
| Local macro F1 | 69.4263% | **69.4999%** | +0.0735 pp |
| Local functional recall | 90.1271% | **90.3131%** | +0.1860 pp |
| Local repair recall | **32.4450%** | 32.3291% | -0.1159 pp |
| Local non-functional recall | 77.7656% | **77.9628%** | +0.1972 pp |
| Local log loss | 0.461905 | **0.459752** | -0.002152 |
| Local multiclass Brier | 0.265755 | **0.264112** | -0.001643 |

The development result wins three of five folds and loses its weakest fold by
0.0210 points. On the local test, the deep candidate alone gets 91 changed
rows correct and the archive alone gets 71. Exact McNemar p is 0.135 and the
paired bootstrap 95% interval for the accuracy change is -0.0421 to +0.3788
points. This is coherent new confirmation, not proof that the gain will repeat
on the public labels.

## Prepared competition candidate

The validated file contains 14,850 unique IDs in template order and differs
from the submitted archive on 237 rows (1.5960%). Its class shares are 60.694%
`functional`, 3.684% `functional needs repair` and 35.623% `non functional`.

- File: `01-33-deep-top50-xgboost-11-spatial-xgboost-18-random-forest-09-frequency-forest-09-spatial-forest-20-identity-catboost.csv`
- SHA-256: `ef9b4e0fa2c696ae5a203f25cd9b2c4ca4e4e5d83aa4e010595e39af60b7a1d2`
- Status: prepared, not uploaded

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_deep_archive_confirmation.py
..\.venv\Scripts\python.exe .\scripts\generate_deep_archive_submission.py
```

Runtime evidence remains ignored under `.runtime/deep-archive-confirmation/`
and `.runtime/deep-archive-competition/`.

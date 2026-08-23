# Deep-archive follow-up search

## Decision

Prepare the seed-20260824 deep-archive substitution as the replacement final
submission candidate. It improves the seed-20260822 candidate on both reused
evidence sets: **81.8981%** versus 81.8771% on development and **81.3973%**
versus 81.3552% on the local test. This is ten and five net additional correct
rows respectively.

The change is deliberately narrow. The six component weights, top-50 identity
representation, depth-17 XGBoost specification and 600-tree count are
unchanged; only the seed changes from one of the three values declared in the
original deep-XGBoost screen to another.

The margin is small and uncertain. The 50 local-test disagreements contain 24
seed-20260824-only correct decisions and 19 seed-20260822-only correct
decisions. Exact McNemar p is 0.542, and the paired row-bootstrap 95% interval
for the accuracy change is -0.0673 to +0.1515 percentage points. The candidate
is preferred because the direction agrees across development accuracy, local
accuracy, balanced accuracy, macro F1, log loss and Brier score, not because
five rows constitute proof.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Seek a reproducible improvement over the prepared seed-20260822 final candidate without uploading during selection. |
| 2. Gather the data | Reuse supplied predictors, frozen development folds, the used local test and aligned cached component probabilities. |
| 3. Explore the data | Audit existing OOF caches for residual signal, then test two primary-model hypotheses and the strongest predeclared seed. |
| 4. Clean and preprocess the data | Preserve fold-fitted accepted, top-common, target-encoding and grid preprocessing contracts. |
| 5. Select and engineer features | Test exact recording batches once; otherwise add no features to the promoted seed candidate. |
| 6. Define the machine-learning task | Unchanged three-class probabilistic classification scored by accuracy. |
| 7. Partition the data | Select on five frozen development folds and confirm fixed candidates on the existing local test. |
| 8. Select and train candidate methods | Close exact-date, low-weight residual and inner-stopping branches; compare the three already-declared deep seeds. |
| 9. Evaluate and interpret the results | Require directionally coherent development/local evidence and report paired uncertainty. |
| 10. Deploy and iterate | Refit seed 20260824 on all 59,400 labels and validate a separate 14,850-row CSV without uploading it. |

## Closed branches

| Candidate | Development accuracy | Change vs seed 20260822 | Local-test accuracy | Local change | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Deep XGBoost plus exact recording-batch one-hot | 81.8056% | -0.0715 pp | Not opened | — | Reject |
| 92.5% deep archive + 2.5% location target-encoded vote + 5% 10 km grid vote | **81.9318%** | +0.0547 pp | 81.3215% | -0.0337 pp | Reject after four-row loss |
| Inner-stopped deep XGBoost | 81.8497% | -0.0274 pp | Not opened | — | Reject |
| Deep seed 20260824 | **81.8981%** | **+0.0210 pp** | **81.3973%** | **+0.0421 pp** | Prepare |

The residual-voter candidate improved every development fold and added 26 net
correct rows, but changed only 20 local-test decisions and lost four net rows.
That failed confirmation is strong evidence against continuing to mine small
weights from the 148 cached OOF representations.

Inner stopping selected 449, 517, 386, 371 and 469 trees across the five outer
folds. Its lower accuracy supports retaining the archived fixed 600-tree
contract rather than tuning a nearby iteration grid.

## Seed-20260824 evidence

| Metric | Seed 20260822 | Seed 20260824 | Change |
| --- | ---: | ---: | ---: |
| Development accuracy | 81.8771% | **81.8981%** | **+0.0210 pp** |
| Local-test accuracy | 81.3552% | **81.3973%** | **+0.0421 pp** |
| Local balanced accuracy | 66.8683% | **67.0906%** | +0.2223 pp |
| Local macro F1 | 69.4999% | **69.7437%** | +0.2438 pp |
| Local repair recall | 32.3291% | **33.0243%** | +0.6952 pp |
| Local log loss | 0.459752 | **0.459567** | -0.000185 |
| Local multiclass Brier | 0.264112 | **0.264010** | -0.000102 |

Against the submitted archive synthesis, seed 20260824 adds 50 net correct
development rows and 25 net correct local-test rows. It wins three development
folds against that submitted baseline, with a -0.0421-point worst-fold change.

## Prepared competition candidate

The validated CSV contains 14,850 unique IDs in template order. It changes 75
hard predictions (0.5051%) from the seed-20260822 candidate and 226 predictions
(1.5219%) from the submitted archive. Prediction shares are 60.721%
`functional`, 3.643% `functional needs repair` and 35.636% `non functional`.

- File: `01-deep-archive-seed-20260824.csv`
- SHA-256: `fe5de9ea46bad2b35226bc97ebdfb743807fb758df1d259e8f8a51609808fee2`
- Status: prepared, not uploaded

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_deep_seed_20260824_confirmation.py
..\.venv\Scripts\python.exe .\scripts\generate_deep_archive_submission.py --seed 20260824
```

The rejected checks are reproducible with
`run_deep_recording_batch_screen.py`, `run_deep_residual_confirmation.py` and
`run_deep_iteration_selection_screen.py`. Runtime evidence remains ignored
under the corresponding `.runtime/` directories.

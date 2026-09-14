# Native-category follow-up, 6 September 2026

## Outcome

**No new submission.** Native-category LightGBM improved average accuracy on
both fold plans, but failed the locked worst-fold stability guard in
confirmation. The **0.8304** incumbent is retained, with all **three** daily
submission slots unused. Deeper spatial-grid CatBoost failed the first screen.

## Fixed experiment

The comparison starts from the **0.8304** public incumbent: the six-component
deep archive with its 0.20 CatBoost slot divided equally between complete
identities and identities plus four coordinate grids. The account was checked
on 6 September and showed rank **1**, 21 submissions and **3 of 3** daily slots
available. Rank is a dated observation, not a permanent claim.

Two model-level hypotheses were fixed before fitting:

- **Deeper spatial-grid CatBoost:** change depth 8 to depth 10, preserving the
  learning rate, regularisation, model seed and feature policy. Average it
  equally with the depth-8 grid model inside that model's existing 0.10 slot.
- **Native-category LightGBM:** use the existing bagged 63-leaf recipe with the
  complete identities and four coordinate grids, represented as native
  categoricals instead of the previous one-hot encoding. Fix `cat_smooth=20`,
  `cat_l2=10`, `min_data_per_group=100` and `max_cat_threshold=32`. Give it 0.10
  of the whole ensemble, scaling every incumbent weight by 0.90.

There is no weight, seed or hyperparameter sweep. Their one fixed combination
may be considered only if both individual changes pass. No row-level
postprocessing, external source data, external labels or test-label probing is
used.

The [live competition rules](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/rules/)
were checked on 6 September: external data remains prohibited and the daily
submission limit must not be circumvented. Supplied data and row-level cached
predictions remain local and ignored by Git.

The native representation follows LightGBM's documented categorical support
and regularisation controls ([official parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)).
Depth 10 is supported for CatBoost's GPU multiclass objective
([official training parameters](https://catboost.ai/docs/en/references/training-parameters/common)).

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Seek a reproducible improvement over the 0.8304 incumbent using two bounded model-level hypotheses. |
| 2. Gather the data | Use only the supplied 59,400 labelled rows during model selection; open competition predictors only after the final decision. |
| 3. Explore the data | Reuse the completed generalisation diagnosis and audit the previous experiments to avoid repeating one-hot LightGBM screens. |
| 4. Clean and preprocess the data | Preserve structural cleaning. Fit native category vocabularies on training rows only; map unseen categories to missing. |
| 5. Select and engineer features | Reuse complete identities and the four established spatial grids; make no new row-dependent feature selection. |
| 6. Define the machine-learning task | Preserve three-class classification by accuracy, with repair-class recall as a guard. |
| 7. Partition the data | Screen on the reused five-fold seed-20260905 plan. Confirm survivors with a fixed five-fold seed-20260906 plan and a newly fitted seven-model incumbent. |
| 8. Select and train candidate methods | Use inner training-only stopping to select iterations, then refit every outer-training row. Fit one recipe per hypothesis. |
| 9. Evaluate and interpret the results | Require positive overall accuracy change, at least three fold wins, worst-fold loss no worse than 0.10 percentage points, and repair-recall loss no worse than 1.00 point on both plans. |
| 10. Deploy and iterate | Generate only candidates passing both checks, lock all files before any public result, and retain the incumbent if the evidence fails. |

## Interpretation limits

The first fold plan was already used for model selection. The second plan
changes partition membership but uses the **same previously explored labelled
population**. It checks partition stability; it is not an untouched dataset or
an unbiased final performance estimate. The guard is exploratory, not a claim
of statistical significance. A new daily allowance is not a reason to use
every slot.

The first-screen protocol, source hashes, data hashes and self-digesting fold
caches are stored under
`.runtime/native-category-followup-20260906-screen/`. A separate confirmation
directory is created only for surviving candidates. Existing source modules
and September 5 caches are not changed by this experiment.

## Reproduction

Run from the capstone repository with its existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s stage-1-pump-it-up/tests -p test_native_category_followup.py -v
.\.venv\Scripts\python.exe -u stage-1-pump-it-up/scripts/run_native_category_followup_screen.py
# Only if the first screen admits a challenger:
.\.venv\Scripts\python.exe -u stage-1-pump-it-up/scripts/confirm_native_category_followup.py
.\.venv\Scripts\python.exe stage-1-pump-it-up/scripts/audit_native_category_confirmation.py
# Only if both fixed partition guards pass:
.\.venv\Scripts\python.exe stage-1-pump-it-up/scripts/generate_native_category_followup.py --generate
```

Cached reruns validate the protocol and probability content rather than
refitting. The confirmation runner refuses an empty or stale eligibility list.
The generator validates both evidence sets and reconstructs the incumbent CSV
before creating any new predictions. It does not upload files.

## Results

The first screen reconstructed incumbent OOF accuracy as **82.0522%**.

| Fixed candidate | OOF accuracy | Net correct rows | Fold wins | Worst-fold change | Repair-recall change | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Depth-10 grid half-slot bag | 82.0387% | -8 | 3/5 | -0.1010 pp | -0.0232 pp | Reject |
| Native LightGBM at 0.10 | 82.0690% | +10 | 3/5 | -0.0589 pp | +0.0695 pp | Confirm unchanged |
| Fixed combination | 82.0690% | +10 | 3/5 | -0.0337 pp | +0.0463 pp | Ineligible: individual CatBoost change failed |

The LightGBM blend changed 468 OOF labels. Ten net additional correct labels
are weak evidence, despite passing the exploratory guard. Only that unchanged
blend advances to the second partition. The failing CatBoost branch is not
retuned or rescued using its combination result.

The screen selected CatBoost iteration counts of 1,013, 1,065, 1,395, 1,349 and
1,391; LightGBM selected 244, 258, 277, 274 and 347. Selection happened inside
each outer-training fold.

### Second-partition confirmation

The seed-20260906 assignment has fingerprint
`7217ac86c4c85eef6e7d8dc47fc0acdc0bafc8934da6d53deac1f6ea32adbf1c`.
All seven incumbent components and the native LightGBM challenger were newly
fitted on these folds. The later forests were computed alongside earlier GPU
fits, using the same fold contracts and no duplicate fits.

| Metric | Incumbent | Native LightGBM at 0.10 | Difference |
| --- | ---: | ---: | ---: |
| OOF accuracy | 81.9428% | 82.0118% | +0.0690 pp / +41 rows |
| Fold wins | — | 4/5 | — |
| Repair recall | — | — | +0.0695 pp |
| Worst-fold accuracy change | — | — | **-0.1515 pp: fails the -0.1000 pp guard** |

The five paired fold changes were **+15, +19, +22, +3 and -18** correct rows,
each over 11,880 validation rows. There were 491 changed predictions overall:
244 previously wrong predictions became correct, 203 previously correct
predictions became wrong, and 44 changed between two incorrect classes.
The independent seven-weight reconstruction reproduced every fold delta and
validated all IDs and cache digests.

A descriptive paired normal-approximation 95% interval for the overall gain
is **-0.0007 to +0.1388 percentage points**. It includes zero and does not
account for adaptive selection or dependence from overlapping training folds;
it is not a significance claim. The LightGBM iteration selections were 281,
273, 278, 268 and 272, with median 273. No full-data refit was performed.

The average gain is worth retaining as evidence for the native-category
representation. It does **not** justify changing the guard, reducing the blend
weight after seeing the failing fold, or submitting the rejected combination.
There are no eligible submissions under the fixed protocol. Competition
predictors were never opened by this experiment, and no new competition CSV
was generated or uploaded.

## Verification and evidence

- Focused checks: **5/5 passed**.
- Full Stage 1 suite: **316/316 passed** using
  `.\.venv\Scripts\python.exe -m unittest discover -s stage-1-pump-it-up/tests -q`.
- First-screen cached replay: passed with the same protocol, probabilities,
  diagnostics and result file; no additional fits.
- Independent confirmation replay: passed all 40 fold caches, IDs, digests,
  explicit seven-model weights, aggregate accuracy and paired fold deltas.
- Generator negative-path check: deliberately refused with
  `No valid twice-gated submission candidates.` before opening competition
  covariates or creating an output directory. The successful generation/upload
  path was not exercised because no candidate qualified.
- The model module and four runners compiled. No dependency or existing
  model-source changes were needed. The pinned environment emitted non-failing
  pandas/joblib deprecation warnings and XGBoost's CPU-input/GPU-prediction
  fallback warning. Compatibility outside the pinned environment was not tested.

Key SHA-256 evidence:

| Artefact | SHA-256 |
| --- | --- |
| First-screen protocol | `de12d40635f27f77590ebfaeb8ac981877ec5c297d1df14b6e104dda7fe873e8` |
| First-screen result | `fe9bd217516a177a5e1ffa73b1489aeff4da9f6ddb33d1c235873825c69a1e86` |
| Confirmation protocol | `cef17a9220d095d9b4335d916ace6084c126705678f3ff43c1e61d0890697765` |
| Confirmation result | `c9a2bfc32835cf02347a42d174b21538eab9f97ffeb08cea005db6f3b9a8ad3c` |

The full machine-readable evidence and independent audit remain under
`.runtime/native-category-followup-20260906-confirmation/`.

---
status: candidate-submission-ready
branch: main
updated: 2026-08-21
---

# Stage 1 data preparation: decisions and next steps

## Working on

Upload and record the prepared Random Forest and histogram-boosting candidate.
Their equal-weight vote leads development-fold accuracy at 81.37% and records
80.82% on the one-time local test. Both components have been refitted on all
59,400 labelled original rows; the validated 14,850-row competition CSV is
ready. After recording its public score, extend the target audit into class
meaning and possible ordinal or hierarchical structure, then run grouped
geographic sensitivity.

The initial date treatment replaces `date_recorded` with
`days_since_recorded`, measured from the fixed 2015-02-02 competition-era
reference. It does not derive separate year, month or day fields.

The fixed column-removal implementation is in
[`src/data_preparation.py`](../src/data_preparation.py), with its declarative
schema and assumptions in
[`src/raw_feature_column_policy.json`](../src/raw_feature_column_policy.json).
The preserved prior overall audit at
[`10-prior-overall-data-audit.ipynb`](../notebooks/data-audit/00-overall/supporting-audits/10-prior-overall-data-audit.ipynb)
applies it independently to both raw feature frames and shows the 40 to 37
column change without altering the source DataFrames.

The notebook's reusable structure and identifier checks now live in
[`src/source_data_validation.py`](../src/source_data_validation.py). The notebook
keeps file loading, progress messages and one-off exploration; the module owns
only the reusable validations for raw feature schema, label-frame structure and
matching IDs. No stateful manager classes are needed for these operations.

## Course scope established on 14 August 2026

Stage 1, including the Pump It Up work, provides optional practice and portfolio
evidence. The course labels the Module 3 to 9 activities as self-study capstone
components, and the Module 6 introduction says that these activities do not count
towards programme completion. The overview also says elsewhere that Stage 1 runs
through Module 11, so the published boundaries conflict.

The required capstone begins in Module 12 with the shared black-box optimisation
challenge. Modules 12 to 24 require numeric queries and progress records. The
Module 25 submission asks for a GitHub repository containing runnable, commented
code, a data sheet, a model card, a non-technical write-up, a README linking to
large data rather than storing it, and relevant supporting material.

Later Stage 2 pages remain locked. The pages available at this date do not state
a minimum model score or formal quality threshold. They support a cautious
working interpretation: the course requires a complete, documented Stage 2
attempt; stronger results determine presentation or distinction opportunities.

Evidence:

- [Capstone project overview](https://classroom.emeritus.org/courses/18642/pages/capstone-project-overview?module_item_id=3236688)
- [Module 6 discussion](https://classroom.emeritus.org/courses/18642/discussion_topics/969793?module_item_id=3237005)
- [Course module list](https://classroom.emeritus.org/courses/18642/modules)

The Module 6 discussion post records the Stage 1 ambiguity and explains that
Pump It Up was selected as an interesting data set, rather than as a commitment
to enter the competition. Other learners reported similar uncertainty about
project choice.

## Evidence already collected

The audit area contains the current evidence and findings:

- [`00-overall-data-audit.md`](../notebooks/data-audit/00-overall/00-overall-data-audit.md)
  contains the findings, decisions, risks and full raw-predictor register.
- [`00-target-label-analysis.ipynb`](../notebooks/data-audit/00-target-label-analysis/00-target-label-analysis.ipynb)
  contains the focused initial target analysis.
- [`04-amount_tsh-deep-dive.ipynb`](../notebooks/data-audit/01-amount_tsh/04-amount_tsh-deep-dive.ipynb)
  preserves the earlier focused `amount_tsh` analysis alongside its canonical
  three-notebook audit.
- [`data-audit/README.md`](../notebooks/data-audit/README.md) indexes all 39 raw
  predictors, including the three proposed structural removals.

### Target label findings

`status_group` has one complete label for each of the 59,400 unique training IDs
and contains exactly the three documented classes:

| `status_group` | Rows | Share |
| --- | ---: | ---: |
| `functional` | 32,259 | 54.31% |
| `functional needs repair` | 4,317 | 7.27% |
| `non functional` | 22,824 | 38.42% |

The archived official problem page defines both `functional` classes as
operational, separated by whether repairs are needed; `non functional` means
not operational. This supplies direct semantic support for testing an
operational/not-operational classifier followed by a repair/no-repair classifier
inside the operational branch. Ordinal severity remains a separate hypothesis
rather than an assumption.

An always-`functional` non-model would achieve 54.31% accuracy. The majority
class is 7.47 times the size of `functional needs repair`. Preserve stratification
in the local test set and development folds, retain challenge accuracy as the
primary comparison measure and inspect the confusion matrix and per-class recall
during diagnosis. Do not apply oversampling or calculate class weights before
local partitioning.

The duplicate-column audit covered both feature files:

| Data set | Rows | Source columns |
| --- | ---: | ---: |
| Training values | 59,400 | 40 |
| Test values | 14,850 | 40 |

### Duplicate and near-duplicate findings

| Category | Finding in training data | Finding in test data | Decision |
| --- | --- | --- | --- |
| Duplicate column labels | None | None | No action |
| Same values, row for row | `quantity` equals `quantity_group` | Same result | Keep `quantity`; drop `quantity_group` |
| Same information under different labels | `payment` and `payment_type` form a one-to-one relabelling | Same result | Keep `payment_type`; drop `payment` |
| More than 99.5% one repeated value | `recorded_by` is 100% `GeoData Consultants Ltd` | Same result | Drop `recorded_by` |
| Other column pairs above 99.5% row agreement | None | None | No action |

The payment mapping was stable in both files:

| `payment` | `payment_type` |
| --- | --- |
| `pay annually` | `annually` |
| `pay monthly` | `monthly` |
| `pay per bucket` | `per bucket` |
| `pay when scheme fails` | `on failure` |
| `never pay` | `never pay` |
| `other` | `other` |
| `unknown` | `unknown` |

`extraction_type` and `extraction_type_group` had the highest remaining literal
agreement: 95.843% in training and 95.684% in test. This falls below the 99.5%
threshold. `num_private` contained zero for about 98.73% of training rows and
98.69% of test rows, so it also falls below the threshold.

Several column families encode deterministic hierarchies:

- extraction type, group and class;
- management and management group;
- water quality and quality group;
- source, source type and source class;
- waterpoint type and waterpoint group.

These hierarchies may help models at different levels of granularity. Keep them
until training-partition analysis or validated model comparisons justify a
removal.

## Settled structural policy

Apply the same fixed policy independently to any raw training or test feature
DataFrame:

| Source column | Treatment | Reason |
| --- | --- | --- |
| `quantity` | Keep | Preferred descriptive field |
| `quantity_group` | Drop | Row-for-row duplicate of `quantity` |
| `payment_type` | Keep | Preferred compact field |
| `payment` | Drop | One-to-one relabelling of `payment_type` |
| `recorded_by` | Drop | Constant in both supplied feature files |
| `id` | Preserve in the returned DataFrame | Required for later joins and competition submissions; exclude from predictors when the modelling workflow separates identifiers |

The column-removal function turns the 40 raw columns into 37 columns, including
`id`. It performs no identifier separation or modelling transformation. A later
workflow step will preserve `id` as metadata and pass the other 36 columns to
the model pipeline as candidate predictors.

## Pipeline boundary

The project will recreate canonical data from the untouched competition CSV
files on each run. Do not create or maintain cleaned CSV copies.

```mermaid
flowchart LR
    A["Immutable source CSVs"] --> B["Load raw feature DataFrame"]
    B --> C["Validate and remove three known redundant columns"]
    C --> D["Separate IDs and target"]
    D --> E["Create reproducible stratified split"]
    E --> F["Explore training partition"]
    F --> G["Fit imputation, encoding and model pipeline"]
    G --> H["Evaluate validation partition"]
```

Structural preparation may inspect schema and agreed equivalence rules. It must
not calculate means, modes, frequencies, target correlations or other learned
statistics. Imputation, encoding and model transforms must fit on the training
partition or current cross-validation fold.

Generic duplicate and constant-column discovery belongs in an audit report. It
must not drop new columns from test data or change the feature policy without a
reviewed decision.

## Required safeguards

The column-removal function must stop with a clear error if any of these
conditions fails:

1. Required raw feature columns are missing, unexpected columns appear or a
   column label is duplicated.
2. `quantity` and `quantity_group` differ, including their missing-value pattern.
3. A `payment` value does not map to the expected `payment_type` value.
4. `recorded_by` contains a value other than the agreed constant.

The function must preserve row count, row order, `id` and the order of all
retained columns. It must not mutate the caller-owned DataFrame.

Identifier integrity, target alignment and matching training/test predictor
schemas remain necessary, but they belong to the later workflow that separates
metadata and constructs the modelling inputs. They are not responsibilities of
the fixed column-removal function.

## Fit with the course ten-step plan

The project uses the canonical course lifecycle while creating the split before
any learned preprocessing to prevent leakage.

| Course step | Project treatment |
| ---: | --- |
| 1. Define the goal and scope | Treat Stage 1 as practice and portfolio work. Predict three waterpoint states and use accuracy as the competition success measure. |
| 2. Gather the data | Load immutable feature and label files, join labels by `id` and retain provenance. |
| 3. Explore the data | Use the target analysis, maintained findings report and predictor catalogue to cover structure, distributions, relationships and limitations. |
| 4. Clean and preprocess the data | Apply the three guarded structural removals. Fit missing-value handling, encoding and any scaling inside the training partition or current fold. |
| 5. Select and engineer features | Start with 36 candidates. Test provisional date, age, availability, coordinate and category treatments through held-out comparisons and ablations. |
| 6. Define the machine-learning task | Record multiclass classification, challenge accuracy, class imbalance and the diagnostic measures needed for the minority class. |
| 7. Partition the data | Complete: reserve a reproducible stratified 20% local test set and freeze five development folds. Add a grouped geographic sensitivity check later. |
| 8. Select and train candidate methods | Compare seven classifier families and five bounded soft-vote combinations with fold-fitted preprocessing. |
| 9. Evaluate and interpret the results | Select the Random Forest and histogram-boosting vote at 81.37%, then confirm 80.82% once on the local test. |
| 10. Deploy and iterate | Complete the full-data refit and submission validation; upload and record its public score, then return to target structure and geographic robustness. |

The lifecycle remains iterative. Early public submissions reached Steps 8 to 10,
but they did not complete the partitioning and local-evaluation gates in Steps 7
and 9.

## Implementation sequence

1. **Complete:** create [`src/`](../src/) for reusable project code.
2. **Complete:** add one public function,
   `remove_known_redundant_columns(raw_features)`, and keep the fixed raw schema,
   mappings and removals in `src/raw_feature_column_policy.json`.
3. **Complete:** preserve the prior overall audit under `data-audit/00-overall/`,
   import the shared source-data validators, call the fixed column-removal
   function for training and test features, and display a concise before/after
   column summary.
4. **Complete:** organise all 39 raw predictors into explicit folders with a
   type-specific breakdown, noteworthy findings and related-feature analysis.
5. **Complete:** build the validated 36-predictor modelling handoff, reserve the
   20% local test set and freeze five stratified development folds with recorded
   seeds and membership fingerprints.
6. **Complete:** establish the majority-class reference across the five frozen
   development folds: 54.31% mean accuracy, 100% `functional` recall and zero
   recall for the other two classes.
7. **Complete:** define the initial 29-feature policy and smoke-test fold-fitted
   median imputation, missing indicators, rare-category handling and one-hot
   encoding on one frozen development fold.
8. **Complete:** evaluate the initial preprocessor and constrained decision
   tree across all five frozen development folds, recording 74.98% mean
   accuracy and per-class confusion evidence.
9. **Complete:** evaluate fold-safe Extra Trees with the same initial feature
   policy, recording 78.84% mean accuracy and much stronger minority-class
   recall than the constrained tree.
10. **Complete:** evaluate histogram gradient boosting with the same feature
   values, frozen folds and explicit elapsed-time evidence, recording 80.21%
   mean accuracy in about three minutes.
11. **Complete:** retain aligned out-of-fold probabilities and evaluate the
   equal-weight Extra Trees and histogram boosting soft vote, recording 80.83%
   mean accuracy.
12. **Complete:** screen logistic regression, Gaussian naïve Bayes, KNN and
   Random Forest; Random Forest leads the single models at 80.59%.
13. **Complete:** compare a bounded set of simple ensembles and select the
   equal-weight Random Forest and boosting vote at 81.37%.
14. **Complete:** confirm 80.82% accuracy once on the local test, refit on all
   labelled rows and validate the competition submission.
15. **Next:** upload and record the candidate, then investigate target-class
   structure and grouped geographic robustness.
16. **Complete:** align the dashboard and supporting documentation with the
   canonical ten-step lifecycle.

Use the repository's current `.venv`. Formal automated tests are low priority
for this known, local data set. Continue using focused smoke checks against both
raw feature files; add a test suite later if the reusable code grows or begins
serving less controlled inputs.

## Acceptance criteria for fixed column removal

- Exactly one public function accepts either raw feature DataFrame.
- Calling it twice with the same input gives the same ordered result.
- The result contains 37 columns, including `id`, and removes only
  `quantity_group`, `payment` and `recorded_by`.
- The caller-owned DataFrame is unchanged.
- A changed duplicate, mapping or constant relationship causes a useful failure.
- The function loads no files, handles no labels or submission data and learns
  no statistics from training or test values.

## Decisions deferred

- Alternative missing-value representations and imputation strategies.
- Treatment of high-cardinality categorical fields.
- Geographic feature engineering and ablation of the elapsed-date feature.
- Removal of hierarchy columns after model comparison.
- Tuning ranges, target-class structure and grouped geographic sensitivity.
- Public leaderboard evidence for the prepared candidate.

These choices require training-partition evidence. The structural preparation
work can proceed without them.

## Resume point

Continue course Step 10, **Deploy and iterate**. Upload the candidate generated
by `05-final-model-and-submission.ipynb` and record its public score without
using that score to disguise the retained local evidence. Then extend the
target-label audit into semantics, pairwise separability and hierarchical or
ordinal formulations using development-only out-of-fold evidence. Grouped
geographic sensitivity remains the parallel robustness check.

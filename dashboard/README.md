# Competition dashboard maintainer notes

This directory contains a static status dashboard for the DrivenData *Pump It
Up* work:

- `index.html`: dashboard content and current values;
- `styles.css`: compact responsive presentation.

The main workflow follows the course's ten-step machine-learning lifecycle.
Keep the step numbers and intent stable. Project-specific checkpoints and
evidence belong within those steps rather than replacing them with a second
process vocabulary.

The dashboard is for personal project control, not portfolio presentation. It
should answer three questions quickly:

1. What measurable progress has been made?
2. What is being worked on now?
3. What should limit the amount of effort spent here?

Keep it dense, factual and usable within a normal desktop viewport. Avoid large
headings, decorative charts, generic project-management language and invented
time estimates.

The top summary has one progress/current-work block followed by exactly five
headline metrics. Do not turn current work back into a narrow metric column or
keep appending columns; add detailed information to the compact panels below.

## Current snapshot

The headline values come from `../project-status.json` when the site is served
locally. The matching HTML values are a readable fallback. The current snapshot
is dated 15 August 2026:

| Metric | Current value |
| --- | ---: |
| Overall plan checkpoints | 19 / 28 |
| Features fully examined | 36 / 36 (initial predictor audit organised) |
| Locally evaluated models | 0 |
| Candidate methods trained | 2, plus one soft-vote ensemble |
| DrivenData submissions | 6 |
| Best leaderboard score | 0.8170 |
| Provisional target score | 0.8225 |
| Chart ceiling | 0.8500 |
| Daily submissions used | 3 / 3 UTC |

The current task is to turn the promising public result into reproducible local
evidence. The process remains iterative, so work has reached later steps without
making the partitioning and evaluation gates disappear:

- **Step 3, explore the data (4 / 4):** the maintained overall findings report,
  predictor catalogue and 117 per-predictor notebooks cover structure,
  distributions, limitations, relationships and feature families.
- **Step 4, clean and preprocess (1 / 3):** guarded structural cleanup is
  implemented. The temporary full-data experiment tried missing markers,
  imputation, encoding, date parts, pump age and coordinate masking, but those
  learned treatments do not count as fold-safe evidence.
- **Step 6, define the task (3 / 4):** multiclass classification, accuracy,
  class balance and the executed label-frame integrity check are recorded.
  Exact feature/label ID alignment remains.
- **Step 7, partition the data (0 / 3):** choose the split policy, create the
  reproducible stratified split and record its local non-model baseline.
- **Step 8, select and train candidates (2 / 3):** Extra Trees and histogram
  gradient boosting, plus their soft vote, are reproducible. Local validation
  must decide between them.
- **Step 9, evaluate and interpret (2 / 3):** constant and candidate public
  comparisons are recorded. Held-out comparison, per-class recall and the
  confusion matrix remain.
- **Step 10, deploy and iterate (2 / 3):** the candidate submissions, hashes and
  scores are recorded. Formal selection and the next evidence-led iteration
  remain.

Six submissions have been made across two UTC dates:

| Submission | Date | DrivenData score |
| --- | --- | ---: |
| All `functional` | 14 August | 0.5461 |
| All `functional needs repair` | 14 August | 0.0719 |
| All `non functional` | 14 August | 0.3820 |
| Extra Trees | 15 August | 0.8050 |
| Histogram gradient boosting | 15 August | 0.7968 |
| Equal-weight soft-vote ensemble | 15 August | **0.8170** |

The 0.5461 all-`functional` result remains the simple leaderboard floor. The
0.8170 ensemble is the best public result and sits 0.55 percentage points below
the provisional 0.8225 target. Both candidate models fitted the complete
labelled data, so the public results are informal evidence only: there is no
retained train/validation estimate and the **Models evaluated** counter remains
zero.

The target-integrity track currently awards three of four checkpoints. The
training shares, task/metric implications and executable label-frame integrity
are recorded. Exact feature/label ID alignment is the remaining checkpoint;
the validator exists in `src/source_data_validation.py` but its evidence has not
yet been retained in the formal baseline workflow.

Working competition context was supplied manually on 14 August 2026 and has
not been fetched or independently verified by the dashboard:

| Reference point | Accuracy |
| --- | ---: |
| Widely achieved ("everyone and their dog") | approximately 82.10% |
| Achieved by approximately 2,500 teams | approximately 82.20% |
| Provisional project target | **82.25%** |
| Reported world record | 82.299% |

The compact daily-best chart records 54.61% on 14 August and 81.70% on 15
August. Its scale now runs from 50% to 85%, with the 82.25% provisional target
drawn inside that range.

## Metric definitions

Keep these definitions stable so that the counters remain meaningful.

### Overall plan checkpoints

The 28 checkpoints are grouped under the course lifecycle. Step 5 feature
coverage has its own 36-feature denominator and is excluded from the overall
checkpoint total:

| Course step | Current | Available |
| --- | ---: | ---: |
| 1. Define the goal and scope | 3 | 3 |
| 2. Gather the data | 2 | 2 |
| 3. Explore the data | 4 | 4 |
| 4. Clean and preprocess the data | 1 | 3 |
| 5. Select and engineer features | 36 | Separate 36-feature metric |
| 6. Define the machine-learning task | 3 | 4 |
| 7. Partition the data | 0 | 3 |
| 8. Select and train candidate methods | 2 | 3 |
| 9. Evaluate and interpret results | 2 | 3 |
| 10. Deploy and iterate | 2 | 3 |
| **Overall checkpoint total** | **19** | **28** |

Increment a track only when its exit evidence exists. Do not award partial
credit for activity alone.

### Features fully examined

A feature counts when it has:

- documented distribution, missingness or placeholder analysis;
- a documented relationship-to-target review where appropriate;
- a recorded modelling treatment or an explicit decision to defer treatment;
- remaining speculative deep dives clearly separated from the current decision.

This means "sufficiently examined for the present modelling loop", not
"exhaustively understood". All 36 candidate predictors now meet this
breadth-first standard and are organised into audit reports; `amount_tsh` also
has deeper feature-specific analysis.

### Models evaluated

Count a model family once after it has a reproducible local evaluation against
the agreed split. Hyperparameter variants do not each increase the headline
counter. The two trained candidates and their soft vote therefore do not yet
increase this counter: they were fitted to the full labelled data and compared
only through public leaderboard scores. Record detailed experiments elsewhere.

### Submission metrics

- **DrivenData submissions:** lifetime submissions made for this project.
- **Best score:** highest recorded DrivenData leaderboard score, not the local
  validation score.
- **Used today:** submissions made during the current UTC day, out of the
  competition limit of three. Reset this value when the UTC date changes.

### Daily best score chart

Add at most one bar per UTC date, using the best leaderboard score known at the
end of that day. The current chart is scaled from 50% to 85%.
Calculate the CSS height as:

```text
--score-height = (percentage score - 50) ÷ 35 × 100%
```

For example, 54.61% becomes `13.17%` and 81.70% becomes `90.57%`. If results
reach or exceed 85%, change the axis and document the new scale rather than
allowing bars to overflow. Keep the full 82.25% target in the summary and
benchmark panel regardless of the chart's temporary scale.

## Real risks and controls

Retain the two current risks unless the working context genuinely changes:

1. **Graded-content displacement:** optional Stage 1 competition work consumes
   time intended for assessed course content. Timebox Stage 1; graded work wins
   when they compete.
2. **Feature rabbit hole:** one relationship receives disproportionate analysis
   and completionism then demands the same depth everywhere. Work breadth-first
   across all 36 predictors; promote a deep dive only when baseline evidence
   supports it.

These are workload risks, not modelling risks. Leakage, imbalance and
cardinality still matter technically, but they belong in analysis and model
evaluation rather than this personal constraint list.

## Update routine

After a meaningful modelling or submission session:

1. Update the snapshot date and supported headline values in
   `../project-status.json`.
2. Keep the fallback values in `index.html` aligned with the status file.
3. Update the course workflow rows and their evidence or next gate without
   changing the ten step numbers or meanings.
4. Update submission totals, daily UTC usage and best score together.
5. Add or update the daily-best chart bar when a day's score is final.
6. Keep this README's current snapshot aligned with the page.
7. Recheck internal links, HTML IDs, CSS brace balance and `git diff --check`.

Do not load competition data from the page, add a backend or introduce a build
tool for the current iteration. If the dashboard later becomes data-driven,
define a small non-sensitive status file rather than reading licensed raw data
or notebook output directly.

## Likely next changes

- Run and retain the exact feature/label ID-alignment check to complete the
  target-integrity evidence.
- Create the reproducible stratified split and compare the local majority-class
  rate with the 0.5461 all-`functional` leaderboard floor.
- Refit Extra Trees and histogram gradient boosting using preprocessing learned
  only from the training fold, then compare both candidates and their soft vote
  on the same held-out rows.
- Record accuracy, per-class recall and the confusion matrix before using the
  0.8170 public score to justify model selection or further submissions.
- Use local errors and ablations to decide which provisional feature treatments
  from the temporary full-data experiment deserve to remain.

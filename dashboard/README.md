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
| Overall plan checkpoints | 10 / 28 |
| Features fully examined | 36 / 36 (initial predictor audit organised) |
| Models evaluated | 0 |
| DrivenData submissions | 3 |
| Best leaderboard score | 0.5461 |
| Provisional target score | 0.8225 |
| Near-term chart ceiling | 0.6000 |
| Daily submissions used | 3 / 3 UTC |

The active work spans several course steps because the process is iterative:

- **Step 4, clean and preprocess:** turn the predictor audit and recorded
  removals into fold-safe preprocessing.
- **Step 6, define the task (2 / 4):** the training distribution and baseline
  implications are recorded. Run the committed label-frame and exact ID
  alignment checks from a clean kernel and retain their output.
- **Step 7, partition the data:** align labels by `id`, create a reproducible stratified split
  and record a local dummy-classifier score.
- **Step 5, select and engineer features:** convert the completed breadth-first
  predictor audit into explicit treatment decisions, with deep investigation
  deferred unless modelling evidence justifies it.

The target audit, breadth-first predictor coverage and dataset-wide exploration
represent a rough minimum of ten hours of foundation work. This is planning
context rather than a deadline; feature or relationship deep dives add to it.

Loop 0 is complete. One constant submission was made for each permitted class:

| Constant prediction | DrivenData score |
| --- | ---: |
| `functional` | **0.5461** |
| `functional needs repair` | 0.0719 |
| `non functional` | 0.3820 |

The all-`functional` score is both the current best leaderboard result and the
"Any chump can get here" floor that subsequent models need to beat. It is the
best of the three constant-class submissions; the local target audit still
needs executable label-frame and exact ID-alignment evidence.

The target audit currently awards two checkpoints:

- training shares are recorded as 54.31% `functional`, 7.27% `functional needs
  repair` and 38.42% `non functional`;
- the three constant-class public scores are close to those shares and provide
  no material evidence of class-prior shift.

The two remaining checkpoints are executable label-frame integrity and exact
feature/label ID alignment evidence. The validators exist in
`src/source_data_validation.py`, but the refactored notebook cells have not yet
been rerun and therefore do not count as completed evidence.

Working competition context was supplied manually on 14 August 2026 and has
not been fetched or independently verified by the dashboard:

| Reference point | Accuracy |
| --- | ---: |
| Widely achieved ("everyone and their dog") | approximately 82.10% |
| Achieved by approximately 2,500 teams | approximately 82.20% |
| Provisional project target | **82.25%** |
| Reported world record | 82.299% |

The dashboard also includes a compact daily-best chart. Its initial point is
54.61% on 14 August 2026. The chart uses 60% as a near-term ceiling so early
movement remains visible; 82.25% remains the actual provisional target.

## Metric definitions

Keep these definitions stable so that the counters remain meaningful.

### Overall plan checkpoints

The 28 checkpoints are grouped under the course lifecycle. Step 5 feature
coverage has its own 36-feature denominator and is excluded from the overall
checkpoint total:

| Course step | Checkpoints |
| --- | ---: |
| 1. Define the goal and scope | 3 |
| 2. Gather the data | 2 |
| 3. Explore the data | 4 |
| 4. Clean and preprocess the data | 3 |
| 5. Select and engineer features | Separate 36-feature metric |
| 6. Define the machine-learning task | 4 |
| 7. Partition the data | 3 |
| 8. Select and train candidate methods | 3 |
| 9. Evaluate and interpret results | 3 |
| 10. Deploy and iterate | 3 |
| **Total** | **28** |

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
counter. Record detailed experiments elsewhere.

### Submission metrics

- **DrivenData submissions:** lifetime submissions made for this project.
- **Best score:** highest recorded DrivenData leaderboard score, not the local
  validation score.
- **Used today:** submissions made during the current UTC day, out of the
  competition limit of three. Reset this value when the UTC date changes.

### Daily best score chart

Add at most one bar per UTC date, using the best leaderboard score known at the
end of that day. The current chart is deliberately scaled from 50% to 60%.
Calculate the CSS height as:

```text
--score-height = (percentage score - 50) × 10%
```

For example, 54.61% becomes `46.1%`. If results reach or exceed 60%, change the
axis and document the new scale rather than allowing bars to overflow. Keep the
full 82.25% target in the summary and benchmark panel regardless of the chart's
temporary scale.

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

- Rerun the source-frame validation cells from a clean kernel and retain their
  output to complete the target-integrity evidence.
- Create the reproducible stratified split and compare the local majority-class
  rate with the 0.5461 all-`functional` leaderboard floor.
- Start the dataset-wide breadth-first summaries before choosing further
  feature deep dives.
- Move the evaluation track forward after the aligned stratified split exists.
- Increment feature coverage only as features meet the definition above.

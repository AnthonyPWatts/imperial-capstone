# Competition dashboard maintainer notes

This directory contains a static status dashboard for the DrivenData *Pump It
Up* work:

Browse the [live dashboard](https://anthonypwatts.github.io/imperial-capstone/dashboard/)
or return to the [Capstone Hub](https://anthonypwatts.github.io/imperial-capstone/).

- `index.html`: dashboard content and current values;
- `styles.css`: compact responsive presentation.

The main workflow follows the course's ten-step machine-learning lifecycle.
Keep the step numbers and intent stable. Project-specific checkpoints and
evidence belong within those steps rather than replacing them with a second
process vocabulary.

The dashboard supports personal project control rather than portfolio
presentation. It should answer three questions quickly:

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

The page loads headline values from `../project-status.json` through GitHub
Pages or the local server. The dashboard uses matching HTML values if the
status request fails. The current snapshot is dated 21 August 2026:

| Metric | Current value |
| --- | ---: |
| Overall plan checkpoints | 28 / 28 |
| Features fully examined | 36 / 36 (initial 29-feature policy selected) |
| Locally evaluated models | 12 |
| Candidate methods trained | 7, plus five soft-vote ensembles |
| DrivenData submissions | 7 |
| Best leaderboard score | 0.8223 |
| Provisional target score | 0.8225 |
| Chart ceiling | 0.8500 |
| Daily submissions used | 1 / 3 UTC |

The current task is investigating target-class structure and grouped geographic
robustness. The process remains iterative, so leaderboard feedback does not
replace retained local model-selection evidence:

- **Step 3, explore the data (4 / 4):** the maintained overall findings report,
  predictor catalogue and 117 per-predictor notebooks cover structure,
  distributions, limitations, relationships and feature families.
- **Step 4, clean and preprocess (3 / 3):** guarded structural cleanup and the
  initial fold-fitted preprocessing pipeline are implemented. The complete
  pipeline has run independently across all five development folds without
  using local test or competition rows.
- **Step 5, select and engineer features (36 / 36 audited):** the initial policy
  uses elapsed recording time, valid pump age and explicit missing-state flags.
  High-cardinality fields and alternative hierarchy levels remain deferred for
  controlled ablations.
- **Step 6, define the task (4 / 4):** multiclass classification, accuracy,
  class balance, label integrity and exact feature/label ID alignment are
  recorded.
- **Step 7, partition the data (3 / 3):** a stratified 20% local test set and
  five development folds are frozen with recorded seeds and fingerprints. The
  five-fold majority reference is 54.31% accuracy.
- **Step 8, select and train candidates (3 / 3):** seven classifier families
  and five bounded equal-weight votes have reproducible fold-safe evaluations.
- **Step 9, evaluate and interpret (3 / 3):** Random Forest leads the single
  models at 80.59%. Its vote with histogram boosting leads development folds at
  81.37% and records 80.82% on the one-time local test.
- **Step 10, deploy and iterate (3 / 3):** both selected components were
  refitted on all labelled rows, the validated candidate CSV was uploaded and
  its `0.8223` public score is recorded.

Seven submissions have been made across three UTC dates:

| Submission | Date | DrivenData score |
| --- | --- | ---: |
| All `functional` | 14 August | 0.5461 |
| All `functional needs repair` | 14 August | 0.0719 |
| All `non functional` | 14 August | 0.3820 |
| Extra Trees | 15 August | 0.8050 |
| Histogram gradient boosting | 15 August | 0.7968 |
| Equal-weight Extra Trees and boosting vote | 15 August | 0.8170 |
| Equal-weight Random Forest and boosting vote | 21 August | **0.8223** |

The 0.5461 all-`functional` result remains the simple public leaderboard floor;
the frozen development folds produce a separate 0.5431 local majority
reference. The Random Forest and histogram-boosting vote now leads the public
results at `0.8223`, an absolute improvement of `0.0053` over the earlier Extra
Trees and boosting vote. Formal comparison covers seven single-model families.
Random Forest leads those at 80.59% mean accuracy. Its equal-weight vote with
histogram boosting leads the bounded ensemble round at 81.37%, including 34.28%
repair recall and 78.03% non-functional recall. The same frozen workflow records
80.82% on the one-time local test, with 32.10% repair recall and 77.81%
non-functional recall. Seven single models and five soft-vote workflows raise
the **Models evaluated** counter to twelve.

The target-integrity track now awards all four checkpoints. Training shares,
task and metric implications, label-frame integrity and exact feature/label ID
alignment are retained in the formal baseline workflow.

Working competition context was supplied manually on 14 August 2026 and has
not been fetched or independently verified by the dashboard:

| Reference point | Accuracy |
| --- | ---: |
| Widely achieved ("everyone and their dog") | approximately 82.10% |
| Achieved by approximately 2,500 teams | approximately 82.20% |
| Provisional project target | **82.25%** |
| Reported world record | 82.299% |

The compact daily-best chart records 54.61% on 14 August, 81.70% on 15 August
and 82.23% on 21 August. Its scale runs from 50% to 85%, with the 82.25%
provisional target drawn inside that range.

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
| 4. Clean and preprocess the data | 3 | 3 |
| 5. Select and engineer features | 36 | Separate 36-feature metric |
| 6. Define the machine-learning task | 4 | 4 |
| 7. Partition the data | 3 | 3 |
| 8. Select and train candidate methods | 3 | 3 |
| 9. Evaluate and interpret results | 3 | 3 |
| 10. Deploy and iterate | 3 | 3 |
| **Overall checkpoint total** | **28** | **28** |

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

Count each materially distinct model family or declared ensemble workflow once
after it has a reproducible local evaluation against the agreed development
folds or local test. Hyperparameter variants do not each increase the headline
counter. The cross-validated majority reference is a non-model benchmark and
does not increase it. The seven single-model families are the constrained tree,
logistic regression, Gaussian naïve Bayes, KNN, Extra Trees, histogram boosting
and Random Forest. Five declared soft votes bring the counter to twelve. Record
detailed experiments elsewhere.

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

For example, 54.61% becomes `13.17%`, 81.70% becomes `90.57%` and 82.23%
becomes `92.09%`. If results reach or exceed 85%, change the axis and document
the new scale rather than allowing bars to overflow. Keep the full 82.25%
target in the summary and benchmark panel regardless of the chart's temporary
scale.

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

- Extend the target-label audit into semantics, pairwise separability and
  possible ordinal or hierarchical formulations.
- Define grouped geographic sensitivity without changing the selected feature
  policy; begin with the lower-cost booster before repeating a full ensemble.
- Use out-of-fold errors and controlled ablations to decide whether class
  weighting, target structure or a feature-family change deserves promotion.

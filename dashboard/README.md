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
status request fails. The current snapshot is dated 23 August 2026:

| Metric | Current value |
| --- | ---: |
| Overall plan checkpoints | 68 / 68 |
| Features fully examined | 36 / 36 (physical hierarchy screens retained granular features) |
| Locally evaluated models | 21 |
| Candidate methods trained | 11, plus ten ensemble workflows |
| DrivenData submissions | 15 |
| Best leaderboard score | 0.8298 |
| Achieved provisional target | ~~0.8225~~ |
| Stretch target score | 0.8260 |
| Chart ceiling | 0.8500 |
| Daily submissions used | 3 / 3 UTC |

The model search is complete. The final submission replaced only the deep
XGBoost seed in the already-fixed six-component archive recipe. Seed 20260824
reached 81.8981% on development, 81.3973% on the used local test and 0.8298 on
the public leaderboard. It was observed at rank 2, 0.0001 behind the displayed
leader. The process remains iterative, so this leaderboard result does not
replace the retained local model-selection evidence:

- **Step 3, explore the data (4 / 4):** the maintained overall findings report,
  predictor catalogue and 117 per-predictor notebooks cover structure,
  distributions, limitations, relationships and feature families.
- **Step 4, clean and preprocess (3 / 3):** guarded structural cleanup and the
  initial fold-fitted preprocessing pipeline are implemented. The complete
  pipeline has run independently across all five development folds without
  using local test or competition rows.
- **Step 5, select and engineer features (36 / 36 audited):** the initial policy
  uses elapsed recording time, valid pump age and explicit missing-state flags.
  The organisation, numeric, management and physical hierarchy loops are
  complete; extraction group-only remains a named parsimony candidate.
- **Step 6, define the task (4 / 4):** multiclass classification, accuracy,
  class balance, label integrity and exact feature/label ID alignment are
  recorded.
- **Step 7, partition the data (3 / 3):** a stratified 20% local test set and
  five development folds are frozen with recorded seeds and fingerprints. The
  five-fold majority reference is 54.31% accuracy.
- **Step 8, select and train candidates (3 / 3):** eleven classifier families
  and ten bounded, calibrated or bagged ensemble workflows have reproducible
  fold-safe evaluations; the ANN family has seven fixed configurations.
- **Step 9, evaluate and interpret (3 / 3):** the seed-20260824 deep archive
  reaches 81.8981% on the frozen development folds and 81.3973% on the used
  local test. Its small seed advantage was reported with paired uncertainty.
- **Step 10, deploy and iterate (3 / 3):** the exact 14,850-row seed-20260824
  artefact scored `0.8298`, exceeded the `0.8260` target and was observed at
  rank 2 on 23 August 2026.

Fifteen submissions have been made across five UTC dates:

| Submission | Date | DrivenData score |
| --- | --- | ---: |
| All `functional` | 14 August | 0.5461 |
| All `functional needs repair` | 14 August | 0.0719 |
| All `non functional` | 14 August | 0.3820 |
| Extra Trees | 15 August | 0.8050 |
| Histogram gradient boosting | 15 August | 0.7968 |
| Equal-weight Extra Trees and boosting vote | 15 August | 0.8170 |
| Equal-weight Random Forest and boosting vote | 21 August | 0.8223 |
| 55% XGBoost and 45% Random Forest vote | 21 August | **0.8241** |
| 60% XGBoost depth bag and 40% Random Forest vote | 21 August | 0.8240 |
| Source-plus-class XGBoost and Random Forest vote | 22 August | **0.8246** |
| Waterpoint-XGBoost and source-class Random Forest vote | 22 August | 0.8244 |
| 2.5× repair-oversampled XGBoost depth bag | 22 August | 0.8174 |
| Complete-identity vote | 23 August | 0.8259 |
| Archive synthesis | 23 August | 0.8288 |
| Deep archive synthesis, seed 20260824 | 23 August | **0.8298** |

The 0.5461 all-`functional` result remains the simple public leaderboard floor;
the frozen development folds produce a separate 0.5431 local majority
reference. The seed-20260824 deep archive now leads the project at `0.8298`,
0.0010 above the archive synthesis and 0.0052 above the earlier source-plus-
class leader. The displayed competition leader scored `0.8299` when the
project result was observed.

Formal comparison now covers eleven single-model families. Random Forest remains
the strongest family from the earlier broad comparison at 80.59%; the new
XGBoost screen supplies the better ensemble component. The leading XGBoost and
Random Forest vote reaches 81.625% on the frozen development folds, including
34.859% repair recall. The earlier forest and histogram-boosting vote remains
the last workflow assessed on the one-time local test, where it recorded 80.82%
accuracy, 32.10% repair recall and 77.81% non-functional recall. Eleven
single-model families and ten ensemble workflows raise the **Models evaluated**
counter to twenty-one. The seven ANN configurations and 84 no-refit blends are
reported as variants and therefore increase the headline family count only once.

The target-integrity track now awards all four checkpoints. Training shares,
task and metric implications, label-frame integrity and exact feature/label ID
alignment are retained in the formal baseline workflow.

Working competition context was supplied manually on 14 August 2026 and has
not been fetched or independently verified by the dashboard:

| Reference point | Accuracy |
| --- | ---: |
| Widely achieved ("everyone and their dog") | approximately 82.10% |
| Achieved by approximately 2,500 teams | approximately 82.20% |
| Achieved provisional target | ~~82.25%~~ |
| Stretch target | **82.60%** |
| Reported world record | 82.99% |

The compact daily-best chart records 54.61% on 14 August, 81.70% on 15 August,
82.41% on 21 August, 82.46% on 22 August and 82.98% on 23 August. Its scale
runs from 50% to 85%, with the 82.60% stretch target drawn inside that range.
The achieved 82.25% provisional target remains visible as a struck-through
benchmark row.

## Metric definitions

Keep these definitions stable so that the counters remain meaningful.

### Overall plan checkpoints

The headline counter is the complete project evidence ledger maintained in
`../project-status.json`. The smaller progress values shown in the course table
are coarse exit gates and do not share that denominator. Step 5 feature
coverage has its own 36-feature denominator and is also excluded from the
headline counter. Keep these course exit gates stable:

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
| **Course exit-gate total** | **28** | **28** |

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
does not increase it. The eleven single-model families are the constrained tree,
logistic regression, Gaussian naïve Bayes, KNN, Extra Trees, histogram boosting,
Random Forest, CatBoost, XGBoost, LightGBM and multilayer perceptron. Ten
declared soft-vote, calibrated-stack and bagged-vote workflows bring the counter
to twenty-one. Record
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

For example, 54.61% becomes `13.17%`, 81.70% becomes `90.57%`, 82.41%
becomes `92.60%`, 82.46% becomes `92.74%` and 82.98% becomes `94.23%`. If
results reach or exceed 85%, change the axis and document the new scale rather
than allowing bars to overflow. Keep the full 82.60% stretch target in the
summary and benchmark panel regardless of the chart's temporary scale.

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

- Keep the rank explicitly labelled as an observation, because the live
  leaderboard can move while the score remains durable evidence.
- Preserve the exact seed-20260824 artefact, manifest, component recipe and
  SHA-256 alongside the screenshot.
- Record the private-leaderboard result when it becomes available; do not
  reopen adaptive public-leaderboard tuning.

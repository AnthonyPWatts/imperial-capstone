# Results

**Week 6 submitted; awaiting results.** The project owner confirmed all eight
submissions on **31 October 2026**. There are **seven future queries per
function** and still **215 measured observations through round 5**. The
[submission record](../submissions/Week_06/README.md) retains the exact inputs.

Start with [results and progress](progress.md) for a chart of all eight
functions, round-by-round scores and the best inputs from returned queries.
The [Stage 2 introduction](../README.md) explains the experiment, and the
[methodology guide](../docs/methodology.md) explains how inputs are chosen.

The [Week 6 model prediction review](week-06-model-review.md) links all eight
executed prediction notebooks, per-function reports and offline GP/neural
explorers. Its original Function 1 proposal used signed-score UCB after completing
its fixed coverage plan. Neural comparators have lower held-out error for
Functions 5 and 6; the selected GP leads for the other six. Those original
proposals remain historical. The submitted selection instead uses Function 1's
central signed-mean point, one Function 5 exploration query and one focused
Function 6 neural-led test. The other five model proposals are retained.

The [Week 5 neural-network and surrogate comparison](week-05-surrogate-comparison.md)
provides separate retrospective evidence for the seven reflection criteria.
Eight executed notebooks compare numerical regression and good/bad classification,
with actual support vectors, checked input gradients and conditional boundaries.
These experiments use feedback through round 4 and predate the fifth-query
returns received on 30 October. Their validation scores are not new BBO outcomes.

The [Week 5 discussion draft](../docs/week-05-discussion-draft.md) preserves the
final seven-prompt reflection. Posting to the course discussion board was
confirmed by the project owner on **29 October 2026**. It records the suspected
F5 boundary maximum as a hypothesis and the evidence against using the current
F3 classifier to guide queries; neither claim changes the measured results.

## Module 14 reporting cutoff

The module-end summary uses **191 observations through round 2**. Its evidence is
the [round 1 record](../submissions/round-01-results.json),
[round 2 record](../submissions/round-02-results.json) and the saved
`Function_N/Week_03/` analyses. Round 2 improved the incumbents for Functions
4, 5 and 7; that evidence informed the submitted Week 3 inputs.

The later returns below and the rolling progress chart are retained as
later history. They are excluded from the [Module 14 overview](../README.md)
and the strategy reflection explaining those query choices.

## Latest record: round 5, outside the Module 14 snapshot

- [Round 5 return](../submissions/round-05-results.json) — received on
  30 October 2026 at 14:57 GMT; all eight inputs match the saved Week 5 submissions.
  Function 4 improves to `0.6501304054949952`; the other incumbents are unchanged.
  The accumulated datasets contain **215 observations**.
- The [Week 6 observation and exploration notebooks](../notebooks/README.md#current-work-week-6-observation-and-exploration-reviews)
  have saved executions and figures for all eight functions. They compare the
  measured returns with the saved forecasts at the submitted inputs. The
  separate Week 6 model notebooks now record provisional next-query choices.

## Earlier record: round 4, outside the Module 14 snapshot

- [Round 4 return](../submissions/round-04-results.json) — received on
  23 October 2026 at 03:53 BST; all eight inputs match the saved Week 4 submissions.
  Function 7 improved to `2.6189682154023832`; the other incumbents are unchanged.
  The accumulated datasets contain **207 observations**.
- The [Week 5 observation notebooks](../notebooks/README.md#previous-work-week-5-observation-reviews)
  follow the current Week 4 pattern, with all eight executions and figures saved.
  All eight functions now have saved Week 5 model studies. The
  [submission and selection review](../submissions/Week_05/README.md)
  contains all eight submitted inputs. Submission was confirmed by the project
  owner on 29 October 2026; the round 5 return is recorded above.

## Earlier record: round 3, outside the Module 14 snapshot

- [Round 3 return](../submissions/round-03-results.json) — received on
  14 October 2026 at 10:16 BST; all eight inputs match the saved Week 3 submissions.
  Function 7 improved its incumbent to `2.55241765269229`; the other bests are unchanged.
  The accumulated datasets contain **199 observations**.
- The [Week 4 notebooks](../notebooks/README.md) and independent local CSV
  snapshots retain the studies that selected the Week 4 queries. Submission
  was confirmed on 20 October 2026; the round 4 return is recorded above.

## Earlier evaluation: round 2

- [Round 2 return](../submissions/round-02-results.json) — received on
  8 October 2026; all eight inputs match the saved Week 2 submissions.
  Functions 4, 5 and 7 improved their incumbents. The accumulated datasets
  contain **191 observations**.
- The [Week 3 notebooks](../notebooks/README.md) review the round 2 returns
  for all eight functions and record the next selections.

Week 3 submission was confirmed on 12 October 2026 and its results arrived on
14 October. See the [submission log](../submissions/README.md).

## Earlier evaluation: round 1

- [Function 1: first returned result](function-1-round-01-analysis.md) —
  5 October 2026; new minimum, unchanged incumbent, updated coverage and model checks.
- [Function 2: first returned result](function-2-round-01-analysis.md) —
  5 October 2026; fifth-ranked response near the incumbent, with updated
  noisy-response model checks and conditional next-query comparisons.
- [Round 1 return](../submissions/round-01-results.json) — confirmed values for
  all eight functions; the datasets contained 183 observations at that point.

The [Week 2 exploration notebooks](../notebooks/README.md#previous-work-week-2)
review the round 1 return for all eight functions. Standalone round 1 reports
in this directory cover Functions 1 and 2.

Keep publishable summaries of each optimisation round here. At minimum, record
the best observed input and output for every function and show progress against
the number of evaluations.

Do not copy course-restricted source arrays into the result files.

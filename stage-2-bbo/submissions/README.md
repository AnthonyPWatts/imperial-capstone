# Submission log

## Module 14 reporting cutoff

The [Module 14 overview](../README.md) stops at the submitted
[Week 3 inputs](Week_03/submissions.txt), using **191 observations through
round 2**. Submission confirmation was recorded on 12 October 2026; the exact
portal timestamp was not supplied. The round 1 and 2 records are the feedback
available for those decisions. The subsequent third-query return is retained
in this log but is outside the module-end snapshot and strategy reflection.

The first README draft is posted on Canvas. The strategy reflection is drafted;
submission and peer-comment completion are not yet confirmed in the project record.

## Copy-and-paste portal inputs

[Week 6](Week_06/submissions.txt) contains the eight prepared inputs in function
order. **Status: submitted; awaiting results.** The project owner confirmed
submission on **31 October 2026**; the exact portal time was not supplied. The
[selection review](Week_06/README.md) retains Functions 2–4, 7 and 8, follows
the Function 1 central-region hunch at `0.537942-0.522217`, switches Function 5
to a global information/coverage query, and adopts Function 6's saved
neural ensemble candidate for one GP-cross-checked focused test. Bounds,
precision and duplicate checks passed against official/retained peer data and
all five earlier weekly input files. **Seven future queries remain per function.**
The submitted input file is unchanged. Results are pending; there are still
215 confirmed observations through round 5 and no new incumbent changes.

Function 1's agreed query follows through on a hunch. None of its outputs so
far is regarded as clearly non-zero in practical or source-detection terms;
the stored numerical values remain unchanged and no detection threshold has
been established. Function 2 is also agreed at **`0.700520-0.000114`**, with
uncertainty weight **1.96**. Function 3 is agreed at
**`0.721213-0.830963-0.001845`**, also with weight **1.96**, retaining the caveat
that the choice relies on peer data and the official-only GP proposes a different
region. Function 4 is agreed at
**`0.408963-0.412546-0.416464-0.438393`**, with weight **1.96** and the caveat
that the GP underpredicts the known positive incumbent. Function 5 is agreed at
**`0.000000-0.000000-0.999999-0.999999`** for one exploration query, retaining
the low-return expectation, unfinished local x3 check and unproven optimum.
Function 6 is agreed at **`0.398303-0.269220-0.627336-0.768441-0.000000`** for
one focused neural-led test. The small overall error advantage, worse weekly
error, tiny predicted gain, model disagreement and boundary extrapolation
remain caveats. Function 7 is agreed at
**`0.000000-0.135108-0.370684-0.306265-0.311144-0.649736`**, with weight
**1.96**, retaining the below-incumbent mean, extrapolation and weight-sensitive
point as caveats. Function 8 is agreed at
**`0.022327-0.096114-0.237034-0.202848-0.982361-0.880106-0.159960-0.678158`**,
with weight **1.96**, retaining sparse-data, extrapolation, modest mean-gain and
weight-sensitivity caveats. All eight choices were finalised by **31 October
2026**. Submission was subsequently confirmed by the owner on the same date.

[Week 5](Week_05/submissions.txt) contains all eight selected inputs, prepared
on **28 October 2026** and ordered from Function 1 to Function 8. The
[selection review](Week_05/README.md) records the retained Function 1 coverage
point, the reviewed Function 2–4 choices and the completed Function 5–8 model
studies. Dimensions, bounds, formatting and duplicate checks passed against
the current snapshots and earlier submissions. **Portal submission was confirmed
by the project owner on 29 October 2026; evaluation was confirmed by the
30 October results email.** The exact portal
submission timestamp was not supplied. Eight future queries remain per function.
The
[proposal record](Week_05/proposals.json) preserves evidence and source hashes.

## Latest evaluation: round 5

[`round-05-results.json`](round-05-results.json) records the Week 5 return received
on **30 October 2026 at 14:57 GMT**. All eight returned inputs match the submitted
queries; the five-round cumulative attachments match the retained history.
There are now **215 official observations**. Function 4 improves from
`-0.12991814521724931` to `0.6501304054949952`; the other incumbents are unchanged.
The [Week 6 notebooks](../notebooks/README.md#current-work-week-6-observation-and-exploration-reviews)
have saved descriptive reviews and figures for all eight functions. No Week 6
query has been selected by this rollover.

The separate [Week 6 model studies](../results/week-06-model-review.md) now
record provisional next queries for all eight functions. They include the
transition to model-guided signed-score UCB for Function 1 and neural
comparators for every function. No Week 6 portal submission is recorded.

## Earlier submitted inputs

[Week 4](Week_04/submissions.txt) contains the eight selected inputs, ordered
from Function 1 to Function 8, with one portal input per line. The project owner
confirmed on **20 October 2026** that the notebooks had been run and the existing
strategies would continue: saved coverage points for Functions 1–3 and the saved
model-led proposals for Functions 4–8. Dimensions, `[0, 1)` bounds, six-decimal
formatting and duplicate status were checked against the Week 4 observations
and earlier submission files. Portal submission was confirmed by the project
owner on **20 October 2026**; the exact submission time was not supplied.

[Week 3](Week_03/submissions.txt) contains the eight selected Week 3 inputs.
Submission was confirmed by the project owner on **12 October 2026**; the
exact portal submission timestamp was not supplied. Its return arrived on
14 October 2026 and all eight inputs matched. The saved notebooks retain their pre-submission wording;
this log records the later submission status.

[Week 2](Week_02/submissions.txt) contains the eight evaluated Week 2 inputs.
Each line contains only the portal input, ordered from Function 1 to Function 8.
The file preserves the submitted proposals; the separate results record below
confirms their evaluation.

Week 2 submission was confirmed on **8 October 2026**; the exact portal
submission timestamp was not recorded. Its model notebooks retain their
pre-submission wording; this log records the later evaluation status.

[Week 1](Week_01/submissions.txt) provides the same plain-text layout for the
eight previously evaluated inputs. The original JSON files remain available
as historical records of the proposals, evidence and returned results.

## Previous evaluation: Week 4

The Week 4 results email arrived on **23 October 2026 at 03:53 BST**.
[`round-04-results.json`](round-04-results.json) records all eight returned
points and values, before/after incumbents, source hashes and proposal matches.
Every input matches `Week_04/submissions.txt`; the four-round cumulative
attachments agree exactly with the retained earlier history. Only eight new
observations were added, giving **207 observations**.

Function 7 is the only improved incumbent, increasing from `2.55241765269229`
to `2.6189682154023832`. All eight Week 5 observation notebooks have saved
executions and independent CSV snapshots; see the
[notebook guide](../notebooks/README.md#previous-work-week-5-observation-reviews).
Function 1's [Week 5 model notebook](../notebooks/Function_1/Week_05/official/model_predictions.ipynb)
compares signed-score and magnitude hypotheses, alongside a separate cohort
comparison in its [report](../notebooks/Function_1/Week_05/report.md).
Its final saved coverage input, `0.912140-0.861783`, is retained; the model
choices are hypothetical.
Function 2's [Week 5 model notebook](../notebooks/Function_2/Week_05/official/model_predictions.ipynb)
now records an unsubmitted proposal with validation and sensitivity checks.
Function 3's [Week 5 model notebook](../notebooks/Function_3/Week_05/official/model_predictions.ipynb)
likewise records an executed model comparison and an unsubmitted proposal.
Function 4's [Week 5 model notebook](../notebooks/Function_4/Week_05/official/model_predictions.ipynb)
now records a new unsubmitted proposal and a separate cohort comparison in
the [week-level report](../notebooks/Function_4/Week_05/report.md).
Functions 5–8 now have executed Week 5 model studies and reports. The
[Week 5 selection review](Week_05/README.md) records the submitted choices,
including the adopted alternatives for Functions 2–4. Submission was confirmed
on 29 October 2026; results arrived on 30 October, as recorded above. Saved
notebook proposal statuses describe their pre-submission analyses; this log
records the later confirmation.

## Previous evaluation: Week 3

The Week 3 results email arrived on **14 October 2026 at 10:16 BST**.
[`round-03-results.json`](round-03-results.json) records all eight returned
points and values, before/after incumbents, source hashes and proposal matches.
All inputs match `Week_03/submissions.txt`. Function 7 improved from
`2.075221839308756` to `2.55241765269229`; the other incumbents are unchanged.
The cumulative attachments agree exactly with both retained earlier returns.
Only one new observation per function was added, giving 199 observations.
The independent Week 4 CSVs and notebooks are prepared; see the
[data notes](../data/README.md#week-4-notebook-snapshots).

## Previous evaluation: Week 2

The Week 2 results email arrived on **8 October 2026 at 18:42 BST**.
[`round-02-results.json`](round-02-results.json) records all eight returned
points and values, before/after incumbents, source hashes and proposal matches.
All eight inputs match `Week_02/submissions.txt`. Functions 4, 5 and 7 improved
their incumbents. The cumulative attachments agree exactly with the retained
Week 1 return; only one new observation per function was added.

The Week 3 consolidated snapshots are described in the
[data notes](../data/README.md#week-3-notebook-snapshots).
The precise submission time was not supplied by the results email.

## Previous evaluation: Week 1

The Week 1 results email arrived on **5 October 2026 at 00:38 BST**.
[`round-01-results.json`](round-01-results.json) records all eight returned
points and values, the before/after incumbents, source hashes and proposal
matches. All eight evaluated inputs match the saved first-query proposals.
The precise submission time was not supplied by the results email.

The original proposal file below is retained unchanged as a historical record.
Its unsubmitted status describes the time it was generated; the separate
results record establishes the later confirmed evaluation.
See the [Function 1 review](../results/function-1-round-01-analysis.md) and
[Function 2 review](../results/function-2-round-01-analysis.md).

## First-query proposal snapshot

[`round-01-proposals.json`](round-01-proposals.json) records the eight first-query
proposals, their rationale and computational evidence. It is explicitly marked
as unsubmitted and records source hashes because the analysis changes were not
committed when that snapshot was generated. The separate results record confirms
the subsequent evaluation; the snapshot's status is historical.

For each round, record:

- function number;
- proposed point at full precision and in portal format;
- acquisition method and key settings;
- source commit;
- returned observation;
- whether it improved the incumbent;
- brief rationale for the next round.

Before submission, validate the number of coordinates, the `[0, 1)` bounds,
six-decimal formatting, absence of spaces and duplicate status.

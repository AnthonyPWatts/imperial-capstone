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

## Latest confirmed evaluation

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

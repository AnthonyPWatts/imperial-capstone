# Submission log

## Copy-and-paste portal inputs

[Week 3](Week_03/submissions.txt) contains the eight approved Week 3 inputs,
ready for manual submission.

[Week 2](Week_02/submissions.txt) contains the eight evaluated Week 2 inputs.
Each line contains only the portal input, ordered from Function 1 to Function 8.
The file preserves the submitted proposals; the separate results record below
confirms their evaluation.

Week 2 submission was confirmed on **8 October 2026**; the exact portal
submission timestamp was not recorded. Its model notebooks retain their
pre-submission wording; this log records the later evaluation status.
Week 3 submission has not yet been confirmed.

[Week 1](Week_01/submissions.txt) provides the same plain-text layout for the
eight previously evaluated inputs. The original JSON files remain available
as historical records of the proposals, evidence and returned results.

## Latest confirmed evaluation

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

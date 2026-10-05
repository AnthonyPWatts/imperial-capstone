# Submission log

## Latest confirmed evaluation

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
as unsubmitted and records source hashes because the analysis changes are not
committed. No returned observations are invented or implied.

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

# Submission log

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

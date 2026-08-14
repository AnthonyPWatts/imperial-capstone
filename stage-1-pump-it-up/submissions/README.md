# Submission log

Record each material DrivenData submission, including:

- date and short model name;
- source commit;
- validation score;
- public and private leaderboard scores when available;
- what changed from the previous submission;
- the conclusion or next experiment.

Generated prediction files should only be added if the competition rules allow
their redistribution.

## 2026-08-14: constant-class baselines

These deterministic submissions filled every row of the supplied submission
template with one target label. They establish the leaderboard floor, verify
the submission workflow and estimate the marginal class balance of the scored
test rows.

| Submission | Training share | Public score | Difference |
|---|---:|---:|---:|
| All `functional` | 0.5431 | 0.5461 | +0.0030 |
| All `functional needs repair` | 0.0727 | 0.0719 | -0.0008 |
| All `non functional` | 0.3842 | 0.3820 | -0.0022 |

- Source commit: not applicable; predictions are deterministic constants.
- Validation score: the corresponding training share shown above.
- Private leaderboard score: not available.
- The three public scores sum to `1.0000`, as expected for exhaustive constant
  predictions under multiclass accuracy.
- The scored class balance is close to the training balance. This experiment
  gives no material evidence of class-prior shift, so no prior correction is
  indicated at this stage.
- The initial public leaderboard baseline to beat is `0.5461` from the majority
  `functional` class.
- Scores are rounded leaderboard aggregates; they do not disclose individual
  test labels or necessarily describe an unscored/private subset.

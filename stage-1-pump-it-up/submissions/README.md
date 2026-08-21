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

## 2026-08-15: early audit-led tree experiments

These submissions turned the settled structural audit findings into three
temporary full-data experiments before the formal modelling workflow. The
reproducible notebook and experiment notes are in
[`2026-08-15-early-experiments/`](2026-08-15-early-experiments/).

| Submission | Public score | Difference from strongest base |
| --- | ---: | ---: |
| Extra Trees | 0.8050 | — |
| Histogram gradient boosting | 0.7968 | -0.0082 |
| Equal-weight soft-vote ensemble | **0.8170** | **+0.0120** |

- Source commit: `68656d6` (`Added early competition submissions`).
- Private leaderboard scores: not available.
- Both candidate models fitted the complete supplied labelled data; no local
  train/validation split or performance estimate was retained.
- The ensemble improved on Extra Trees by 1.20 percentage points and histogram
  gradient boosting by 2.02 percentage points.
- The ensemble score is 27.09 percentage points above the earlier 0.5461
  constant-`functional` public baseline.
- These public results are an informal experiment only. Formal partitioning,
  candidate comparison and interpretation remain part of the later modelling
  workflow.

## 2026-08-21: forest-plus-boosting candidate prepared

Formal model comparison screened seven classifier families and five bounded
equal-weight probability votes. Random Forest plus histogram gradient boosting
led the five frozen development folds and was selected before opening the local
test.

| Evidence | Accuracy | Repair recall | Non-functional recall |
| --- | ---: | ---: | ---: |
| Five-fold development mean | 0.8137 | 0.3428 | 0.7803 |
| One-time local test | 0.8082 | 0.3210 | 0.7781 |

- Both components were refitted unchanged on all 59,400 labelled rows.
- The validated candidate contains 14,850 competition IDs in template order.
- Prediction shares are 60.59% `functional`, 3.82% `functional needs repair`
  and 35.58% `non functional`.
- Candidate SHA-256: `995f2eebda763a42ceca42db1915f10f7ccb494090bbeda0be48bd07f6fa18aa`.
- The generated CSV remains local and is excluded from Git.
- Submission status: **ready, not yet uploaded**.
- Public and private leaderboard scores: not available.
- Source commit: record before upload.

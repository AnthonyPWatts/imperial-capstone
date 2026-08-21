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

## 2026-08-21: forest-plus-boosting candidate submitted

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
- Public leaderboard score: **0.8223**, a new project best and an absolute
  improvement of `0.0053` over the earlier Extra Trees and boosting vote.
- Private leaderboard score: not available.
- Submission status: **submitted on 21 August 2026**.
- Source commit: `3340744` (`Added broad classifier screening and submission candidate`).

## 2026-08-21: bounded blend calibration produced no submission

Two fixed probability weights and a nested calibrated stack were compared with
the recreated equal-weight Random Forest and histogram-boosting incumbent. A
challenger had to improve mean development-fold accuracy and win at least three
of the five frozen folds.

| Candidate | Mean accuracy | Change | Fold wins |
| --- | ---: | ---: | ---: |
| Equal-weight incumbent | 81.402% | — | — |
| 40% Random Forest + 60% boosting | 81.425% | +0.023 pp | 2/5 |
| 60% Random Forest + 40% boosting | 81.271% | -0.130 pp | 0/5 |
| Calibrated stack | 81.406% | +0.004 pp | 2/5 |

None passed the gate, so no CSV was generated and no leaderboard submission was
made. The result supports retaining the simple equal vote and moving the next
submission experiment to an independent feature or model hypothesis. See
[`2026-08-21-blend-calibration/`](2026-08-21-blend-calibration/) for the full
runtime caveat and notebook link.

## 2026-08-21: XGBoost component challengers generated

A frozen-feature model-family screen evaluated CatBoost, XGBoost and LightGBM
standalones and compared each directly with the Random Forest and
histogram-boosting members of the incumbent. The reserved local test remained
unopened. No new standalone beat the incumbent, but replacing histogram
boosting with XGBoost produced five gate-passing combinations.

| Candidate | Mean accuracy | Change | Fold wins | Worst fold change |
| --- | ---: | ---: | ---: | ---: |
| 55% child-weight-1 depth-8 XGBoost + 45% Random Forest | **81.625%** | **+0.223 pp** | 4/5 | -0.021 pp |
| 60% equal depth-6/7/8 XGBoost bag + 40% Random Forest | 81.595% | +0.194 pp | 5/5 | +0.021 pp |

Both were refitted on all labelled rows and structurally validated against the
14,850-row competition template. Their OOF predictions disagree by 1.172% and
their competition predictions by 1.192%, satisfying the second-candidate
diversity requirement. Additional seeds reproduce the leading XGBoost
hyperparameter effect. The first candidate scored **0.8241** publicly, a new
project best and an improvement of 0.0018 over the prior 0.8223 leader. The
independently preselected second candidate scored **0.8240**, only 0.0001 behind
the first. Private leaderboard scores are not available. See
[`2026-08-21-model-family-screen/`](2026-08-21-model-family-screen/) for the
complete gate, stability result, prediction shares and hashes.

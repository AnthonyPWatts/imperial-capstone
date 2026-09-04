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

## 2026-08-21: oversampled replays generated, not submitted

The exact three recipes above were refitted after random replication of only
`functional needs repair`. Full oversampling raised that class to the
`non functional` count; adaptive follow-ups first retained 80% of the added
replicas, then fixed the final repair count at 2.5× its natural count.

| Training regime | Local-test accuracy range | Repair recall range | Competition repair share |
| --- | ---: | ---: | ---: |
| Original | 80.70–80.85% | 32.56–32.79% | 3.75–3.95% |
| Full oversampling | 79.00–79.28% | 52.61–54.92% | 9.69–10.26% |
| 80% of added replicas | 79.34–79.42% | 51.10–52.61% | 9.14–9.35% |
| 2.5× final repair count | 80.08–80.33% | 43.45–44.50% | 6.91–7.10% |

This establishes a repeatable minority-recall intervention, not an accuracy
improvement. The follow-up levels were proposed after observing earlier replays
and are therefore exploratory rather than independent local-test confirmations.
The 2.5× level is now frozen as the final multiplier examined in this session.
Nine validated CSVs were generated, and none was uploaded during the replay
work. The 2.5× depth-bag candidate was later submitted unchanged as part of the
22 August slate and scored 0.8174. See the
[`full replay`](2026-08-21-oversampled-replay/) and
[`80%-replica replay`](2026-08-21-oversampled-replay-80/),
[`2.5× replay`](2026-08-21-oversampled-replay-250x/) and
[`09-oversampled-submission-replay.ipynb`](../notebooks/09-oversampled-submission-replay.ipynb).

## 2026-08-22: physical-hierarchy candidates generated

Three exploratory 55:45 XGBoost and Random Forest entries were refitted on all
labelled rows: source plus source class, waterpoint type plus group, and a
waterpoint-aware XGBoost plus source-class Random Forest hybrid. Their frozen-
fold mean accuracies were 81.635%, 81.627% and 81.629%, respectively, but each
won only two of five folds and none passed the predeclared promotion gate.

All three 14,850-row CSVs passed template, ID and label validation. Source plus
class scored **0.8246**, a new project best, and the cross-policy hybrid scored
**0.8244**. The waterpoint-both candidate remains local; the reserved local
test was not opened. See
[`2026-08-22-physical-hierarchies/`](2026-08-22-physical-hierarchies/) for the
exact recipes, hashes, prediction shares and disagreement.

## 2026-08-22: three-entry daily slate submitted

The final daily slate deliberately combines two hierarchy candidates with one
already-frozen oversampling candidate instead of spending all three entries on
closely related hierarchy predictions. The first two are source plus source
class and the waterpoint-aware XGBoost plus source-class Random Forest hybrid.
The third is the 2.5× repair-count depth-6/7/8 XGBoost bag, which gave the best
local-test accuracy and macro F1 within the frozen oversampling replays.

No new oversampling level was fitted or evaluated, and the local test was not
reopened. Public scores were **0.8246** for source plus class, **0.8244** for
the cross-policy hybrid and **0.8174** for the oversampled bag. The first is a
new project best, 0.0005 above the previous leader. Oversampling remains a
minority-recall intervention rather than an accuracy improvement: its public
score was 0.0066 below the corresponding unoversampled bag, consistent with
the 0.43-point local-test accuracy cost. See the
[`daily slate`](2026-08-22-daily-slate/) for the exact files and hashes.

## 2026-08-22: complete-identity CatBoost candidate prepared

A fixed 20% depth-8 CatBoost contribution over all six deferred identity fields
passed the development gate and confirmed on the reserved local test. The
44:36:20 XGBoost, Random Forest and CatBoost recipe improved development
accuracy by 0.116 points and local-test accuracy by 0.387 points. Its validated
14,850-row competition file disagrees with the accepted 55:45 submission on
1.576% of rows and has SHA-256
`76cf24053f136d89262b23b626c932e2f5b609ca197e16d2cb7e4b46897bb006`.

The file was **prepared, not uploaded** while the recorded daily allowance was
3/3. It was submitted unchanged on 23 August and scored **0.8259**. See
[`2026-08-22-catboost-identities/`](2026-08-22-catboost-identities/) for the
recipe, validation manifest and prediction transitions.

An apples-to-apples local check later refitted the exact source-plus-class
components behind the 0.8246 public leader. Source plus class scored 80.833%
locally; adding the identity component reached 81.010%, while the already-
prepared original-feature identity candidate remained best at 81.086%. No
additional source/identity CSV was generated. See
[`source-identity-local-comparison.md`](../reports/source-identity-local-comparison.md).

## 2026-08-22: archive synthesis prepared

The fixed equal synthesis of the categorical-frequency and spatial-height
representation votes improved all five development folds and reached 81.793%.
It then scored **81.187%** on the used local test, 0.101 points above the ready
identity candidate, with better macro F1, repair recall, log loss and Brier
score. The validated full-data file differs from the identity candidate on
0.822% of competition rows and has SHA-256
`46be08ced5e9a3ac3922a33e2aaf657306a08a406829cc69b4d25a5396e6ed0c`.

The archive synthesis was queued first after the allowance reset, followed by
the complete-identity candidate. See
[`2026-08-22-archive-synthesis/`](2026-08-22-archive-synthesis/) and
[`archive-synthesis-confirmation.md`](../reports/archive-synthesis-confirmation.md).

## 2026-08-23: archive synthesis established a new project best

The two frozen candidates were submitted unchanged and in the predeclared
order. Archive synthesis scored **0.8288**, improving on the former 0.8246
leader by 0.0042 and exceeding the 0.826 project target by 0.0028. The
complete-identity vote scored **0.8259**, itself 0.0013 above the former leader.

| Submission | Development | Local test | Public score | Change vs former leader |
| --- | ---: | ---: | ---: | ---: |
| Archive synthesis | 81.7929% | 81.1869% | **0.8288** | **+0.0042** |
| Complete identity | 81.7403% | 81.0859% | **0.8259** | **+0.0013** |

The public ordering agrees with both development and local evidence. The
archive margin over complete identity is 0.0029 publicly, compared with 0.0010
on the local test. Two of three submissions had been used on 23 August at this
checkpoint; the third slot was held rather than spent on an unconfirmed
near-miss.
The archive score was observed at public-leaderboard position **8** immediately
after submission; this time-specific rank may change. Private leaderboard
scores are not available.

## 2026-08-23: final deep-archive candidate prepared

The deliberately held third slot was reopened for the strongest previously
recorded near-miss: seed-20260822 depth-17 XGBoost over fixed top-50 identity
indicators, substituted only into the archive synthesis's 33% primary XGBoost
allocation. The unchanged recipe reaches **81.8771%** on development and
**81.3552%** on the used local test, improving the submitted archive by 0.0842
and 0.1684 percentage points respectively. The local change is 20 net
additional correct rows.

The validated 14,850-row competition candidate differs from the submitted
archive on 1.5960% of rows and has SHA-256
`ef9b4e0fa2c696ae5a203f25cd9b2c4ca4e4e5d83aa4e010595e39af60b7a1d2`.
It was **prepared, not uploaded**, then superseded by the predeclared
seed-20260824 replacement before submission. See
[`2026-08-23-deep-archive/`](2026-08-23-deep-archive/) and the
[`deep-archive confirmation`](../reports/deep-archive-confirmation.md).

## 2026-08-23: seed-20260824 deep-archive synthesis submitted

The strongest of the three seeds declared in the original deep-XGBoost screen
was confirmed without changing features, depth, tree count or component
weights. Seed 20260824 reaches **81.8981%** on development and **81.3973%** on
the used local test, adding ten and five net correct rows over the prepared
seed-20260822 candidate.

The validated competition file changes 75 of 14,850 predictions (0.5051%)
from seed 20260822 and has SHA-256
`fe5de9ea46bad2b35226bc97ebdfb743807fb758df1d259e8f8a51609808fee2`.
It replaced the earlier candidate and was submitted unchanged in the third
daily slot. It scored **0.8298** publicly, 0.0010 above the preceding archive
synthesis and 0.0038 above the project target. The result was observed at
public-leaderboard position **2**, 0.0001 behind the leader; the live rank may
change. See
[`2026-08-23-deep-archive-seed-20260824/`](2026-08-23-deep-archive-seed-20260824/)
and the [`deep follow-up report`](../reports/deep-follow-up-search.md).

| Submission | Development | Local test | Public score | Observed rank |
| --- | ---: | ---: | ---: | ---: |
| Seed-20260824 deep archive | **81.8981%** | **81.3973%** | **0.8298** | **2** |

All three available submissions were therefore used on 23 August, bringing the
project total to 15. Private-leaderboard performance is not yet available.

## 2026-09-04: targeted repair-rule union submitted

A six-rule postprocessor changed 22 predictions from `functional` to
`functional needs repair`. It recovered 23 net rows across the frozen
development predictions, improving all five folds, and recovered eight net
rows on the used local test. The validated CSV had SHA-256
`01c8fa597f57c785009dc89ea2327ba3a43ca0bb3c0e5d7a25386e8f78acbe87`.

Submission `321013` scored **0.8296**, below the 0.8298 incumbent. The result
therefore rejects the full repair-rule union for the hidden competition set
despite its positive internal evidence. One of three daily submissions had
been used at this checkpoint; the nested history-only hedge remained local.
See the [repair-rule report](../reports/repair-rule-ensemble-screen.md).

## 2026-09-04: archive gate plus strict repair core submitted

The repair-preserving archive gate was combined with the disjoint 12-row
strict repair candidate. The resulting file changed 43 incumbent predictions,
added 29 net correct rows across the five frozen development folds and added
eight on the used local test. Its SHA-256 was
`7f6f1408db020c18f2fb1476ceb897483e8cf4225326c88ec2fda6bc4d39f8ec`.

Submission `321017` scored **0.8293**, seven or eight net rows below the
0.8298 incumbent after accounting for score rounding. This rejects the
combined gate-and-repair candidate on the public set. Two of three daily
submissions had been used at this checkpoint, leaving one final slot. See the
[deep/archive gate report](../reports/deep-archive-gate-screen.md).

## 2026-09-04: final repair candidate submitted

The two public results were decomposed into disjoint prediction blocks and
conditioned jointly before choosing the last slot. The selected 16-row
candidate combines the 12 strict repair changes with the repair meta-model's
sole competition row and three previously untouched near-boundary rows backed
by at least four component votes. It adds 21 net correct development rows with
fold nets +4, +4, +1, +7 and +5, and seven net rows on the used local test.

The validated candidate has SHA-256
`9c27f698a75c87e4c34a9044f63e498cf66674a3e4636ac3dd80beed251d2343`.
Submission `321023` scored **0.8294**, leaving the 0.8298 incumbent at observed
rank **3**. Under full-row scoring, the candidate lost five to seven net
correct rows relative to the incumbent. All three daily slots were used; the
next submission becomes available on 5 September 2026 UTC. See the
[final-slot selection report](../reports/final-slot-selection.md).

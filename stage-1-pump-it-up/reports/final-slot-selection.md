# Final-slot public-score-conditioned repair selection

## Decision

The 16-row candidate was submitted as submission `321023` and scored
**0.8294**. It did not displace the **0.8298** incumbent, which remained at
observed rank **3**. All three daily slots were used. The score establishes the
ordering, but not an exact net row count: the
[local competition reference](../instructions/competition-reference.md#files-and-submission)
requires one submitted prediction per test row and defines classification rate,
but does not say that the public leaderboard evaluates all 14,850 rows.

The selected override consisted
of the 12-row strict repair candidate, the repair meta-model's sole competition
selection (ID `60481`), and three previously untouched near-boundary rows where
at least four of six component models vote repair.

This is a first-place-or-bust choice rather than the safest expected-score
choice. An exact block model conditioned jointly on today's two public results
estimated a 25.6% chance of reaching 0.8301 or better, with deliberately broad
sensitivity checks ranging from 7.7% to 26.7%. The eight-row residual, meta and
consensus candidate has a slightly better expected net and lower downside, but
the selected 16-row candidate has the largest primary-model upper tail. A lower
final score cannot displace the retained 0.8298 best score.

## Conditional full-denominator arithmetic

The following was the arithmetic used for the historical decision. It is exact
only under the unsupported assumption that the displayed public score is
ordinary accuracy over all 14,850 competition rows:

| Submission | Public score | Possible correct rows | Net versus incumbent |
| --- | ---: | ---: | ---: |
| Incumbent | 0.8298 | 12,322 or 12,323 | — |
| 22-row repair union | 0.8296 | 12,319 or 12,320 | -4 to -2 |
| 43-row gate plus strict repair | 0.8293 | exactly 12,315 | -8 or -7 |
| Outright-first threshold | 0.8301 | exactly 12,327 | needs +5 or +4 |

Under that hypothetical denominator, the repair union decomposes into strict
core `C`, weaker history `W` and
identity `I`; its hidden net is therefore `C + W + I ∈ {-4, -3, -2}`. The
second candidate decomposes into archive gate `G`, the shared strict core `C`
and new residual history `H`; its net is `G + C + H ∈ {-8, -7}`. Cancelling
the shared block establishes `(G + H) - (W + I) ∈ {-5, -4}` but does not reveal
any individual row label.

The gate is excluded. A Jeffreys-smoothed Dirichlet-multinomial model using
the frozen out-of-fold block outcomes and both score constraints assigned the
31-row gate an expected hidden net of -7.67 and only a 0.2% probability of a
positive net. A final audit added an independent block `Q`: three untouched
rows with at least four repair votes and a blended repair-to-functional ratio
of at least 0.96. Its seven OOF analogues contain five repair and two functional
labels. The same model gives the following primary ranking:

| Candidate | Competition flips | Expected net | Chance of 0.8301+ |
| --- | ---: | ---: | ---: |
| **Strict, meta and consensus (`C + H + M + Q`)** | **16** | **+1.95** | **25.6%** |
| Residual, meta and consensus (`H + M + Q`) | 8 | +2.13 | 24.4% |
| Fixed core, meta and consensus (`C + M + Q`) | 12 | +1.44 | 17.5% |
| Prior strict plus meta (`C + H + M`) | 13 | +0.89 | 13.2% |

The conditioning event was itself rare under the validation-informed prior,
so these probabilities were decision aids rather than calibrated guarantees.
Because the local reference does not disclose the public denominator, the
table and block constraints must not be treated as recovered correct counts or
row-label evidence.

## Candidate evidence and integrity

The final rule union changes 43 frozen development predictions for **+21 net
correct** rows, with fold nets `+4, +4, +1, +7, +5`. It changes 11 used
local-test predictions for **+7 net correct** rows. The strict component remains
positive in every development fold. The consensus threshold is post-selected,
has only seven OOF analogues and no local trigger, and four votes do not imply
four independent models because three components are related forests.

- File: `02-strict-meta-consensus-repair.csv`
- Rows: 14,850 unique IDs in template order
- Changes versus the 0.8298 incumbent: 16
- Transition: `functional` to `functional needs repair` only
- SHA-256: `9c27f698a75c87e4c34a9044f63e498cf66674a3e4636ac3dd80beed251d2343`
- Status: submitted as `321023`; public score **0.8294**

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Maximise the chance that the one remaining submission reaches the 0.8301 outright-first threshold. |
| 2. Gather the data | Reuse supplied labels, frozen predictions, two public scores and their exact submitted row masks. |
| 3. Explore the data | Decompose both public probes into disjoint, interpretable prediction blocks. |
| 4. Clean and preprocess the data | Validate aligned IDs, class labels, score rounding and cached evidence fingerprints. |
| 5. Select and engineer features | Use rule membership and the independent meta/identity overlap as block evidence. |
| 6. Define the machine-learning task | Select a small multiclass postprocessing mask under a first-place utility. |
| 7. Partition the data | Retain the frozen five development folds and untouched local-test partition. |
| 8. Select and train candidate methods | Reuse fixed candidates; fit no new base model after observing public scores. |
| 9. Evaluate and interpret the results | Condition block outcome distributions jointly on both leaderboard probes and report sensitivity. |
| 10. Deploy and iterate | Generate one immutable CSV, verify it, then stage it for the final confirmed submission. |

## Reproduction

Run from the project root:

```powershell
.\.venv\Scripts\python.exe .\stage-1-pump-it-up\scripts\prepare_final_slot_candidate.py
```

The generator refuses to overwrite a different CSV and reconstructs all
labelled evidence from the committed candidate masks and frozen partitions.

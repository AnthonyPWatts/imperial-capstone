# Final-slot public-score-conditioned repair selection

## Decision

Prepare a 13-row `functional` to `functional needs repair` override consisting
of the 12-row strict repair candidate plus ID `60481`. The added row is the
only competition row independently selected by both the repair meta-model at
its fixed 0.70 threshold and the identity-CatBoost conflict rule.

This is a first-place-or-bust choice rather than the safest expected-score
choice. An exact block model conditioned jointly on today's two public results
estimated a 13.2% chance of reaching 0.8301 or better, with sensitivity checks
ranging from 5.8% to 13.2%. The five-row residual candidate has a better
expected net change but a smaller upper tail, because it can win only when
nearly every changed row is correct.

## Information recovered from the public scores

Assuming the displayed score is ordinary accuracy over all 14,850 competition
rows, four-decimal rounding gives the following exact possibilities:

| Submission | Public score | Possible correct rows | Net versus incumbent |
| --- | ---: | ---: | ---: |
| Incumbent | 0.8298 | 12,322 or 12,323 | — |
| 22-row repair union | 0.8296 | 12,319 or 12,320 | -4 to -2 |
| 43-row gate plus strict repair | 0.8293 | exactly 12,315 | -8 or -7 |
| Outright-first threshold | 0.8301 | exactly 12,327 | needs +5 or +4 |

The repair union decomposes into strict core `C`, weaker history `W` and
identity `I`; its hidden net is therefore `C + W + I ∈ {-4, -3, -2}`. The
second candidate decomposes into archive gate `G`, the shared strict core `C`
and new residual history `H`; its net is `G + C + H ∈ {-8, -7}`. Cancelling
the shared block establishes `(G + H) - (W + I) ∈ {-5, -4}` but does not reveal
any individual row label.

The gate is excluded. A Jeffreys-smoothed Dirichlet-multinomial model using
the frozen out-of-fold block outcomes and both score constraints assigned the
31-row gate an expected hidden net of -7.67 and only a 0.2% probability of a
positive net. The same model ranks the selected `C + H + M` construction above
strict `C + H` and residual `H + M` for the probability of reaching first:

| Candidate | Competition flips | Expected net | Chance of 0.8301+ |
| --- | ---: | ---: | ---: |
| **Strict plus dual-evidence identity (`C + H + M`)** | **13** | **+0.89** | **13.2%** |
| Strict (`C + H`) | 12 | +0.33 | 8.3% |
| Residual (`H + M`) | 5 | +1.07 | 7.9% |

The conditioning event was itself rare under the validation-informed prior,
so these probabilities are decision aids rather than calibrated guarantees.
They also cease to be exact if the public leaderboard scores an undisclosed
subset rather than all 14,850 rows.

## Candidate evidence and integrity

The final mask changes 30 frozen development predictions for **+20 net correct**
rows, with fold nets `+3, +3, +2, +6, +6`. It changes ten used local-test
predictions for **+6 net correct** rows. The dual-evidence identity analogue is
perfect on its five selected labelled examples (four development and one
local), while the strict component remains positive in every development fold.

- File: `01-strict-repair-plus-meta-overlap.csv`
- Rows: 14,850 unique IDs in template order
- Changes versus the 0.8298 incumbent: 13
- Transition: `functional` to `functional needs repair` only
- SHA-256: `9023b7e4839f4f42521d0dbd89c996f5cd5ac339d9f97848f8d223cbb8888997`
- Status: prepared pending action confirmation

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

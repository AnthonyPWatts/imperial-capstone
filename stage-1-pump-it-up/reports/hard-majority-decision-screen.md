# Hard-majority decision screen

## Decision

Retain the promoted 44:36:20 soft vote. Taking the hard majority of XGBoost,
Random Forest and complete-identity CatBoost lowers mean accuracy from 81.7403%
to **81.5341%**, a **-0.2062 percentage-point** change with **0/5** fold wins.

No alternative tie-break or confidence threshold is screened. No local-test or
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether component agreement is a better decision rule than soft probability strength. |
| 2. Gather the data | Reuse aligned OOF probabilities from the three promoted components. |
| 3. Explore the data | Count majority and all-disagree rows without refitting. |
| 4. Clean and preprocess the data | Not applicable; cached components were previously fitted fold-safely. |
| 5. Select and engineer features | Not applicable; no representation change. |
| 6. Define the machine-learning task | Unchanged three-class classification and accuracy metric. |
| 7. Partition the data | Reuse the five frozen folds and untouched local test. |
| 8. Select and train candidate methods | No training; apply one deterministic hard-decision combiner. |
| 9. Evaluate and interpret the results | Compare hard consensus directly with the promoted soft vote. |
| 10. Deploy and iterate | Reject hard majority and retain probability magnitudes. |

## Fixed rule

- When at least two component argmax labels agree, select that label.
- When all three labels differ, select the promoted soft-vote argmax.

The components produce a hard majority for 47,360 of 47,520 development rows.
Only 160 rows require the all-disagree soft tie-break.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Promoted identity soft vote | **81.7403%** | — | — | — | — | incumbent |
| Hard majority with soft tie-break | 81.5341% | -0.2062 pp | 0/5 | -0.3683 pp | -1.7375 pp | No |

The soft vote uses confidence to resolve the common two-versus-one pattern.
Hard majority discards that information and lets two weaker, mildly confident
components override one strong probability. The result supports probability
averaging rather than additional hard-vote rules.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_decision_combiner_screen.py
```

Runtime evidence remains ignored under `.runtime/decision-combiner-screen/`.

# Cumulative XGBoost screen

## Decision

Retain the fixed arithmetic archive synthesis. Two cumulative binary XGBoost
boundaries reproduce the accepted multiclass XGBoost's **81.0143%** standalone
accuracy but do not improve its archive contribution materially. The best
fixed equal boundary bag reaches **81.8035%**, only 0.0105 percentage points
above the 81.7929% archive incumbent.

That change represents five development rows. Although it wins three folds,
its worst fold loses 0.0526 points and the total gain is far below the
predeclared 0.10-point promotion threshold. The branch therefore stops before
local-test confirmation or competition inference.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether the ordered relationship between functional, repair and non-functional states provides a genuinely new ensemble signal. |
| 2. Gather the data | Reuse the frozen 47,520-row development partition and existing archive-component OOF probabilities. |
| 3. Explore the data | Treat repair as the middle state and inspect boundary crossings without tuning thresholds. |
| 4. Clean and preprocess the data | Reuse the accepted fold-fitted one-hot preprocessing independently within every outer-training fold. |
| 5. Select and engineer features | Use the accepted predictors unchanged; the only new representation is cumulative target decomposition. |
| 6. Define the machine-learning task | Fit `repair or worse` versus functional and non-functional versus `functional or repair`, then reconstruct three memberships. |
| 7. Partition the data | Reuse the five frozen stratified folds, with early-stopping selection confined to each outer-training partition. |
| 8. Select and train candidate methods | Fit one accepted-spec depth-8, child-1 XGBoost model per boundary and fold. |
| 9. Evaluate and interpret the results | Compare standalone accuracy, a fixed archive substitution and an equal multiclass/cumulative boundary bag. |
| 10. Deploy and iterate | Stop before local-test or competition use because the best change misses the promotion gate by an order of magnitude. |

## Probability reconstruction

The fitted boundaries estimate:

$$
q_1 = P(\text{repair or non-functional}), \qquad
q_2 = P(\text{non-functional}).
$$

Coherence requires $q_2 \le q_1$. Where independently fitted boundaries cross,
both are projected to their midpoint. Class memberships are then:

$$
P(F)=1-q_1, \qquad P(R)=q_1-q_2, \qquad P(N)=q_2.
$$

Between 24.89% and 26.80% of validation rows require this projection. This is
substantial evidence that the two independently fitted boundaries do not
naturally behave as one ordinal distribution. Projection remains deterministic
and is tested for valid non-negative memberships summing to one.

## Results

| Candidate | Accuracy | Change vs archive | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing archive synthesis | 81.7929% | — | — | — | — |
| Accepted multiclass XGBoost | 81.0143% | — | — | — | — |
| Cumulative XGBoost | 81.0143% | — | — | — | — |
| Cumulative XGBoost archive substitution | 81.7677% | -0.0253 pp | 3/5 | -0.1157 pp | -0.1451 pp |
| Equal multiclass/cumulative archive bag | **81.8035%** | **+0.0105 pp** | 3/5 | -0.0526 pp | +0.0577 pp |

The cumulative model disagrees with accepted XGBoost on 2.90% of development
rows, with exactly balanced unique-correct shares of 1.2668%. It is therefore
different, but not stronger. Equal boundary averaging recovers five additional
archive predictions and slightly improves repair recall; this is useful model
diagnosis rather than sufficient evidence for promotion after extensive reuse
of the same folds.

No class weights, thresholds, ordinal spacing, boundary-specific features or
blend weights were screened. Avoiding those follow-ups prevents a five-row
effect from becoming another adaptive validation search.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_ordinal_xgboost_screen.py
```

Runtime evidence remains ignored under `.runtime/ordinal-xgboost-screen/`.

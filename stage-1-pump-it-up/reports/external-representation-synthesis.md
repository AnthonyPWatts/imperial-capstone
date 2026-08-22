# External-representation synthesis

## Decision

Retain the promoted 44:36:20 complete-identity vote. Equal averaging of the two
fixed archive-derived representation candidates reaches **81.7929%**, a
**+0.0526 percentage-point** change. It improves all **5/5** folds, with a
+0.0105-point worst-fold change, but remains below the predeclared +0.10-point
promotion gate.

Adding the strongest earlier feature near-miss, the fixed pump-age cohort,
reduces the result to 81.7698%. No further near-miss accumulation or blend-
weight search is justified. No local-test or competition prediction was
generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test once whether two independently motivated archive-derived representations make complementary corrections. |
| 2. Gather the data | Reuse aligned cached OOF probabilities from accepted, occurrence-count and spatial-height components. |
| 3. Explore the data | Build on two separately completed screens that each produced a small positive representation bag. |
| 4. Clean and preprocess the data | Not applicable; every cached component was already fitted inside the same frozen outer folds. |
| 5. Select and engineer features | Combine the existing all-categorical occurrence-count and ten-neighbour height representations without creating a new feature. |
| 6. Define the machine-learning task | Unchanged three-class classification and accuracy metric. |
| 7. Partition the data | Reuse identical OOF row order, class order and frozen-fold fingerprint; keep local test closed. |
| 8. Select and train candidate methods | No refit; equally average the two complete fixed candidate votes. |
| 9. Evaluate and interpret the results | Compare mean accuracy, all five fold changes, repair recall and changed hard decisions. |
| 10. Deploy and iterate | Reject promotion, stop broad near-miss accumulation and require a genuinely new signal for further model work. |

## Fixed algebraic recipe

The synthesis is the equal probability average of:

- the frequency-forest representation vote: 44% accepted XGBoost, 18%
  accepted Random Forest, 18% occurrence-count Random Forest and 20% identity
  CatBoost;
- the spatial-height representation vote: 22% accepted XGBoost, 22% spatial-
  height XGBoost, 18% accepted Random Forest, 18% spatial-height Random Forest
  and 20% identity CatBoost.

After collecting common components, the resulting weights are:

| Component | Weight |
| --- | ---: |
| Accepted XGBoost | 33% |
| Spatial-height XGBoost | 11% |
| Accepted Random Forest | 18% |
| Occurrence-count Random Forest | 9% |
| Spatial-height Random Forest | 9% |
| Complete-identity CatBoost | 20% |

These weights are an exact average of prior candidates, not a fit to OOF
labels.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Archive-derived representation synthesis | **81.7929%** | **+0.0526 pp** | **5/5** | **+0.0105 pp** | -0.2026 pp | No |
| Archive plus pump-age synthesis | 81.7698% | +0.0295 pp | 2/5 | -0.0105 pp | -0.3765 pp | No |
| Promoted complete-identity vote | 81.7403% | — | — | — | — | incumbent |

The archive synthesis changes only 0.6902% of the promoted vote's hard labels.
It uniquely corrects 0.3409% of all development rows and uniquely breaks 0.2883%,
which explains the net 0.0526-point gain. The effect is consistent but too
small to distinguish safely from adaptive validation selection after the large
number of experiments already viewed on these folds.

The pump-age follow-up is the only orthogonal near-miss accumulation tested. It
does not compound, so the branch closes without adding one-vs-rest boundaries,
seed bags, text models or tuned contributions.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_external_representation_synthesis.py
```

Runtime evidence remains ignored under
`.runtime/external-representation-synthesis/`.

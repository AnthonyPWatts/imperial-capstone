# Physical-state interaction screen

## Decision

Retain the existing archive synthesis without explicit physical-state
composites. Replacing complete-identity CatBoost with a model containing
`quantity` crossed with extraction type, source and waterpoint type reaches
**81.7908%**, one net development row below the **81.7929%** incumbent. It wins
three of five folds, but the mean change is -0.0021 percentage points and its
worst fold loses 0.1368 points.

An equal original/interaction CatBoost bag reaches 81.7845%. Neither candidate
passes the promotion gate, so the local test remains closed and no competition
prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether observed water quantity changes the meaning of pump, source and waterpoint types beyond their separate fields. |
| 2. Gather the data | Reuse supplied quantity, extraction type, source and waterpoint type values with cached archive components. |
| 3. Explore the data | Audit composite support without using labels to choose interaction subsets. |
| 4. Clean and preprocess the data | Reuse conservative identity normalisation and native CatBoost missing states. |
| 5. Select and engineer features | Add exactly quantity×extraction type, quantity×source and quantity×waterpoint type while retaining all constituents. |
| 6. Define the machine-learning task | Unchanged three-class accuracy. |
| 7. Partition the data | Keep CatBoost stopping and refitting inside each of the five frozen outer folds. |
| 8. Select and train candidate methods | Fit one depth-8 complete-identity CatBoost and make fixed 20% substitution and equal within-CatBoost bag comparisons. |
| 9. Evaluate and interpret the results | Compare direct and archive votes on accuracy, fold stability and repair recall. |
| 10. Deploy and iterate | Close the branch without local-test or competition inference because every full vote trails its incumbent. |

## Interaction contract

The three native categories are:

- `quantity × extraction_type`;
- `quantity × source`;
- `quantity × waterpoint_type`.

No interaction subset, CatBoost parameter or contribution was selected from
the results. All composites have strong development support:

| Composite | Levels | Singleton row share | Rows in groups ≥20 |
| --- | ---: | ---: | ---: |
| Quantity × extraction type | 77 | 0.0084% | 99.5960% |
| Quantity × source | 50 | 0% | 99.8380% |
| Quantity × waterpoint type | 32 | 0.0042% | 99.9158% |

The model selected 1,255–2,564 trees across the five folds. The interaction
layer therefore has adequate category and training support; its failure is not
an unseen-level or early-stopping artefact.

## Results

| Candidate | Accuracy | Change vs archive | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing archive synthesis | **81.7929%** | — | — | — | — |
| Physical-state CatBoost archive substitution | 81.7908% | -0.0021 pp | 3/5 | -0.1368 pp | -0.0580 pp |
| Equal CatBoost-subfamily archive bag | 81.7845% | -0.0084 pp | 1/5 | -0.0631 pp | +0.2026 pp |
| Promoted complete-identity vote | 81.7403% | -0.0526 pp | 0/5 | -0.0842 pp | +0.2026 pp |
| Equal direct CatBoost bag | 81.7361% | -0.0568 pp | 1/5 | -0.1157 pp | +0.0578 pp |
| Direct physical-state CatBoost vote | 81.7319% | -0.0610 pp | 1/5 | -0.1368 pp | +0.1157 pp |
| Physical-state interaction CatBoost | 80.5072% | -1.2858 pp | 0/5 | -1.9571 pp | -6.3993 pp |

The replacement's three fold wins do not compound: losses in the other two
folds erase them. Because the standalone CatBoost is also weaker than the
original complete-identity component, there is no evidence for a quantity,
interaction or contribution grid.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_physical_state_interaction_screen.py
```

Runtime evidence remains ignored under
`.runtime/physical-state-interaction-screen/`.

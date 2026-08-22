# Complete physical-hierarchy identity CatBoost

## Decision

Retain the promoted complete-identity CatBoost without physical hierarchy
back-offs. Adding all seven supplied parent and intermediate fields lowers the
promoted vote from 81.7403% to **81.7256%**, a **-0.0147 percentage-point**
change with **2/5** fold wins. Standalone CatBoost also weakens.

No hierarchy subsets or ensemble weights are tuned. No local-test or
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether full coarse-to-fine physical context helps native identity learning. |
| 2. Gather the data | Reuse all supplied extraction, management, quality, source and waterpoint hierarchy fields. |
| 3. Explore the data | Build on the completed hierarchy audits and prior four-parent result. |
| 4. Clean and preprocess the data | Apply conservative normalisation and explicit missing states. |
| 5. Select and engineer features | Add all seven deferred hierarchy fields as native CatBoost categoricals. |
| 6. Define the machine-learning task | Unchanged three-class classification. |
| 7. Partition the data | Reuse the same five frozen development folds. |
| 8. Select and train candidate methods | Refit only the promoted depth-8 identity CatBoost specification. |
| 9. Evaluate and interpret the results | Compare standalone CatBoost and its unchanged 20% ensemble contribution. |
| 10. Deploy and iterate | Reject the full back-off representation and close hierarchy subsets. |

## Representation

The fixed policy adds:

- `extraction_type_group` and `extraction_type_class`;
- `management_group`;
- `quality_group`;
- `source_type` and `source_class`;
- `waterpoint_type_group`.

Together with the 15 accepted categoricals and six deferred identities, this
produces 28 native categorical fields and 42 engineered features.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Promoted identity vote | **81.7403%** | — | — | — | — | incumbent |
| Full-hierarchy identity vote | 81.7256% | -0.0147 pp | 2/5 | -0.1263 pp | -0.0289 pp | No |
| Complete-identity CatBoost | 80.6229% | -1.1174 pp | 0/5 | -1.5888 pp | -5.8489 pp | No |
| Full-hierarchy identity CatBoost | 80.5997% | -1.1406 pp | 0/5 | -1.7466 pp | -5.9070 pp | No |

The deterministic parents repeat information already present in their granular
children. Native categorical handling does not convert that redundancy into a
stronger model, consistent with the earlier four-parent result.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_catboost_identity_full_hierarchy_screen.py
```

Runtime evidence remains ignored under
`.runtime/catboost-identity-full-hierarchy/`.

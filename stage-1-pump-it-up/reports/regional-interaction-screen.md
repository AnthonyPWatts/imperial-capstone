# Shared regional-interaction layer

## Decision

Retain the promoted complete-identity vote without explicit regional
composites. Adding three uniformly specified region-by-physical categories to
the identity CatBoost produces a small **81.7635%** direct-vote result, **+0.0231
percentage points** with 4/5 fold wins. Applying the same layer to all three
model families reaches only **81.7445%**, while XGBoost and Random Forest both
weaken individually.

No region, physical-field subset or ensemble weight is tuned. No local-test or
competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test the smallest pooled alternative to one independently fitted classifier per region. |
| 2. Gather the data | Reuse supplied region, extraction type, source and waterpoint type fields. |
| 3. Explore the data | Audit level counts, singleton exposure and support at the existing rare-category boundary. |
| 4. Clean and preprocess the data | Reuse conservative category normalisation and fold-fitted support-below-20 grouping for one-hot models. |
| 5. Select and engineer features | Add exactly region×extraction type, region×source and region×waterpoint type while retaining every constituent field. |
| 6. Define the machine-learning task | Unchanged pooled three-class classification and accuracy metric. |
| 7. Partition the data | Reuse the frozen five folds; preprocessing and model fitting stay inside each outer-training partition. |
| 8. Select and train candidate methods | Refit accepted XGBoost, Random Forest and complete-identity CatBoost specifications under one shared policy. |
| 9. Evaluate and interpret the results | Compare family-specific substitution, complete replacement and one equal original/layered representation bag. |
| 10. Deploy and iterate | Reject promotion and close explicit regional layering because the best gain is only 0.023 points. |

## Shared-layer contract

The three composites are fixed for every region and model. No region receives
its own learner, threshold, feature subset or validation decision. This keeps
all 47,520 development rows available to each global model and avoids missing-
class probability problems in small regional samples.

The composites are well supported in the full development audit:

| Composite | Levels | Singleton row share | Rows in groups ≥20 |
| --- | ---: | ---: | ---: |
| Region × extraction type | 239 | 0.040% | 98.824% |
| Region × source | 172 | 0.040% | 99.310% |
| Region × waterpoint type | 109 | 0.021% | 99.766% |

For XGBoost and Random Forest, support is re-estimated within each fold before
rare grouping. CatBoost receives the normalised composites natively beside the
six deferred identities.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Regional interactions in identity CatBoost | **81.7635%** | +0.0231 pp | 4/5 | -0.0105 pp | -0.1159 pp | No |
| Full regional-interaction vote | 81.7445% | +0.0042 pp | 2/5 | -0.0842 pp | -0.0002 pp | No |
| Promoted complete-identity vote | 81.7403% | — | — | — | — | incumbent |
| Equal original/layered representation bag | 81.7235% | -0.0168 pp | 1/5 | -0.0842 pp | -0.0291 pp | No |
| Regional interactions in global components | 81.6961% | -0.0442 pp | 1/5 | -0.1157 pp | -0.0871 pp | No |
| Regional-interaction 55:45 vote | 81.5236% | -0.2168 pp | 0/5 | -0.3788 pp | +1.2158 pp | No |
| Regional-interaction XGBoost | 81.0375% | +0.0231 pp vs original XGBoost | — | — | — | diagnostic |
| Regional-interaction Random Forest | 80.4714% | -0.1199 pp vs original forest | — | — | — | diagnostic |
| Regional-interaction CatBoost | 80.5303% | -0.0926 pp vs identity CatBoost | — | — | — | diagnostic |

XGBoost gains slightly alone, but combining both regional global components
loses accuracy because the forest weakens and the errors are not complementary.
The CatBoost substitution makes useful corrections in four folds despite a
weaker standalone score, but the mean gain is less than one quarter of the
promotion threshold. Explicit composites therefore do not justify regional
subsets, alternative interactions or a local-test use.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_regional_interaction_screen.py
```

Runtime evidence remains ignored under `.runtime/regional-interaction-screen/`.

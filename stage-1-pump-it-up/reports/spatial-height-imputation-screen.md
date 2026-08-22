# Spatial GPS-height imputation screen

## Decision

Retain the promoted 44:36:20 complete-identity vote. Ten-neighbour spatial
reconstruction makes both accepted tree components slightly stronger in
isolation, but direct replacement weakens the ensemble. One fixed equal bag of
original and imputed representations reaches **81.7677%**, a **+0.0274
percentage-point** change with **4/5** fold wins and a -0.0526-point worst fold.
This remains below the +0.10-point promotion gate.

No neighbour count, distance rule, component weight or model parameter is
tuned. No local-test or competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test learned reconstruction of missing GPS height found in the official competition solution archive. |
| 2. Gather the data | Reuse supplied coordinates and height measurements inside frozen development folds. |
| 3. Explore the data | Confirm that 16,376 development heights are unavailable and that 14,938 retain valid coordinates. |
| 4. Clean and preprocess the data | Preserve the accepted coordinate envelope, numeric policy and original height-missing flag. |
| 5. Select and engineer features | Fill unavailable height from ten distance-weighted measured neighbours; use the training median only for invalid coordinates. |
| 6. Define the machine-learning task | Unchanged three-class classification and accuracy metric. |
| 7. Partition the data | Fit each neighbour index only on the current outer-training partition; leave every validation row unchanged for scoring. |
| 8. Select and train candidate methods | Refit accepted XGBoost and Random Forest once, test direct and component-cross replacements, then one equal representation bag. |
| 9. Evaluate and interpret the results | Compare accuracy, fold stability and repair recall with the promoted vote. |
| 10. Deploy and iterate | Reject promotion, retain the near-miss evidence and test whether independent archive-derived representations compound. |

## External lead and fixed adaptation

DrivenData's [official Pump It Up solution archive](https://github.com/drivendataorg/pump-it-up)
records a 0.8243 public score for `mrbeer`. Its
[solution notes and code](https://github.com/drivendataorg/pump-it-up/tree/master/mrbeer)
use a separate XGBoost regression stage to reconstruct zero GPS height before
classification. The archived code concatenates labelled and competition
feature frames and factorises all strings. This experiment instead uses a
smaller and auditable geographic hypothesis: height changes smoothly between
nearby measured points.

The transformer indexes only non-zero training-partition heights with valid
coordinates. Missing rows with coordinates receive an inverse-distance mean of
the ten nearest measurements. Rows outside the Tanzania coordinate envelope
receive the training-partition median. The accepted `gps_height_missing` flag
remains one, so the classifier can distinguish measured from reconstructed
values.

## Results

| Candidate | Accuracy | Leader change | Fold wins | Worst fold | Repair-recall change | Passes gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Equal original/spatial representation bag | **81.7677%** | **+0.0274 pp** | 4/5 | -0.0526 pp | +0.0869 pp | No |
| Promoted complete-identity vote | 81.7403% | — | — | — | — | incumbent |
| Spatial-height XGBoost cross | 81.7256% | -0.0147 pp | 2/5 | -0.1157 pp | -0.0291 pp | No |
| Spatial-height Random Forest cross | 81.7193% | -0.0210 pp | 2/5 | -0.1263 pp | -0.2605 pp | No |
| Full spatial-height replacement | 81.7151% | -0.0253 pp | 2/5 | -0.1263 pp | -0.2027 pp | No |
| Spatial-height XGBoost | 81.0501% | — | — | — | — | diagnostic |
| Accepted XGBoost | 81.0143% | — | — | — | — | diagnostic |
| Spatial-height Random Forest | 80.6145% | — | — | — | — | diagnostic |
| Accepted Random Forest | 80.5913% | — | — | — | — | diagnostic |

Both imputed components improve slightly: +0.0358 points for XGBoost and
+0.0231 points for Random Forest. Their changed probability boundaries do not
replace the originals cleanly, but equal averaging inside the existing 44% and
36% allocations supplies a small, stable gain. Its magnitude does not justify
neighbour-count or blend-weight tuning on the repeatedly used folds.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_spatial_height_imputation_screen.py
```

Runtime evidence remains ignored under
`.runtime/spatial-height-imputation-screen/`.

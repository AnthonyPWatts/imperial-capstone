# Radial frequency-forest screen

## Decision

Retain the existing archive synthesis. Adding the archived geodesic distance
from `(0, 0)` improves the standalone occurrence-count Random Forest from
**80.6103%** to **80.7471%**, a useful **+0.1368 percentage-point** component
gain. It does not improve the complete fixed ensemble: direct substitution
reaches 81.7908%, 0.0021 points below the 81.7929% incumbent.

An equal split of the existing 9% frequency allocation between the original
and radial forests reaches 81.7950%. That is one net development row, one of
five fold wins and far below the +0.10-point gate. The local test remains
closed and no competition prediction was generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Complete the archived occurrence-count representation with its omitted radial coordinate and seek independent ensemble corrections. |
| 2. Gather the data | Reuse supplied coordinates, fixed development folds and cached promoted/archive components. |
| 3. Explore the data | Use one externally documented radial definition; do not screen origins, projections or distance families. |
| 4. Clean and preprocess the data | Apply the accepted Tanzania coordinate envelope; invalid pairs become missing and are median-imputed inside each training fold with an indicator. |
| 5. Select and engineer features | Add only great-circle distance in kilometres from `(0, 0)` to the existing all-categorical occurrence-count frame. |
| 6. Define the machine-learning task | Unchanged three-class classification and accuracy objective. |
| 7. Partition the data | Learn occurrence counts, medians and every forest strictly inside each frozen outer-training fold. |
| 8. Select and train candidate methods | Fit one accepted-spec Random Forest, make one direct archive substitution, then one equal within-frequency-family bag. |
| 9. Evaluate and interpret the results | Compare accuracy, fold stability, repair recall and prediction disagreement with the existing archive synthesis. |
| 10. Deploy and iterate | Close the branch without local-test or competition inference because neither complete vote passes the gate. |

## Feature definition

The external reference is DrivenData's official solution archive and the
[`madRid` second-place code](https://github.com/drivendataorg/pump-it-up/blob/master/madRid/PumpItUp_DataDriven_2nd_place_madRid_team.R).
That implementation combines categorical occurrence counts with geodesic
distance from `(0, 0)`. The adapted feature is deterministic and target-free.
Unlike the archived script, it does not treat the `(0, -2e-08)` missing-location
sentinel as a real point.

For valid longitude $\lambda$ and latitude $\phi$ in radians, distance is:

$$
d = 2R\arcsin\left(\sqrt{\sin^2(\phi/2) +
\cos(\phi)\sin^2(\lambda/2)}\right),
$$

where $R=6371.0088$ km. The radial frame has 44 engineered features and 49
transformed columns in fold one.

## Results

| Candidate | Accuracy | Change vs archive | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal frequency-subfamily archive bag | **81.7950%** | **+0.0021 pp** | 1/5 | -0.0210 pp | +0.0578 pp |
| Existing archive synthesis | 81.7929% | — | — | — | — |
| Radial archive substitution | 81.7908% | -0.0021 pp | 1/5 | -0.0210 pp | +0.0868 pp |
| Radial frequency-forest representation bag | 81.7803% | -0.0126 pp | 1/5 | -0.0737 pp | -0.4345 pp |
| Promoted identity vote | 81.7403% | -0.0526 pp | 0/5 | -0.0842 pp | +0.2026 pp |
| Radial frequency forest | 80.7471% | -1.0459 pp | 0/5 | -1.2416 pp | +2.1421 pp |
| Original frequency forest | 80.6103% | -1.1827 pp | 0/5 | -1.5152 pp | +1.9973 pp |

The radial and original frequency forests disagree on 2.5337% of development
rows. Radial alone is correct on 1.1995% and original alone on 1.0627%, so the
new feature improves that component rather than merely changing its class
balance. The ensemble substitution changes only 0.2462% of archive decisions,
and the gains and losses cancel almost exactly.

This is good component engineering but not a new submission. The evidence also
argues against searching alternative origins or tuning the frequency allocation
on folds that have already supported many experiments.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_radial_frequency_forest_screen.py
```

Runtime evidence remains ignored under
`.runtime/radial-frequency-forest-screen/`.

# Spatial-density screen

## Decision

Retain the archived deep-XGBoost near-miss without waterpoint-density features.
Adding fold-fitted distances to the nearest and 10th-nearest training
waterpoints reduces deep XGBoost from **81.4710%** to **81.2963%**. Its fixed
archive substitution reaches **81.8182%**, only 0.0253 percentage points above
the archive incumbent and 0.0589 points below the existing deep substitution.

The candidate wins three folds against the archive but loses its worst fold by
0.2104 points and reduces repair recall by 0.4635 points. It fails the
development gate, so no local-test or competition prediction is generated.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Test whether target-free waterpoint isolation and settlement density add a new supplied-data signal. |
| 2. Gather the data | Reuse only supplied coordinates, frozen development rows and cached archive components. |
| 3. Explore the data | Distinguish continuous neighbourhood density from previously rejected coordinate grids, centroids and target-based spatial outcomes. |
| 4. Clean and preprocess the data | Apply the accepted Tanzania coordinate envelope and training-fold median fallback for invalid coordinate pairs. |
| 5. Select and engineer features | Add exactly distance to the nearest other fitted waterpoint and to the 10th nearest; do not open a rank or radius grid. |
| 6. Define the machine-learning task | Retain unchanged three-class probabilistic classification and accuracy. |
| 7. Partition the data | Fit the spatial reference separately inside each outer-training fold and exclude every fitting row from its own neighbourhood. |
| 8. Select and train candidate methods | Refit one fixed seed of archived depth-17, 600-tree XGBoost over accepted, top-50 identity and density features. |
| 9. Evaluate and interpret the results | Compare component strength, diversity and one fixed archive substitution with the density-free deep candidate. |
| 10. Deploy and iterate | Stop before local or competition use because both the component and ensemble weaken materially. |

## Fold-safe density contract

Coordinates are converted to radians and queried with a haversine `BallTree`.
During model fitting, each row's own zero-distance match is removed before
selecting neighbour ranks. Validation and later inference rows query only the
fitted training reference. This avoids the train/prediction mismatch that
would otherwise make every training nearest-neighbour distance zero.

Invalid coordinate pairs receive the training partition's median distance at
each rank. The accepted `coordinates_missing` feature remains available, so
the model can distinguish fallback values from measured neighbourhoods. The
implementation tests self-exclusion, invalid-coordinate fallback, finite
values and stable transformed dimensions.

## Results

| Candidate | Accuracy | Change vs archive | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing archived deep substitution | **81.8771%** | +0.0842 pp | 3/5 | -0.0210 pp | +0.6945 pp |
| Spatial-density deep substitution | 81.8182% | +0.0253 pp | 3/5 | -0.2104 pp | -0.4635 pp |
| Existing archive synthesis | 81.7929% | — | — | — | — |
| Archived deep XGBoost | **81.4710%** | — | — | — | — |
| Spatial-density deep XGBoost | 81.2963% | — | — | — | — |

The density component disagrees with the density-free deep model on 4.2551% of
development rows. The original model is uniquely correct on 2.0118%, versus
1.8371% for density XGBoost. Density is therefore not merely redundant: it
makes a substantial number of changed but less accurate decisions.

This closes nearest-neighbour spatial-density work. Selecting other ranks,
radius counts, distance transforms, clustering methods or transductive
competition references after a clear component loss would create an adaptive
geographic grid rather than test a supported hypothesis.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_spatial_density_screen.py
```

Runtime evidence remains ignored under `.runtime/spatial-density-screen/`.

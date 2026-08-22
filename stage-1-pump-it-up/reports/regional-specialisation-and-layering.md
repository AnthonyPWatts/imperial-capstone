# Regional specialisation and layering

## Conclusion

Region matters, but giving regions separate classifiers does not improve the
competition model. The fixed partially pooled candidate reached **81.469%**
mean development-fold accuracy, **0.156 percentage points below** the accepted
global vote, and lost on all five folds. Hard routing to regional classifiers
lost 1.120 points.

The accepted XGBoost and Random Forest already receive region, region code,
district, LGA, basin and coordinates. They can learn region-conditioned
interactions while retaining evidence from all regions. The regional layer
mostly removes that useful pooling.

No candidate passed the existing promotion gate. The local test remained
closed and no competition prediction was generated.

## What “more layers” means here

Several different ideas are often described as layering:

- adding depth within a tree or hidden layers within a neural network;
- stacking one model's probabilities into another model;
- hierarchical classification, where one decision precedes another;
- a mixture of experts, where a routing rule selects or weights specialist
  models.

“One classifier per region” is the last of these: a hard-gated mixture of
experts. It is not intrinsically deeper, and it is only helpful when the
specialists learn enough region-specific structure to compensate for the data
discarded by segmentation.

## Course-aligned lifecycle

| Step | Application in this experiment |
| --- | --- |
| 1. Define the goal and scope | Test whether regional specialisation improves frozen-fold accuracy beyond a global geography-aware ensemble. |
| 2. Gather the data | Reuse the validated 59,400 labelled rows; the 14,850 competition rows are not needed for selection. |
| 3. Explore the data | Compare regional support, class shares, global-model probabilities and regional OOF performance. |
| 4. Clean and preprocess the data | Reuse the accepted fold-fitted preprocessing separately inside every eligible regional expert. |
| 5. Select and engineer features | Retain the accepted 29-feature policy; region is the router as well as an existing global predictor. |
| 6. Define the machine-learning task | Three-class classification under nominal accuracy. |
| 7. Partition the data | Reuse the untouched local test and five frozen development folds. |
| 8. Select and train candidate methods | Test regional prior layers, one shared-spec Random Forest expert per eligible region, hard routing and a fixed 80:20 partial pool. |
| 9. Evaluate and interpret the results | Apply the existing accuracy gate and inspect recall, probability quality, routing coverage and region-level results. |
| 10. Deploy and iterate | Not applicable: no candidate passed the gate, so no refit, local-test evaluation or submission is justified. |

## Why the hypothesis was plausible

The regional target distributions are materially different. On the development
partition, 21.77% of Kigoma rows need repair, compared with 0.32% in Dar es
Salaam. Regional accuracy also ranges widely: the accepted vote reaches 71.06%
in Kigoma and 90.05% in Dar es Salaam.

However, the accepted probabilities already reproduce those regional repair
rates closely:

| Region | Observed repair share | Accepted mean repair probability | Accepted accuracy | Accepted repair recall |
| --- | ---: | ---: | ---: | ---: |
| Kigoma | 21.77% | 21.82% | 71.06% | 51.02% |
| Dar es Salaam | 0.32% | 0.38% | 90.05% | 0.00% |

The low hard-label repair frequency in Dar es Salaam is therefore mostly an
argmax consequence, not evidence that the global model missed the regional
base rate.

## Experimental contract

The expert policy was deliberately uniform rather than tuned by region:

- fit each expert only on its region's outer-training rows;
- use the existing 300-tree Random Forest recipe and fold-fitted preprocessing;
- require at least 500 regional training rows and 20 rows from every class;
- fall back to the accepted global probabilities when support is insufficient;
- align all expert probability columns to the three canonical labels;
- test hard routing as a stress diagnostic;
- make only one selection-eligible candidate: 80% global probabilities plus
  20% routed regional probabilities.

Twenty regions were eligible in every fold. Dar es Salaam always fell back:
its outer-training sets contained 487–507 rows and one fold contained no repair
example. Overall expert coverage was 98.69%. Several regional fits also had
entirely missing numeric features, causing the existing imputer to drop those
columns for that expert. That is valid fold-local behaviour, but another
concrete cost of fragmenting the data.

Two prior-ratio layers provide a cheap check before interpreting the fitted
experts. They multiply the accepted probabilities by regional-to-national
class-prior ratios estimated from outer-training rows. The first is
unsmoothed; the second adds 10,000 national-prior pseudo-observations. They are
diagnostics, not extra selection candidates.

## Results

The accepted baseline is 81.6246% accuracy, 34.8588% repair recall, log loss
0.46851 and multiclass Brier score 0.26669.

| Candidate | Selection candidate | Accuracy | Change | Fold wins | Worst fold | Repair-recall change | Log-loss change |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accepted global vote | Baseline | **81.6246%** | — | — | — | — | — |
| Prior layer, smoothing 10,000 | No | 81.5046% | -0.1199 pp | 0/5 | -0.1578 pp | +1.7369 pp | +0.00091 |
| 80% global + 20% regional RF | **Yes** | 81.4689% | -0.1557 pp | 0/5 | -0.2210 pp | +0.3764 pp | +0.00208 |
| Hard-routed regional RF | No | 80.5051% | -1.1195 pp | 0/5 | -1.2521 pp | +2.1132 pp | +0.17822 |
| Unsmoothed prior layer | No | 80.4188% | -1.2058 pp | 0/5 | -1.3889 pp | +5.8195 pp | +0.02212 |

The paired row-bootstrap 95% interval for the 80:20 accuracy change is
-0.2420 to -0.0673 percentage points. This secondary interval treats rows as
independent and is not a substitute for the fold gate, but it agrees with the
five fold losses.

The result also demonstrates why recall must not be confused with the stated
objective. Both hard routing and prior correction predict more repair cases,
but every accuracy comparison is worse. Unsurprisingly, the unsmoothed prior
layer double-counts regional information most aggressively.

## Decision and sensible next move

Retain the global two-component ensemble. Do not fit separately tuned XGBoost
or XGBoost/Random Forest ensembles for all 21 regions: they would multiply the
number of choices while some regions have little or no minority-class support.

If regional interactions are revisited, the smaller next experiment is a tiny
predeclared set of `region × physical feature` composites. That preserves
partial pooling and allows the global trees to decide whether a regional
interaction is useful. A genuine shared-trunk/per-region-head neural model is
conceptually cleaner than isolated classifiers, but the existing MLP evidence
is weak and does not justify that complexity yet.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_regional_specialisation_screen.py --force
```

The ignored runtime directory contains the candidate, routing and per-region
CSV summaries plus the serialised evaluations. Stable logic is in
`src/regional_specialisation_evaluation.py`.

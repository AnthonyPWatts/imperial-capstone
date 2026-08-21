# Training-label provenance and semantics

## Decision

For the DrivenData task, treat `status_group` as a **flat nominal three-class target**. It is a survey-time administrative condition label, not a timeless pump property, future-failure outcome, engineering diagnosis or rehabilitation-priority score.

Two binary trees remain useful, explicit modelling hypotheses:

1. **Tree A — operational semantics:** operational versus non-operational, then repair-needed versus no-repair within the operational branch.
2. **Tree B — maintenance decision:** intervention-needed versus no-intervention, then non-operational versus repair-needed within the intervention branch.

Tree A follows DrivenData's published wording. Tree B represents an action-oriented maintenance decision. Neither is the competition's scoring ontology: classification accuracy gives every wrong class the same cost. A simple order from `functional` to `functional needs repair` to `non functional` is plausible but remains a hypothesis; the source evidence does not establish a single latent severity scale, and certainly not equal spacing.

The most faithful account of the historical source process is **multiple latent axes collapsed into one label**: current service, hardware condition, repair need, breakdown duration, water quantity/seasonality, water quality and reporting judgement.

## What this investigation changed

This is a useful negative result, not a null result.

1. **It bounded the provenance claim.** The 74,250-row total, class shares, fields, dates, IDs and collector identity make the February 2014 WPMS origin compelling, but no row-level build manifest or exact seven-to-three-class transformation has been found. The project can now state that distinction cleanly instead of presenting inference as fact.
2. **It corrected the meaning of the label.** `status_group` is a historic administrative snapshot assembled from observation and local reporting. It is not a failure-time outcome, engineering diagnosis, current condition or direct rehabilitation priority.
3. **It tested the attractive hierarchy claims.** Tree A improves repair recall, but neither binary tree beats flat nominal accuracy with the controlled learner. Their value is an explicit change of decision costs, not discovery of the hidden labelling algorithm.
4. **It tested the ordinal shortcut properly.** Two cumulative targets, probability-coherence correction and nested cut-off calibration do not produce an accuracy win. The ordinal construction reduces severe two-step mistakes, but 28.8% of the independently estimated cumulative pairs cross and calibrated accuracy is still below flat multiclass.
5. **It established the modelling decision.** Keep flat multiclass as the competition contract. Retain oversampling, class weighting, trees or ordinal decisions only as named experiments whose minority-recall or cost trade-offs are reported explicitly.

The investigation therefore closes off the seductive but unsupported route of numerically encoding the classes, adjusting two thresholds against a holdout and claiming the target has been solved. A materially stronger ordinal claim would need repeated elemental observations—flow now, fault, downtime, repairability and repair outcome—or a stronger cumulative learner that wins under untouched evaluation without threshold leakage.

## Provenance: fact, inference and unknown

### Documented facts

- [DrivenData says](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/24/) the competition data came from the Taarifa waterpoints dashboard, which aggregated Tanzanian Ministry of Water data. This describes the publication route; it does not show that the original baseline labels were crowdsourced through Taarifa.
- The [World Bank's 2016 completion report](https://documents1.worldbank.org/curated/en/919091467777086225/pdf/ICR3737-P087154-Box396252B-PUBLIC-disclosed-7-1-16.pdf) says the Ministry engaged a local firm in 2011. The firm geotagged and collected basic data from about 75,000 water points during 2012–2013.
- The [Tanzania Ministry of Water's 2013 sector report](https://www.maji.go.tz/uploads/publications/en1568462815-2013_Water_Sector_Status_Report.pdf) reports 75,777 points mapped by June 2013. The system was intended for coverage reporting, planning, monitoring and maintenance decisions.
- A later [World Bank completion report](https://documents1.worldbank.org/curated/en/099042423190529784/pdf/P16175705053eb050ae2907c76d3ea463f.pdf) describes the 2013 national exercise as 74,250 water points: 55% functional, 7% needing repair and 38% non-functional.
- Verplanke and Georgiadou's peer-reviewed [Wicked Water Points](https://doi.org/10.3390/ijgi6080244) reports that a water-point collector recorded `STATUS` and that the February 2014 source release contained 74,250 points after accounting for an omitted region. It documents duplicate, identifier, observer, definition and data-processing problems.
- The same paper records seven `STATUS` values: functional; functional needing repair; four overlapping duration-banded non-functional variants; and an unqualified non-functional value. A separate consultant-derived `STATUS2` collapsed these into functional — including repair-needed — versus non-functional.
- The companion paper [Tensions in Rural Water Governance](https://doi.org/10.3390/ijgi6090266) says the 2010–2013 baseline used a conditional classification programme. Hardware problems and some short-duration non-functionality could be classed as needing repair. It also reports that local officials sometimes treated short interruptions as functional.
- [DrivenData's competition contract](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/) is simpler: `functional` is operational with no repair needed; `functional needs repair` is operational but needs repair; `non functional` is not operational.

### Strong local inference

The complete competition extract contains exactly 74,250 rows: 59,400 labelled and 14,850 unlabelled. Its combined IDs are exactly the integers 0–74,249, its features correspond closely to the source field form, all labelled rows name `GeoData Consultants Ltd` as `recorded_by`, and the labelled partition's class shares are 54.31%, 7.27% and 38.42%. This is compelling triangulation that the competition is an 80/20 split of the cleaned February 2014 WPMS release.

It remains an inference. No cited source provides a row-level crosswalk, build manifest or explicit statement that maps the competition files to that release.

The three-class target probably retains or cleans the source `STATUS` field rather than using binary `STATUS2`. The names and shares strongly support that reading, but the exact rule used to turn seven historical values into three `status_group` values is not published.

### Unknown

- The exact source-release file and transformation script used by DrivenData/Taarifa.
- Which duration-banded `STATUS` values, if any, were merged into the competition's repair-needed class.
- Whether individual records were re-inspected, manually corrected or excluded before the competition split.
- Whether the 31 rows dated 2002 or 2004 are carried-forward source dates or errors.
- Whether exact predictor duplicates represent duplicated records or genuinely distinct, observationally indistinguishable points.

## What the label meant at collection

The source evidence supports a **reference-epoch observation**:

- the unit was a mapped rural public water point;
- a water-point collector recorded the status during the baseline campaign, informed by direct observation and local sources such as district engineers, village officials, water committees and users;
- the row's `date_recorded` is the documented entry date and the best available
  reference-date proxy, overwhelmingly 2011–2013; the observation-to-entry lag
  is not documented;
- the construct supported administrative coverage and maintenance reporting;
- the label could combine current service, hardware trouble and how long the point had been out of service.

It does not establish that water physically flowed at the instant of collection. It is not a verified fault diagnosis, repair quote, repair outcome, failure date, survival target or present-day state.

This distinction matters because the later competition gloss makes `functional needs repair` sound strictly operational, while the historical programme could place recently non-functional points in that class. The competition label is authoritative for scoring; the source history is authoritative for understanding its limitations.

## Empirical checks

All analysis is aggregate. No competition rows or source identifiers are reproduced here.

### Class balance

| Class | Rows | Share |
|---|---:|---:|
| `functional` | 32,259 | 54.31% |
| `functional needs repair` | 4,317 | 7.27% |
| `non functional` | 22,824 | 38.42% |

The close match to the World Bank's 55% / 7% / 38% national summary strengthens the provenance inference. It also makes repair-needed a materially under-represented class.

### Time profile

| Year | Rows |
|---:|---:|
| 2002 | 1 |
| 2004 | 30 |
| 2011 | 28,674 |
| 2012 | 6,424 |
| 2013 | 24,271 |

The labels are therefore historic snapshots with different as-of dates. They cannot be read as current condition without a new observation and an explicit contemporary rubric.

### Same-visit feature relationships

Bias-corrected Cramér's V gives a descriptive, not causal, view of categorical association with `status_group`:

| Feature | Cramér's V |
|---|---:|
| `quantity` | 0.309 |
| `extraction_type_class` | 0.241 |
| `waterpoint_type_group` | 0.227 |
| `region` | 0.200 |
| `payment_type` | 0.182 |
| `water_quality` | 0.138 |
| `source_class` | 0.070 |
| `management_group` | 0.049 |

`quantity` is strongly related to the label but is not a mechanical restatement of it. There are 194 rows whose recorded quantity is `dry` but whose class is one of the two operational-labelled classes, and 9,138 rows with `enough` quantity but a non-functional label. These are not automatically errors: quantity may refer to source availability rather than delivered service, and the source audit describes quantity as subjective and contextual.

For competition reproduction, same-visit fields are legitimate supplied predictors. For a model intended to decide which sites to inspect, they may be unavailable at prediction time or may reproduce the collector's judgement. That intended-use boundary must be defined before feature selection.

### Indistinguishable rows

Excluding the competition ID, 72 labelled rows form 35 groups that are exact matches across all 39 supplied predictors. One group contains conflicting labels. A deterministic model using only these predictors cannot classify every member of that group correctly.

This is direct evidence of some irreducible ambiguity. It does not identify which label is wrong, and it is not grounds for relabelling either row.

## Target structures

### Flat nominal

This is the correct default for the competition. The submission is one of three named classes, and the public metric is multiclass accuracy. Nominal treatment does not claim the concepts are unrelated; it avoids inventing distances or a decision hierarchy.

### Ordinal hypothesis

The intuitive ranking is:

`functional` < `functional needs repair` < `non functional`

This may be useful for a proper ordinal classifier with cumulative thresholds. It should not be encoded as 0/1/2 and passed to ordinary regression as though adjacent distances were equal. The source programme mixed multiple dimensions, and a long-term dry or abandoned point is not simply one scalar unit beyond a cheap repair.

The executable probe now trains the two proper cumulative binary targets:

- $P(Y \geq 1)$: repair-needed or non-functional versus functional;
- $P(Y \geq 2)$: non-functional versus functional or repair-needed.

Independent cumulative estimates crossed on 28.8% of development rows before coherence projection. Symmetric projection produced valid three-class probabilities. Cut-offs were also tuned by nominal accuracy on a rotating calibration fold nested inside each outer training partition; no evaluated row entered model fitting or cut-off selection.

### Tree A: operational first

```text
Operational?
├─ no  → non functional
└─ yes → Repairs needed?
          ├─ no  → functional
          └─ yes → functional needs repair
```

This follows DrivenData's documented semantics. The historical source process weakens the first split because repair-needed could include short interruptions.

### Tree B: intervention first

```text
Intervention needed?
├─ no  → functional
└─ yes → Operational?
          ├─ yes → functional needs repair
          └─ no  → non functional
```

This follows a maintenance decision. It assumes both repair-needed and non-functional points warrant intervention, but the label contains no repair feasibility, cost, cause, urgency, alternative supply or beneficiary impact. It is therefore not a complete rehabilitation policy.

### Multiple latent axes

This is the strongest source-level interpretation. A better future label system would store elemental observations — water flowing now, flow duration, quantity, quality, fault type, downtime, repair requirement and repairability — then derive a versioned class for each decision. That would make rule changes auditable and separate physical condition from administrative action.

## Frozen-fold structural probe

The notebook reuses the existing validated 36-predictor handoff, untouched local-test boundary, five frozen development folds, fold-fitted preprocessing and constrained depth-12 decision tree. This is a deliberately modest separability probe, not a final model comparison.

### Binary tasks

| Task | Evaluated rows | Positive share | Balanced accuracy | ROC AUC | Negative recall | Positive recall |
|---|---:|---:|---:|---:|---:|---:|
| Tree A root: operational-labelled positive | 47,520 | 61.6% | 0.769 | 0.858 | 0.610 | 0.928 |
| Tree A branch: repair positive within operational-labelled | 29,261 | 11.8% | 0.622 | 0.792 | 0.979 | 0.265 |
| Tree B root: intervention positive | 47,520 | 45.7% | 0.753 | 0.835 | 0.892 | 0.613 |
| Tree B branch: non-functional positive within intervention-labelled | 21,713 | 84.1% | 0.716 | 0.878 | 0.469 | 0.963 |
| Direct pair: non-functional positive versus functional | 44,066 | 41.4% | 0.780 | 0.866 | 0.906 | 0.654 |

The extreme functional/non-functional pair is easiest. Repair/no-repair within operational-labelled rows is much harder, with only 26.5% repair recall at the default threshold. That is consistent with class imbalance, a fuzzy construct, missing fault variables or all three; the diagnostic cannot separate those explanations.

### Composed three-class predictions

| Structure | Accuracy | Macro recall | Functional recall | Repair recall | Non-functional recall |
|---|---:|---:|---:|---:|---:|
| Flat multiclass | 0.750 | 0.566 | 0.908 | 0.152 | 0.640 |
| Tree A: operational first | 0.748 | 0.588 | 0.914 | 0.240 | 0.610 |
| Tree B: intervention first | 0.748 | 0.574 | 0.892 | 0.178 | 0.651 |
| Ordinal: reconstructed probability argmax | 0.742 | 0.573 | 0.908 | 0.198 | 0.611 |
| Ordinal: two fixed 0.5 cut-offs | 0.724 | 0.578 | 0.882 | 0.265 | 0.588 |
| Ordinal: nested calibrated cut-offs | 0.745 | 0.550 | 0.925 | 0.115 | 0.610 |

With this fixed learner, flat multiclass wins accuracy by about 0.15–0.21 percentage points, exactly the objective the competition rewards. Tree A improves repair recall by 8.9 percentage points and macro recall by 2.2 points while reducing non-functional recall. Tree B makes a different trade-off.

The results justify keeping structural models as explicit candidates when recall or action costs matter. They do not prove either tree generated the labels, and a stronger learner could change the numeric ordering.

The ordinal result is more nuanced than a nominal win/loss. Reconstructed-probability argmax reduces mean absolute ordinal error from 0.430 to 0.420 and severe two-step errors from 17.9% to 16.3%, but loses 0.8 percentage points of accuracy. Fixed cut-offs trade more accuracy for repair recall. Nested accuracy calibration selects repair cut-offs of 0.75–0.80 and recovers accuracy to 0.745 by almost abandoning repair predictions: repair recall falls to 11.5%.

This does not disprove every ordinal learner. It does reject the simple claim that this target is so strongly one-dimensional that two cumulative thresholds plus honest tuning yield an easy competition win. The large crossing rate, mixed source semantics and weak repair discrimination all point in the same direction.

## Label-noise and proxy risks

1. **Snapshot staleness:** repair, breakdown, seasonality and abandonment can change status after observation.
2. **Definition drift:** later Tanzanian reporting programmes used revised rules; an accurate historic model may disagree with a current inspector.
3. **Observer/informant discretion:** the source research documents ambiguous definitions and local reporting incentives.
4. **Same-visit proxies:** quantity and quality can partly reproduce the historic reporter's reasoning but may not be available before an inspection.
5. **Geographic and institutional proxies:** location, management, technology and installer may encode both physical environments and data-collection practice.
6. **Missing action variables:** no verified fault diagnosis, repair cost, parts availability, repair success, time since failure or beneficiary-harm measure is present.
7. **Random-fold scope:** frozen random folds estimate interpolation within the same historic national extract, not temporal transfer, geographic transfer or present-day performance.
8. **Equal error cost:** competition accuracy treats all confusions equally; maintenance operations generally should not.

## Ten-step lifecycle disposition

| Step | Disposition |
|---:|---|
| 1. Define the goal and scope | Completed for target provenance, semantics and structure. |
| 2. Gather the data | Competition files plus public authoritative/peer-reviewed provenance sources; no external source entered modelling. |
| 3. Explore the data | Completed for counts, dates, source fingerprints, proxy relationships and exact duplicates. |
| 4. Clean and preprocess the data | Reused the validated modelling handoff; no relabelling. |
| 5. Select and engineer features | Reused the frozen 36-feature policy for diagnostics; no production feature decision. |
| 6. Define the machine-learning task | Compared nominal, ordinal, both binary trees and multiple latent axes. |
| 7. Partition the data | Reused the frozen development folds; local test untouched. |
| 8. Select and train candidate methods | Completed bounded flat, tree and proper cumulative-threshold ordinal probes. |
| 9. Evaluate and interpret the results | Recorded fold discrimination, nominal accuracy, class recall, macro recall, ordinal-distance errors and nested cut-off calibration. |
| 10. Deploy and iterate | Deployment not applicable. Next real-world iteration needs timestamped repeated labels under a current, versioned rubric. |

## Modelling implications

- Keep the competition target unchanged and compare candidates on accuracy plus repair recall, non-functional recall, macro recall and confusion matrices.
- Apply any oversampling or class weighting only within each training fold; leave validation and test distributions natural.
- Keep Tree A and Tree B as named experimental objectives so their different decision meanings are visible.
- Treat an ordinal encoding as a model hypothesis, not a fact about class spacing. The recorded cumulative-threshold probe did not improve nominal accuracy.
- Do not call a competition prediction current status without current predictors, an as-of time and a current labelling rule.
- Do not turn `status_group` directly into rehabilitation priority without cost, feasibility, need and beneficiary information.

The executable evidence and recorded outputs are in [`04-target-structure-and-robustness.ipynb`](../notebooks/04-target-structure-and-robustness.ipynb).

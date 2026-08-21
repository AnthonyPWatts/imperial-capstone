# Notebooks

Use numbered notebooks so the analysis can be followed in order.

## Data audit

`data-audit/` is the canonical audit area. Its dedicated target-label analysis
is [`data-audit/00-target-label-analysis/00-target-label-analysis.ipynb`](data-audit/00-target-label-analysis/00-target-label-analysis.ipynb),
named for the column's modelling role rather than the dataset-specific column
name. The area also contains an overall audit and one CSV-order-prefixed folder
for every raw non-`id` predictor, including columns proposed for structural
removal. See [`data-audit/README.md`](data-audit/README.md) for the full index.

Every predictor has the same three-part contract:

1. `01-<feature>-basic-breakdown.ipynb`: the repeatable analysis for its type.
2. `02-<feature>-noteworthy-findings.ipynb`: evidence, interpretation and decision.
3. `03-<feature>-related-features.ipynb`: nominated relationships and pairwise checks.

The maintained findings report is
[`data-audit/00-overall/00-overall-data-audit.md`](data-audit/00-overall/00-overall-data-audit.md).
Supporting executable and feature-family audits remain under
`data-audit/00-overall/`. Rebuild the per-predictor notebooks and their index
with `data-audit/generate_data_audit_notebooks.py`; the generator does not
overwrite the findings report.

The temporary full-data candidate experiment sits with its generated submission
records under
[`../submissions/2026-08-15-early-experiments/`](../submissions/2026-08-15-early-experiments/).
It does not replace the formal split and evaluation sequence below.

## Formal modelling sequence

1. [`02-baseline.ipynb`](02-baseline.ipynb): validated 59,400-row,
   36-predictor original modelling-frame handoff, kept separate from the
   unlabelled competition data, with a frozen 20% local test set and five
   stratified development folds plus the cross-validated majority-class
   reference.
2. [`03-model-comparison.ipynb`](03-model-comparison.ipynb): initial 29-feature
   policy and fold-fitted preprocessing, with seven classifier families and a
   bounded equal-weight ensemble round. Random Forest plus histogram boosting
   leads the development folds at 81.37%.
3. [`04-target-structure-and-robustness.ipynb`](04-target-structure-and-robustness.ipynb):
   provenance and collection-time meaning of the target, competing nominal,
   ordinal and two-stage interpretations, exact-predictor conflicts, and a
   frozen-fold constrained-tree probe of both binary hierarchies.
4. [`05-final-model-and-submission.ipynb`](05-final-model-and-submission.ipynb):
   frozen selection, one-time 80.82% local-test result, full-data refit and the
   validated competition candidate.
5. [`06-blend-calibration-and-submission.ipynb`](06-blend-calibration-and-submission.ipynb):
   two bounded component weights and a genuinely nested calibrated stack. None
   passed the mean-plus-three-fold gate, so no candidate was generated and the
   local test remained unopened.
6. [`07-model-family-screen-and-submissions.ipynb`](07-model-family-screen-and-submissions.ipynb):
   bounded CatBoost, XGBoost, LightGBM and bagged-tree trials on the unchanged
   feature policy, including direct pairings with each incumbent component, a
   stricter second-wave gate, seed stability and selection of up to two
   materially different candidates.
7. [`08-minority-class-oversampling.ipynb`](08-minority-class-oversampling.ipynb):
   fold-safe comparison of the unbalanced Random Forest, balanced class weights
   and traceable random oversampling, plus reproducible local development and
   fold-specific oversampled-data artefacts.
8. [`09-oversampled-submission-replay.ipynb`](09-oversampled-submission-replay.ipynb):
   exact replay of the three 21 August submission recipes under full,
   80%-replica and final 2.5× oversampling, comparing labelled local-test
   metrics and unlabelled competition prediction shares without uploading any.
9. [`10-probability-reliability-and-edge-cases.ipynb`](10-probability-reliability-and-edge-cases.ipynb):
   no-refit diagnosis of confidence reliability, classwise probabilities,
   winning margins, component disagreement and repair-related edge groups after
   freezing the 2.5× multiplier.
10. [`11-geography-feature-family-and-robustness.ipynb`](11-geography-feature-family-and-robustness.ipynb):
    fixed-model coordinate and administrative ablations, paired-coordinate
    validity, fold-fitted hierarchical centroid imputation, bounded component
    crossing and an LGA-disjoint transfer sensitivity.
11. [`12-funder-installer-high-cardinality.ipynb`](12-funder-installer-high-cardinality.ipynb):
    no-refit presentation of the fold-fitted rare-grouping and frequency screen
    for `funder` and `installer`, plus the best challenger's LGA-disjoint
    sensitivity and retained-feature decision.
12. [`13-numeric-state-and-imputation.ipynb`](13-numeric-state-and-imputation.ipynb):
    no-refit presentation of numeric sentinel, state, removal and fold-fitted
    geographic-imputation policies, bounded component crossing and the closest
    challenger's LGA-disjoint sensitivity.
13. [`14-management-hierarchy.ipynb`](14-management-hierarchy.ipynb): no-refit
    comparison of management, scheme management, their deterministic group and
    bounded joint or relationship representations.

Notebook 05 was completed before the next-round Notebook 04 so today's selected
candidate could be submitted without conflating later robustness experiments
with the frozen selection.

Keep notebooks focused on analysis. Move stable, reusable code into `../src/`.

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
14. [`15-ann-candidate-screen.ipynb`](15-ann-candidate-screen.ipynb): no-refit
    presentation of seven fold-safe scaled one-hot MLPs, their audited
    organisation representations, error diversity and fixed probability blends.
15. [`16-physical-categorical-hierarchies.ipynb`](16-physical-categorical-hierarchies.ipynb):
    no-refit presentation of extraction, source, quality and waterpoint
    hierarchy ablations, parsimony evidence and bounded component crosses.
16. [`17-regional-specialisation.ipynb`](17-regional-specialisation.ipynb):
    presentation of regional-prior diagnostics, fold-fitted regional Random
    Forest routing, explicit fallback and the fixed partially pooled candidate.
17. [`18-expanded-ensemble-voters.ipynb`](18-expanded-ensemble-voters.ipynb):
    no-refit presentation of bounded CatBoost, LightGBM and MLP additions to the
    accepted two-voter ensemble, including diversity and probability quality.
18. [`19-binary-reduction.ipynb`](19-binary-reduction.ipynb): fold-safe removal
    of repair cases from training, a rebuilt eleven-family binary screen, nine
    fixed votes and assessment against unchanged three-class development and
    local-test labels.
19. [`20-fuzzy-target-membership.ipynb`](20-fuzzy-target-membership.ipynb):
    fold-safe triangular soft memberships at four fixed overlaps, accepted-
    component refits, one-fuzzy-component crosses and crisp-label evaluation.
20. [`21-class-membership-probabilities.ipynb`](21-class-membership-probabilities.ipynb):
    accepted out-of-fold and competition class probabilities retained as
    three-way row memberships, with calibration, ambiguity and repair-boundary
    diagnostics and the outlier-filtering hand-off.
21. [`22-outlier-filtering.ipynb`](22-outlier-filtering.ipynb): bounded
    training-only physical, duplicate-conflict and Isolation Forest filters,
    unchanged validation scoring, class-removal audits and component crosses.
22. [`23-cross-fitted-target-encoding.ipynb`](23-cross-fitted-target-encoding.ipynb):
    inner-cross-fitted multiclass identity encodings, complete accepted-model
    refits, fixed component crosses and a bounded low-weight contribution curve.
23. [`24-cross-fitted-spatial-outcomes.ipynb`](24-cross-fitted-spatial-outcomes.ipynb):
    inner-cross-fitted neighbour class rates and radius, invalid-coordinate
    fallback, direct spatial voting and accepted-component refits.
24. [`25-catboost-deferred-identities.ipynb`](25-catboost-deferred-identities.ipynb):
    one fixed depth-8 native-categorical CatBoost representation over all six
    deferred identities, bounded low-weight contributions and the first new
    promotion-gate pass.
25. [`26-catboost-identity-confirmation.ipynb`](26-catboost-identity-confirmation.ipynb):
    frozen development-to-local-test refit, paired accuracy changes, probability
    quality and promotion to a full-labelled-data competition refit.
26. [`27-catboost-identity-context.ipynb`](27-catboost-identity-context.ipynb):
    pre-fit composite-support audit, one context-qualified native CatBoost and
    fixed-weight comparison with the promoted complete-identity recipe.
27. [`28-catboost-identity-follow-ups.ipynb`](28-catboost-identity-follow-ups.ipynb):
    cached source-plus-class component cross and one diversity-motivated
    depth-7 complete-identity CatBoost substitution.
28. [`29-deferred-name-text.ipynb`](29-deferred-name-text.ipynb): fold-fitted
    character TF-IDF over all six deferred names, one regularised logistic model
    and one fixed low-weight contribution to the promoted vote.
29. [`30-catboost-identity-physical-backoffs.ipynb`](30-catboost-identity-physical-backoffs.ipynb):
    one complete-identity CatBoost augmented with all four deterministic
    physical parents as native back-off layers at the unchanged 20% weight.
30. [`31-transductive-pseudo-labelling.ipynb`](31-transductive-pseudo-labelling.ipynb):
    fold-safe teacher/student XGBoost over a fixed 98%-confidence subset of
    unlabelled competition rows, with original validation left untouched.
31. [`32-confident-label-filtering.ipynb`](32-confident-label-filtering.ipynb):
    three-way inner-OOF XGBoost disagreement filtering inside each outer fold,
    class-removal audit and untouched validation scoring.
32. [`33-recording-time-features.ipynb`](33-recording-time-features.ipynb):
    one fixed recording year, month and year-month policy beside elapsed days,
    with accepted XGBoost and Random Forest refits.
33. [`34-identity-frequencies.ipynb`](34-identity-frequencies.ipynb): six
    fold-fitted deferred-identity occurrence counts beside accepted features.
34. [`35-pump-age-cohorts.ipynb`](35-pump-age-cohorts.ipynb): one fixed
    categorical age cohort tested across all promoted components.
35. [`36-catboost-identity-seed-bag.ipynb`](36-catboost-identity-seed-bag.ipynb):
    one additional fixed CatBoost seed and an equal within-component seed bag.
36. [`37-catboost-identity-full-hierarchy.ipynb`](37-catboost-identity-full-hierarchy.ipynb):
    all seven supplied physical parents added to native identity CatBoost.
37. [`38-identity-hashing.ipynb`](38-identity-hashing.ipynb): one fixed 4,096-
    column target-free hash block for the six deferred identities.
38. [`39-one-vs-rest-xgboost.ipynb`](39-one-vs-rest-xgboost.ipynb): independent
    binary XGBoost class boundaries and one equal boundary bag.
39. [`40-one-vs-rest-catboost.ipynb`](40-one-vs-rest-catboost.ipynb): independent
    native-identity CatBoost class boundaries and fixed boundary bags.
40. [`41-hard-majority-decisions.ipynb`](41-hard-majority-decisions.ipynb): one
    deterministic hard-consensus rule with the promoted soft-vote tie-break.
41. [`42-categorical-frequency-forest.ipynb`](42-categorical-frequency-forest.ipynb):
    all categorical/date levels replaced by fold-fitted occurrence support in a
    compact Random Forest and equal representation bag.
42. [`43-spatial-height-imputation.ipynb`](43-spatial-height-imputation.ipynb):
    fixed ten-neighbour coordinate reconstruction for unavailable GPS height,
    accepted component refits and one equal original/imputed representation bag.
43. [`44-external-representation-synthesis.ipynb`](44-external-representation-synthesis.ipynb):
    one no-refit equal average of the two archive-derived candidate votes and a
    closing orthogonal pump-age synthesis check.
44. [`45-transductive-frequency-forest.ipynb`](45-transductive-frequency-forest.ipynb):
    fixed-universe target-free category support from development and supplied
    competition covariates, one forest refit and unchanged ensemble weights.
45. [`46-learned-numeric-imputation.ipynb`](46-learned-numeric-imputation.ipynb):
    fold-fitted spatial height followed by target-free log-amount regression,
    accepted component refits and fixed ensemble substitutions.
46. [`47-recording-batch-catboost.ipynb`](47-recording-batch-catboost.ipynb):
    exact recording date as one well-supported native CatBoost batch category,
    with direct and archive-synthesis substitutions.
47. [`48-regional-interactions.ipynb`](48-regional-interactions.ipynb): three
    fixed region-by-physical composites evaluated across all promoted model
    families without splitting the training sample.
48. [`49-source-identity-local-comparison.ipynb`](49-source-identity-local-comparison.ipynb):
    controlled local-test ordering of the public-leading source policy, its
    fixed identity cross and the already-prepared identity candidate.
49. [`50-archive-synthesis-confirmation.ipynb`](50-archive-synthesis-confirmation.ipynb):
    fixed six-component synthesis confirmation on the used local test and its
    validated full-labelled-data competition candidate.
50. [`51-radial-frequency-forest.ipynb`](51-radial-frequency-forest.ipynb):
    archived origin-distance completion of the occurrence-count forest and its
    fixed archive substitutions.
51. [`52-physical-state-interactions.ipynb`](52-physical-state-interactions.ipynb):
    fixed quantity-by-extraction, source and waterpoint CatBoost composites and
    their promoted/archive substitutions.
52. [`53-exact-archive-frequency-forest.ipynb`](53-exact-archive-frequency-forest.ipynb):
    faithful adapted 1,000-tree, five-feature-per-split completion of the
    archived radial occurrence-count forest.
53. [`54-radial-random-forest.ipynb`](54-radial-random-forest.ipynb): shared
    radial distance in the accepted one-hot Random Forest and its fixed
    promoted/archive crosses.
54. [`55-probability-stack.ipynb`](55-probability-stack.ipynb): one constrained
    centred-log multinomial combiner over all six archive memberships.

Notebook 05 was completed before the next-round Notebook 04 so today's selected
candidate could be submitted without conflating later robustness experiments
with the frozen selection.

Keep notebooks focused on analysis. Move stable, reusable code into `../src/`.

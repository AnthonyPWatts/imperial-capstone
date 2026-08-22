# Source code

Keep stable, reusable behaviour in focused modules. Prefer ordinary functions
over stateful `Manager` classes unless a later responsibility genuinely needs
shared state or a lifetime of its own.

Current modules:

- `baseline_evaluation.py` evaluates the majority-class reference across the
  frozen development folds without exposing the local test or competition rows.
- `data_preparation.py` exposes `remove_known_redundant_columns` for the single
  fixed 40-to-37 column-removal step.
- `data_partitioning.py` freezes the stratified local test membership and five
  development cross-validation folds, exposes them as a scikit-learn split and
  records non-sensitive membership fingerprints.
- `feature_engineering.py` applies the deterministic initial feature policy,
  including elapsed days from a fixed competition-era date rather than raw
  calendar fields, validates paired coordinates against a conservative
  Tanzania envelope and exposes the policy as a scikit-learn transformer.
- `geography_features.py` declares the bounded geography feature-family screen,
  including coordinate and administrative ablations, composites, fixed grids
  and fold-fitted hierarchical coordinate-centroid imputation.
- `geography_evaluation.py` evaluates those policies with the accepted
  XGBoost/Random Forest recipe, applies the pre-declared promotion gate and
  creates LGA-disjoint development folds for a separate transfer sensitivity.
- `high_cardinality_features.py` declares conservative `funder` and `installer`
  policies, preserves distinct missing and sentinel states, and fits rare or
  frequency mappings only from the current training partition.
- `high_cardinality_evaluation.py` evaluates those organisation policies with
  the accepted XGBoost/Random Forest recipe and the unchanged promotion gate.
- `numeric_features.py` declares bounded amount, height, population and
  `num_private` policies, including fold-fitted LGA, region and global median
  fallbacks that retain the original missing-state flags.
- `numeric_evaluation.py` evaluates those numeric policies with the accepted
  recipe and supports the two bounded no-refit cross-policy component votes.
- `management_features.py` declares deterministic policies for management,
  scheme management, the coarse group and their bounded joint representations.
- `management_evaluation.py` evaluates those management-family policies with
  the accepted XGBoost/Random Forest recipe and unchanged promotion gate.
- `ann_evaluation.py` evaluates bounded scaled one-hot multilayer perceptrons
  on the frozen folds, reuses audited organisation-feature policies and tests
  fixed low-weight probability blends against the accepted ensemble.
- `physical_hierarchy_features.py` declares the bounded extraction, source,
  quality and waterpoint hierarchy policies while preserving the exact accepted
  frame for each family's baseline.
- `physical_hierarchy_evaluation.py` evaluates those policies with the accepted
  vote, applies separate accuracy and parsimony gates and supports bounded
  no-refit crosses of policy-specific model components.
- `regional_specialisation_evaluation.py` fits one shared-spec Random Forest
  per sufficiently supported region inside each outer fold, provides an
  explicit global fallback, evaluates hard and partially pooled routing, and
  diagnoses fold-fitted regional-prior corrections.
- `expanded_ensemble_evaluation.py` evaluates three fixed low-weight third
  voters around the accepted 55:45 ensemble from aligned OOF probabilities,
  including strength, diversity and probability-quality diagnostics.
- `binary_reduction_evaluation.py` removes repair cases only from each current
  training partition, evaluates eleven binary model families and fixed soft
  votes against the unchanged three-class target, and refits a frozen binary
  recipe for an explicitly labelled local-test comparison.
- `fuzzy_target_evaluation.py` converts triangular adjacent memberships into
  source-mass-preserving weighted training observations, refits the accepted
  XGBoost and Random Forest components on frozen folds, and evaluates complete
  fuzzy votes plus one-fuzzy-component crosses against crisp labels.
- `model_preprocessing.py` combines that transformer with fold-fitted numeric
  imputation, missing indicators, rare-category handling and one-hot encoding;
  it can return identical sparse or dense feature values, optionally standardises
  numeric values inside each fold for scale-sensitive methods, and its first-fold
  smoke check never accesses local test or competition rows.
- `model_evaluation.py` evaluates the constrained tree, logistic regression,
  Gaussian naïve Bayes, KNN, Extra Trees, histogram boosting and Random Forest
  across the frozen development folds. It returns metrics, timing diagnostics,
  aligned out-of-fold probabilities, confusion matrices, pairwise diversity and
  fixed-weight soft votes over two or more components. It also exposes a nested
  forest-plus-boosting stack whose regularised combiner is fitted from inner
  out-of-fold probabilities separately inside every outer fold.
- `blend_submission.py` refits a development-selected fixed-weight vote or
  calibrated stack on all labelled rows and hands its probabilities to the
  shared validated competition-prediction builder.
- `gpu_model_evaluation.py` runs the bounded CatBoost, XGBoost, LightGBM and
  bagged-tree family screens on the frozen folds and feature policy. It selects
  boosting tree counts on an inner split before refitting and scoring the
  untouched outer validation fold; fixed-size forests remain fold-fitted. Its
  optional preprocessor factory lets controlled feature-family screens retain
  the same nested tree-count selection contract, while its optional CatBoost
  feature callback supports audited native-categorical representations without
  changing earlier candidates.
- `model_screen_submission.py` selects gate-passing model combinations,
  requires a materially different second candidate, refits shared members only
  once and builds validated competition predictions.
- `final_model.py` freezes the selected Random Forest and histogram-boosting
  vote, performs its one-time local-test evaluation, refits both components on
  all labelled original rows and validates the competition submission. Its
  public prediction builder is reused by later submission challengers.
- `modelling_data.py` exposes a small `ModellingData` record plus preparation
  and summary functions for the validated pre-split original data and the
  separate competition-scoring data.
- `oversampling.py` creates traceable random-oversampling variants from one
  training partition at a time, compares baseline, balanced-weight and
  oversampled Random Forest recipes on frozen folds, and writes compressed
  development/fold artefacts with seeds, source lineage and checksums.
- `oversampled_submission_replay.py` freezes the three 21 August recipe weights
  and XGBoost iteration counts, fits their unique components once per training
  regime, and reports nominal/class-aware metrics plus submission class shares.
- `ordinal_target.py` constructs coherent cumulative-threshold probabilities,
  applies explicit two-cut-off decisions, selects cut-offs on a disjoint
  calibration partition and reports order-aware error diagnostics.
- `probability_diagnostics.py` separates hard-label balance from probability
  reliability, computes log-loss/Brier/calibration summaries, and identifies
  low-margin, disagreement and repair-related edge groups without refitting. It
  also validates and exports row-level three-class membership tables with
  confidence, runner-up, margin, entropy and ambiguity fields.
- `outlier_filtering_evaluation.py` fits bounded physical, exact-duplicate and
  class-blind multivariate filters only on each outer-training partition,
  audits class-specific removal and refits the accepted components while every
  original validation row remains in scoring.
- `target_encoding_features.py` normalises five deferred identities and adds
  empirical-Bayes multiclass rates whose training values are generated by a
  fixed inner cross-fit; unseen validation identities receive training priors.
- `target_encoding_evaluation.py` refits the accepted components under the
  location, organisation and combined target-encoding policies and applies the
  unchanged selection gate.
- `spatial_outcome_features.py` creates inner-cross-fitted local class rates
  and neighbour radius from a training-only haversine Ball Tree, with class-
  prior fallback for invalid coordinates and no self-neighbours.
- `spatial_outcome_evaluation.py` compares the fixed spatial feature policy,
  its direct probability voter and component crosses with the accepted vote.
- `catboost_identity_evaluation.py` adds all six deferred identity fields as
  conservatively normalised native CatBoost categoricals, optionally adds the
  three predeclared supported context pairs and evaluates each fixed depth-8
  representation on the frozen folds.
- `catboost_identity_confirmation.py` refits the accepted and preselected
  complete-identity recipes from development to local test, scores unchanged
  three-class labels, reports paired hard-prediction changes and builds both
  validated full-data competition records from shared components.
- `source_data_validation.py` exposes `validate_raw_feature_schema`,
  `validate_label_frame` and `validate_aligned_ids` for the three source-frame
  checks used by the audit notebook.
- `raw_feature_column_policy.json` contains the fixed schema and evidence-backed
  removal assumptions shared by those modules.
- `predictor_audit.py` contains the common numeric, categorical, target-support,
  structural-missingness, train/test coverage, hierarchy and pairwise-relationship
  summaries used by the focused audit notebooks. Its missingness summaries keep
  pandas nulls and source blanks separate from feature-specific semantic sentinels.
- `predictor_audit_catalogue.json` records the raw predictor order, audit type,
  provisional disposition, current evidence and explicitly nominated related
  features used to generate the canonical audit structure.

File loading, notebook narration, plotting and one-off exploration remain in the
notebooks until repeated use gives them a clearer reusable boundary.

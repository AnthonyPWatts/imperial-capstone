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
  the same nested tree-count selection contract.
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
  low-margin, disagreement and repair-related edge groups without refitting.
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

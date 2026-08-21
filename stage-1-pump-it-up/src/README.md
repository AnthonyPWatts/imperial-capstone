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
  calendar fields, and exposes it as a scikit-learn transformer.
- `model_preprocessing.py` combines that transformer with fold-fitted numeric
  imputation, missing indicators, rare-category handling and one-hot encoding;
  it can return identical sparse or dense feature values, optionally standardises
  numeric values inside each fold for scale-sensitive methods, and its first-fold
  smoke check never accesses local test or competition rows.
- `model_evaluation.py` evaluates the constrained tree, logistic regression,
  Gaussian naïve Bayes, KNN, Extra Trees, histogram boosting and Random Forest
  across the frozen development folds. It returns metrics, timing diagnostics,
  aligned out-of-fold probabilities, confusion matrices, pairwise diversity and
  equal-weight soft votes over two or more components.
- `final_model.py` freezes the selected Random Forest and histogram-boosting
  vote, performs its one-time local-test evaluation, refits both components on
  all labelled original rows and validates the competition submission.
- `modelling_data.py` exposes a small `ModellingData` record plus preparation
  and summary functions for the validated pre-split original data and the
  separate competition-scoring data.
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

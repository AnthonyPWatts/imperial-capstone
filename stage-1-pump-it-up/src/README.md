# Source code

Keep stable, reusable behaviour in focused modules. Prefer ordinary functions
over stateful `Manager` classes unless a later responsibility genuinely needs
shared state or a lifetime of its own.

Current modules:

- `data_preparation.py` exposes `remove_known_redundant_columns` for the single
  fixed 40-to-37 column-removal step.
- `source_data_validation.py` exposes `validate_raw_feature_schema`,
  `validate_label_frame` and `validate_aligned_ids` for the three source-frame
  checks used by the audit notebook.
- `raw_feature_column_policy.json` contains the fixed schema and evidence-backed
  removal assumptions shared by those modules.
- `predictor_audit.py` contains the common numeric, categorical, target-support,
  train/test coverage, hierarchy and pairwise-relationship summaries used by the
  focused audit notebooks.
- `predictor_audit_catalogue.json` records the raw predictor order, audit type,
  provisional disposition, current evidence and explicitly nominated related
  features used to generate the canonical audit structure.

File loading, notebook narration, plotting and one-off exploration remain in the
notebooks until repeated use gives them a clearer reusable boundary.

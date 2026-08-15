# Notebooks

Use numbered notebooks so the analysis can be followed in order.

## Data audit

- `01-data-audit.ipynb`: main audit and decision log.
- `01-data-audit-status_group.ipynb`: focused initial analysis of the target.
- `01-data-audit-coordinates.ipynb`: paired coordinate validity, repeated locations and regional coverage.
- `01-data-audit-amount_tsh.ipynb`: focused analysis of `amount_tsh`.
- `01-data-audit-numeric.ipynb`: `gps_height`, `num_private` and `population` distributions, sentinels and drift.
- `01-data-audit-temporal.ipynb`: recording-date coverage, construction-year validity and derived pump age.
- `01-data-audit-binary.ipynb`: nullable `public_meeting` and `permit` states.
- `01-data-audit-geographic-categories.ipynb`: administrative and basin categories, hierarchy conflicts and coverage.
- `01-data-audit-organisation-and-name.ipynb`: funder, installer and waterpoint-name cardinality and memorisation risk.
- `01-data-audit-management.ipynb`: management and scheme overlap, missingness and identity risk.
- `01-data-audit-infrastructure.ipynb`: extraction and source hierarchies.
- `01-data-audit-service-categories.ipynb`: payment, water quality/quantity and waterpoint-type categories.

The focused notebooks share one audit contract: preserve raw blank/sentinel
distinctions, report training and test coverage, mark support for target-rate
comparisons, check related-field mappings, and finish with a per-feature decision
register. Rebuild them with `generate_predictor_audit_notebooks.py`, then execute
them from this directory so the project-relative data paths resolve.

## Modelling sequence

1. `02-baseline.ipynb`
2. `03-model-comparison.ipynb`
3. `04-tuning-and-evaluation.ipynb`
4. `05-final-model-and-submission.ipynb`

Keep notebooks focused on analysis. Move stable, reusable code into `../src/`.

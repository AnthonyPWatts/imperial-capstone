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

## Modelling sequence

1. `02-baseline.ipynb`
2. `03-model-comparison.ipynb`
3. `04-tuning-and-evaluation.ipynb`
4. `05-final-model-and-submission.ipynb`

Keep notebooks focused on analysis. Move stable, reusable code into `../src/`.

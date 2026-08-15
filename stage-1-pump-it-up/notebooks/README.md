# Notebooks

Use numbered notebooks so the analysis can be followed in order.

## Data audit

`data-audit/` is the canonical audit area. It contains an overall audit and one
CSV-order-prefixed folder for every raw non-`id` predictor, including columns
proposed for structural removal. See [`data-audit/README.md`](data-audit/README.md)
for the full index.

Every predictor has the same three-part contract:

1. `01-<feature>-basic-breakdown.ipynb`: the repeatable analysis for its type.
2. `02-<feature>-noteworthy-findings.ipynb`: evidence, interpretation and decision.
3. `03-<feature>-related-features.ipynb`: nominated relationships and pairwise checks.

Existing broader and feature-family audits are preserved under
`data-audit/00-overall/`. Rebuild the canonical notebooks with
`data-audit/generate_data_audit_notebooks.py`.

## Modelling sequence

1. `02-baseline.ipynb`
2. `03-model-comparison.ipynb`
3. `04-tuning-and-evaluation.ipynb`
4. `05-final-model-and-submission.ipynb`

Keep notebooks focused on analysis. Move stable, reusable code into `../src/`.

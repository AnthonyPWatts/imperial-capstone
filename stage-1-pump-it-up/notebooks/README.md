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
3. `04-target-structure-and-robustness.ipynb`: planned next-round work on class
   meaning, pairwise separability and grouped geographic sensitivity.
4. [`05-final-model-and-submission.ipynb`](05-final-model-and-submission.ipynb):
   frozen selection, one-time 80.82% local-test result, full-data refit and the
   validated competition candidate.

Notebook 05 was completed before the next-round Notebook 04 so today's selected
candidate could be submitted without conflating later robustness experiments
with the frozen selection.

Keep notebooks focused on analysis. Move stable, reusable code into `../src/`.

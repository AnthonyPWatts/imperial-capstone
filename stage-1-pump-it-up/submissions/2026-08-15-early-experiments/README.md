# Early submission experiments — 15 August 2026

This folder contains a deliberately temporary, self-contained experiment that
turns the completed data audit into three useful DrivenData submissions before
the formal modelling workflow begins.

The notebook does not create or evaluate a train/validation split. Each model
fits on the complete supplied labelled data, and no local performance comparison
is recorded here. Data splitting and model evaluation remain deferred to the
formal modelling phase.

Feature preparation follows only evidence already recorded in the project:

- remove the three agreed redundant columns;
- keep the useful categorical hierarchies;
- separate `id` from predictors;
- turn `date_recorded` into calendar fields;
- derive pump age where `construction_year` is known;
- mark common zero placeholders and mask invalid coordinates;
- use fixed categorical missing markers and full-data numeric imputation.

## Three experiments

1. `01-extra-trees.csv` — an Extra Trees ensemble over the audit-led features.
2. `02-hist-gradient-boosting.csv` — histogram gradient boosting over the same
   features, giving a materially different learning algorithm.
3. `03-soft-vote-ensemble.csv` — an equal-probability blend of the first two
   full-data models.

The generated CSV files are ignored by Git because they repeat the competition
test identifiers. Run `01-generate-early-submissions.ipynb` from this directory
to recreate them from the untouched source data.

## Retained submissions

The exact submitted CSVs remain in this local directory. They are deliberately
ignored by Git to avoid redistributing the competition test identifiers. The
tracked notebook recreates them, while these hashes identify the submitted
bytes unambiguously:

| File | Public score | SHA-256 |
| --- | ---: | --- |
| `01-extra-trees.csv` | 0.8050 | `E2ACDCB283BBCD9DF84311C2BD706F2051B454FD78B1142412517BA7EC273948` |
| `02-hist-gradient-boosting.csv` | 0.7968 | `7F0E79E8B1AAFDB3FC499E23D185FC1B5AB9747B1402D9B429AA14757EFE843E` |
| `03-soft-vote-ensemble.csv` | 0.8170 | `AC488100EB2DD739BD81D801123ED6B44CF86FD083343AA6C416FBC2036759ED` |

The numbering is an experiment sequence, not a performance ranking. The public
scores are also recorded in the parent submission log.

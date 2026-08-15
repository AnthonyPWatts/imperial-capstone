# Pump It Up capstone

Working area for the DrivenData **Pump It Up: Data Mining the Water Table**
competition.

Source: <https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/>

## Contents

- `instructions/competition-reference.md`: concise brief, data dictionary,
  submission shape and links to the authoritative pages.
- `instructions/source-pages/`: dated offline copies of the public competition
  pages.
- `data/`: competition data downloaded after joining.
- `notebooks/`: numbered modelling notebooks and the per-predictor
  `data-audit/` hierarchy.
- `src/`: reusable loading, preprocessing, modelling and evaluation code.
- `reports/`: concise findings and model interpretation.
- `submissions/`: notes about submitted models and their scores.

## Current position

The raw-data audit covers all 39 non-identifier predictors. Three structural
removals are settled, leaving 36 candidate predictors. Six submissions have
tested the submission path and two audit-led model families. The best public
score is `0.8170` from an equal-weight soft-vote ensemble of Extra Trees and
histogram gradient boosting.

The audit-led candidates fitted the complete labelled data. Their leaderboard
scores provide experiment evidence, but no local validation estimate or formal
model comparison. See the [submission log](submissions/README.md), the
[maintained audit report](notebooks/data-audit/00-overall/00-overall-data-audit.md)
and the [data-preparation handoff](reports/data-preparation-next-steps.md).

## Next modelling loop

1. Validate and align feature and label identifiers, then freeze a reproducible
   stratified train/validation split.
2. Recreate missing-value handling, encoding and feature engineering with every
   learned treatment fitted inside the training partition.
3. Compare the majority-class reference, Extra Trees, histogram gradient
   boosting and their soft vote on the same held-out rows.
4. Record accuracy, per-class recall and the confusion matrix, with attention to
   the minority `functional needs repair` class.
5. Run a geographic sensitivity check and targeted feature-family ablations
   before selecting a model.
6. Refit the selected workflow, validate submission shape and record the next
   DrivenData result with its source commit and evidence.

The practical question is how well maintenance data can distinguish functional,
repairable and non-functional water pumps. Class imbalance, missing values,
high-cardinality categories and geographic leakage will shape the model design.

## Data handling

The competition data is governed by the DrivenData competition rules. Do not
commit or redistribute the downloaded CSV/ZIP files. The local `.gitignore`
excludes everything under `data/` except its inventory README.

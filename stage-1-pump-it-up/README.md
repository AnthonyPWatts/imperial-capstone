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
- `notebooks/`: numbered analysis and modelling notebooks.
- `src/`: reusable loading, preprocessing, modelling and evaluation code.
- `reports/`: concise findings and model interpretation.
- `submissions/`: notes about submitted models and their scores.

## Initial modelling plan

1. Audit the labels, missing values, suspicious placeholders and geographic or
   date fields.
2. Establish a reproducible stratified split and a simple majority-class
   baseline.
3. Build a transparent preprocessing pipeline for numeric and categorical
   features.
4. Compare a decision tree with a tree ensemble.
5. Tune only the strongest candidate using cross-validation, then inspect
   per-class errors and feature importance.
6. Validate the submission shape, generate predictions and record the result.

The practical question is how well maintenance data can distinguish functional,
repairable and non-functional water pumps. Class imbalance, missing values,
high-cardinality categories and geographic leakage will shape the model design.

## Data handling

The competition data is governed by the DrivenData competition rules. Do not
commit or redistribute the downloaded CSV/ZIP files. The local `.gitignore`
excludes everything under `data/` except its inventory README.

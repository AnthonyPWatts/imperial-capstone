# Pump It Up capstone

Working area for the DrivenData **Pump It Up: Data Mining the Water Table**
competition.

Source: <https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/>

Browse the [live Stage 1 dashboard](https://anthonypwatts.github.io/imperial-capstone/dashboard/)
and [training label map](https://anthonypwatts.github.io/imperial-capstone/map/),
or return to the [Capstone Hub](https://anthonypwatts.github.io/imperial-capstone/).

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
removals are settled, leaving 36 candidate predictors. Seven submissions have
tested the submission path and the audit-led modelling workflow. The best
public score is `0.8223` from the selected equal-weight soft-vote ensemble of
Random Forest and histogram gradient boosting, up from the previous `0.8170`.

The formal workflow reserves a stratified 20% local test and fixes five
development folds. It compares seven classifier families: a constrained tree,
logistic regression, Gaussian naïve Bayes, KNN, Extra Trees, histogram gradient
boosting and Random Forest. Random Forest is the strongest single model at
80.59% mean accuracy. An equal-weight Random Forest and boosting vote leads the
bounded ensemble round at 81.37%, with 34.28% repair recall and 78.03%
non-functional recall.

The selected vote records 80.82% accuracy on the one-time local test, including
32.10% repair recall and 77.81% non-functional recall. Both components have
then been refitted on all 59,400 labelled rows. The structurally validated
14,850-row competition CSV scored `0.8223` on the public leaderboard. See the
[submission log](submissions/README.md), the [maintained audit report](notebooks/data-audit/00-overall/00-overall-data-audit.md)
and the [data-preparation handoff](reports/data-preparation-next-steps.md).

A bounded follow-up compared 40:60 component weights and a nested calibrated
stack with the equal vote; none passed its gate. A subsequent frozen-feature
model-family screen found that XGBoost is a better partner for Random Forest
than the incumbent histogram booster. The leading 60% depth-8 XGBoost and 40%
Random Forest vote reaches 81.555% across the current-runtime development folds,
up 0.154 percentage points from the recreated incumbent. Two structurally
validated, materially different competition candidates are ready locally and
have not yet been submitted.

## Next modelling loop

1. Record the leaderboard outcomes of the two validated XGBoost and Random
   Forest candidates without using the first result to select the second.
2. Resume the separate evidence-led feature-family plan on a fixed accepted
   model recipe.
3. Extend the target audit into pairwise separability and hierarchical class
   structure if later error analysis supports it.

The practical question is how well maintenance data can distinguish functional,
repairable and non-functional water pumps. Class imbalance, missing values,
high-cardinality categories and geographic leakage will shape the model design.

## Data handling

The competition data is governed by the DrivenData competition rules. Do not
commit or redistribute the downloaded CSV/ZIP files. The local `.gitignore`
excludes everything under `data/` except its inventory README.

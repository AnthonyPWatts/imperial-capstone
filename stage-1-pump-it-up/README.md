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
removals are settled, leaving 36 candidate predictors. Six submissions have
tested the submission path and two audit-led model families. The best public
score is `0.8170` from an equal-weight soft-vote ensemble of Extra Trees and
histogram gradient boosting.

The audit-led candidates fitted the complete labelled data. Their leaderboard
scores provide experiment evidence, but no local validation estimate or formal
model comparison. The formal baseline now reserves a stratified 20% local test
set and fixes five validation folds across the remaining development data. A
constrained decision tree using the initial 29-feature policy averages 74.98%
accuracy across those folds, compared with 54.31% for the majority reference.
Extra Trees improves mean accuracy to 78.84%, repair-class recall from 15.17%
to 39.40% and non-functional recall from 63.96% to 78.24%. Its five-fold run
took about 18 minutes. Histogram gradient boosting raises mean accuracy to
80.21% and completes the same folds in about three minutes, but repair recall
falls to 28.69%. The local test and competition rows remain untouched. See the
[submission log](submissions/README.md), the [maintained audit report](notebooks/data-audit/00-overall/00-overall-data-audit.md)
and the [data-preparation handoff](reports/data-preparation-next-steps.md).

## Next modelling loop

1. Retain out-of-fold probabilities for Extra Trees and histogram boosting,
   then evaluate their equal-weight soft vote across the frozen folds.
2. Compare the blend with the 80.21% boosting leader using accuracy, per-class
   recall and the additional cost of refitting Extra Trees.
3. Decide whether to select a workflow or test class weighting, a more
   economical forest or a targeted feature-family ablation.
4. Run a geographic sensitivity check and targeted feature-family ablations
   before selecting a model.
5. Evaluate the selected workflow once on the untouched local test set.
6. Refit the selected workflow, validate submission shape and record the next
   DrivenData result with its source commit and evidence.

The practical question is how well maintenance data can distinguish functional,
repairable and non-functional water pumps. Class imbalance, missing values,
high-cardinality categories and geographic leakage will shape the model design.

## Data handling

The competition data is governed by the DrivenData competition rules. Do not
commit or redistribute the downloaded CSV/ZIP files. The local `.gitignore`
excludes everything under `data/` except its inventory README.

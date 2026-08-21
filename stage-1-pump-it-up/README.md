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
removals are settled, leaving 36 candidate predictors. Nine submissions have
tested the submission path and the audit-led modelling workflow. The best
public score is `0.8241` from the selected child-weight-1 XGBoost and Random
Forest vote, up from the previous `0.8223`.

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
than the incumbent histogram booster. The leading 55% child-weight-1 depth-8
XGBoost and 45% Random Forest vote reaches 81.625% across the current-runtime
development folds, up 0.223 percentage points from the recreated incumbent.
Two structurally validated, materially different competition candidates were
generated locally. The leading candidate scored 0.8241 publicly; the second
was preselected without using that result and scored 0.8240.

The completed target-structure investigation traces the 74,250 competition
rows to a strong but explicitly inferential match with the cleaned February
2014 Tanzanian Water Point Mapping release. It retains flat nominal prediction
for competition scoring: a constrained-tree operational hierarchy improved
repair recall from 15.2% to 24.0%, but slightly reduced accuracy from 75.0% to
74.8%. A proper cumulative-threshold ordinal probe reduced severe two-step
errors but not nominal accuracy; 28.8% of its independent cumulative estimates
crossed before coherence correction, and nested cut-off tuning still trailed
flat multiclass.

The separate imbalance experiment raised Random Forest repair recall
from 36.9% to 44.7% with random oversampling and to 50.0% with balanced class
weights on development folds. A later exact replay of the three 21 August
recipes did open the labelled local test: full oversampling gained 20.0–22.4
points of repair recall but lost 1.4–1.8 points of accuracy. Retaining 80% of
the added replicas recovered a little accuracy. Six validated competition
CSVs were generated for possible later experiments; none was uploaded. A final
2.5× minority-count point brought competition repair predictions to 6.91–7.10%
and reduced the XGBoost blends' local accuracy cost to 0.40–0.43 points while
retaining a 10.9–11.9-point repair-recall gain. Probability diagnostics show
that its raw repair probabilities are optimistic despite the balanced hard
labels. The multiplier is frozen and three additional validated CSVs were
generated without upload. Compressed data and generated CSVs remain ignored
local artefacts.

The fixed-model geography screen compared twelve alternatives with the accepted
representation. None passed the frozen-fold gate: removing coordinates cost
0.734 percentage points, removing named geography cost 0.299 points, and the
best challenger was fold-fitted hierarchical coordinate-centroid imputation at
81.566%, 0.059 points below the accepted vote. On deliberately harder
LGA-disjoint folds, centroid imputation improved 0.161 points and won four of
five folds, but absolute accuracy fell to 72.26% and repair recall to 2.99%.
This records material geographic-transfer risk without reopening the local test.
See the [geography robustness report](reports/geography-feature-family-and-robustness.md).

The next fixed-model screen introduced `funder` and `installer` separately and
together through fold-fitted rare grouping and frequency. None of ten
challengers passed the frozen-fold gate. Funder frequency was the only result
above baseline at 81.656%, a 0.032-point gain with only two fold wins; installer
and identity-bearing representations were weaker. An LGA-disjoint sensitivity
improved 0.250 points but breached the worst-fold limit and reduced repair
recall, so it did not reopen the primary decision. See the
[high-cardinality report](reports/funder-installer-high-cardinality-screen.md).

The numeric state and imputation screen then compared eleven challengers for
amount, height, population and `num_private`. Every complete vote trailed the
accepted 81.625% recipe; the closest was fold-fitted geographic population
imputation at 81.591%. That treatment improved a separate LGA-disjoint mean by
0.128 points and won four folds, but failed the primary comparison and the
grouped worst-fold limit. The existing numeric representation therefore remains
selected. See the
[numeric-feature report](reports/numeric-state-and-imputation-screen.md).

The management hierarchy screen compared nine alternatives to the accepted
`management` plus `scheme_management` representation. None passed the gate.
Management-only was the best challenger at 81.637%, just 0.013 points above
baseline while removing twelve transformed columns; the difference is too
small for accuracy promotion, so it remains a named parsimony candidate. Coarse
groups, composites and disagreement states were weaker. See the
[management hierarchy report](reports/management-hierarchy-screen.md).

## Next modelling loop

1. Retain the accepted feature policy and continue bounded categorical
   hierarchy ablations with `extraction_type`, `extraction_type_group` and
   `extraction_type_class`, followed by source, quality and waterpoint families.
2. Keep the LGA-disjoint result as a robustness warning; do not replace the
   competition-aligned frozen-fold selection metric silently.
3. If one later upload slot is reserved for a minority-recall experiment, use
   the 2.5× replay as an explicitly exploratory candidate; do not claim
   an expected accuracy gain from the local-test evidence.
4. Retain the operational hierarchy as a named candidate for decision-focused
   work; do not replace the competition's flat target silently.

The practical question is how well maintenance data can distinguish functional,
repairable and non-functional water pumps. Class imbalance, missing values,
high-cardinality categories and geographic leakage will shape the model design.

## Data handling

The competition data is governed by the DrivenData competition rules. Do not
commit or redistribute the downloaded CSV/ZIP files. The local `.gitignore`
excludes everything under `data/` except its inventory README.

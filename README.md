# Imperial ML and AI capstone

Coursework and portfolio work for the Imperial College London Professional
Certificate in Machine Learning and Artificial Intelligence.

## Project structure

| Area | Purpose |
| --- | --- |
| [`index.html`](index.html) | Static Capstone Hub joining the project stages and working tools |
| [`stage-1-pump-it-up/`](stage-1-pump-it-up/) | Applied multiclass classification using the DrivenData *Pump It Up* competition |
| [`stage-2-bbo/`](stage-2-bbo/) | The assessed black-box optimisation (BBO) capstone covering eight unknown functions |
| [`dashboard/`](dashboard/) | Static plan and progress dashboard for the Stage 1 competition |
| [`map/`](map/) | Browser-side interactive map using locally selected Stage 1 training labels |

Stage 1 is a self-contained machine-learning project used to practise the
course workflow on a real operational problem. Stage 2 follows the course's
fixed BBO brief and will be developed as each set of observations is released.

## Current position

As of 22 August 2026, Stage 1 has audited all 39 raw predictors, settled three
structural removals and organised the remaining 36 candidate predictors.
Twelve DrivenData submissions have been made. The best is the source-plus-class
55% XGBoost and 45% Random Forest vote at `0.8246`; its cross-policy hybrid
followed at `0.8244`.

The formal workflow freezes a stratified 20% local test and five development
folds, then compares eleven classifier families and ten bounded ensemble workflows
with fold-fitted preprocessing. Random Forest is the strongest single model in
the earlier broad comparison at 80.59% mean accuracy. The leading XGBoost and
Random Forest vote reaches 81.625% on the frozen development folds. The earlier
forest and histogram-boosting vote remains the last workflow assessed on the
one-time local test, where it recorded 80.82% accuracy, 32.10% repair recall and
77.81% non-functional recall.

The accepted XGBoost and Random Forest workflows were refitted on all 59,400
labelled rows. Their validated 14,850-row competition submissions scored
`0.8241` and `0.8240` on the public leaderboard. A subsequent fixed-model
geography screen retained the accepted feature policy: none of twelve
alternatives passed the frozen-fold gate. Hierarchical coordinate-centroid
imputation improved 4/5 LGA-disjoint folds but exposed a roughly 9.5-point
accuracy drop under that harder geographic-transfer design. The subsequent
fold-fitted `funder` and `installer` screen also retained the policy: none of
ten rare-grouped or frequency challengers passed the gate. Funder frequency
was only 0.032 points above baseline and won two of five folds. Numeric
state and imputation treatments subsequently left the accepted policy
unchanged: all eleven challengers trailed the 81.625% baseline. Categorical
hierarchy ablation retained the management pair as well; management-only was
just 0.013 points higher, below the promotion threshold. A bounded
seven-configuration MLP screen also
retained the accepted tree ensemble: its best standalone reached 78.590%, and
the closest of 84 fixed blends reached 81.616%, just below baseline. The four
physical hierarchy screens subsequently retained granular extraction, source,
quality and waterpoint features. Source plus class was the closest challenger
at 81.635%, but its +0.011-point change won only two folds. Its exploratory
full-data submission nevertheless scored `0.8246`, with the waterpoint-XGBoost
and source-forest cross at `0.8244`. A frozen 2.5× repair-oversampled bag scored
`0.8174`, confirming the expected accuracy penalty without prompting another
oversampling loop. Independent follow-ups then retained the global two-voter
recipe: the fixed 80:20 global/regional-expert blend lost 0.156 development
points, while the closest CatBoost, LightGBM or MLP third-voter addition lost
0.023 points. A subsequent full binary-reduction rebuild improved conditional
functional/non-functional accuracy by 1.073 points, but forced repair recall to
zero and reduced full development accuracy by 1.538 points to 80.086%. Its
79.992% result on the already-used local test confirmed the trade-off; no
competition prediction was generated. Triangular fuzzy memberships were also
unhelpful: the closest fuzzy-XGBoost/hard-forest cross reached 81.534%, 0.090
points below baseline with zero fold wins. The local test remained closed.
The accepted ensemble's ordinary class probabilities were then retained as
row-level memberships rather than changing the training labels. Its development
OOF probabilities remain well aligned with the observed class shares and have
0.708% top-label expected calibration error. On competition data, mean repair
membership is 7.178%, although only 3.953% of rows receive repair as their hard
prediction. This creates a useful uncertainty artefact without another model
selection loop. The next bounded experiment is fold-fitted, training-only
outlier filtering. That screen subsequently retained every row: strict
duplicate cleaning reached 81.566%, while the two class-blind Isolation Forest
policies lost on every fold. The next distinct loop is leakage-safe,
cross-fitted target encoding of deferred high-cardinality fields. That screen
also retained the accepted policy: its best complete vote reached 81.111%.
A 20% organisation-aware XGBoost contribution gained 0.048 points, below the
promotion threshold. The next experiment moves to cross-fitted local spatial
class rates. That screen also stopped: the feature vote fell to 80.995%, while
the direct spatial voter was flat and damaged repair recall. The next distinct
model strategy is ordered native-categorical learning over the complete
deferred identity set. That fixed CatBoost representation produced the first
new promotion-gate pass: a 20% contribution reached 81.740%, up 0.116 points
with four fold wins. The frozen 44:36:20 recipe then confirmed at 81.086% on the
local test, 0.387 points above the identically refitted accepted ensemble, and
its validated full-data competition file is now prepared but not uploaded. It
changes 1.576% of accepted predictions. The next bounded experiment asks
whether fixed geographic context can disambiguate the native identity fields.
That context check subsequently tied the promoted vote exactly and was stopped;
the next no-refit test crosses the identity voter with the strongest existing
physical representation. That cross gained only 0.006 points and a depth-7
identity substitution also trailed, closing those fine-tuning branches. The
next distinct representation is fold-safe character n-grams over sparse names.
That text model also reduced the ensemble result. The next bounded test gives
the identity CatBoost explicit physical hierarchy back-off categories. Those
parents added only three aggregate development wins, so the next distinct test
is fold-safe high-confidence transductive pseudo-labelling.
The main evidence is available in the
[live Stage 1 dashboard](https://anthonypwatts.github.io/imperial-capstone/dashboard/),
[submission log](stage-1-pump-it-up/submissions/README.md),
[data-audit report](stage-1-pump-it-up/notebooks/data-audit/00-overall/00-overall-data-audit.md)
and [regional](stage-1-pump-it-up/reports/regional-specialisation-and-layering.md)
and [expanded-voter](stage-1-pump-it-up/reports/expanded-ensemble-voter-screen.md)
and [binary-reduction](stage-1-pump-it-up/reports/binary-reduction-screen.md)
and [fuzzy-membership](stage-1-pump-it-up/reports/fuzzy-target-membership-screen.md)
and [class-membership](stage-1-pump-it-up/reports/class-membership-probabilities.md)
and [outlier-filtering](stage-1-pump-it-up/reports/outlier-filtering-screen.md)
and [target-encoding](stage-1-pump-it-up/reports/cross-fitted-target-encoding-screen.md)
and [spatial-outcome](stage-1-pump-it-up/reports/cross-fitted-spatial-outcome-screen.md)
and [CatBoost identity](stage-1-pump-it-up/reports/catboost-deferred-identity-screen.md)
and [identity confirmation](stage-1-pump-it-up/reports/catboost-identity-confirmation.md)
and [identity context](stage-1-pump-it-up/reports/catboost-identity-context-screen.md)
and [identity follow-ups](stage-1-pump-it-up/reports/catboost-identity-follow-up-screen.md)
and [name text](stage-1-pump-it-up/reports/deferred-name-text-screen.md)
and [physical back-offs](stage-1-pump-it-up/reports/catboost-identity-physical-backoff-screen.md)
reports.

A bounded blend comparison retained the earlier equal vote: neither fixed
40:60 alternative nor a nested calibrated stack improved at least three of the
five development folds. The subsequent frozen-feature family screen found
XGBoost to be a better Random Forest partner than histogram boosting, producing
the two new public results without reopening the reserved local test.

Stage 2 remains scaffolded while the Module 12 inputs and dates are unavailable.

## Live project hub

The [Capstone Hub](https://anthonypwatts.github.io/imperial-capstone/) links to
both project stages. Open the
[Stage 1 dashboard](https://anthonypwatts.github.io/imperial-capstone/dashboard/)
or [training label map](https://anthonypwatts.github.io/imperial-capstone/map/).
The
[Stage 2 workspace](https://anthonypwatts.github.io/imperial-capstone/stage-2-bbo/)
remains an empty state until its course inputs arrive.

The repository includes the non-sensitive status snapshot that the public
pages load. It excludes competition datasets. Visitors select local copies
through the training map's file pickers.

## Modelling environment

Create a project-local environment and install the recorded notebook and
scikit-learn runtime before executing Stage 1 analyses:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The competition CSV files must already be present under
`stage-1-pump-it-up/data/`; they remain excluded from Git.

## Local project hub

Run the static site from this directory so that the hub, shared status snapshot
and local-data map all resolve from the same origin:

```powershell
python -m http.server 8000
```

Then open <http://localhost:8000/>. The two stage cards lead into separate
workspaces; stage-specific tools and progress are available only after selecting
a stage. Stage 2 remains an explicit empty state until its course inputs and
dates are available.
Non-sensitive stage labels and dashboard headline values live in
[`project-status.json`](project-status.json). Detailed evidence and modelling
decisions remain in the relevant notebooks, reports and maintainer notes.

## Working conventions

- Keep exploratory work in numbered notebooks.
- Move code worth reusing into `src/`.
- Record decisions and results rather than relying on notebook output alone.
- Do not commit competition downloads, supplied BBO observations or secrets.
- Prefer a simple, reproducible baseline before tuning more complex models.

The scaffold gives each part of the modelling work a home. Each stage README
describes the immediate plan and the evidence still missing.

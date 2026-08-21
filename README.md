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

As of 21 August 2026, Stage 1 has audited all 39 raw predictors, settled three
structural removals and organised the remaining 36 candidate predictors. Nine
DrivenData submissions have been made. The best is a 55% XGBoost and 45%
Random Forest vote at `0.8241`; an independently preselected XGBoost depth-bag
alternative followed at `0.8240`.

The formal workflow freezes a stratified 20% local test and five development
folds, then compares ten classifier families and ten bounded ensemble workflows
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
state-plus-magnitude treatments are the next data-focused loop. The main
evidence is available in the
[live Stage 1 dashboard](https://anthonypwatts.github.io/imperial-capstone/dashboard/),
[submission log](stage-1-pump-it-up/submissions/README.md),
[data-audit report](stage-1-pump-it-up/notebooks/data-audit/00-overall/00-overall-data-audit.md)
and [organisation-feature report](stage-1-pump-it-up/reports/funder-installer-high-cardinality-screen.md).

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

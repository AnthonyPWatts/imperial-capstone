# Artificial neural-network candidate screen

## Decision

Retain the accepted 55% XGBoost and 45% Random Forest vote. Seven bounded
multilayer-perceptron (MLP) candidates and 84 fixed probability blends were
evaluated on the five frozen development folds; none passed the promotion gate.

The strongest standalone ANN was a two-hidden-layer 128→64 ReLU network at
78.590% mean accuracy, 3.035 percentage points below the accepted 81.625%
vote. The closest blend assigned 2% weight to the compact 64-unit ANN and
reached 81.616%, 0.008 points below baseline. It won one fold and therefore
does not justify replacing or extending the accepted recipe.

The labelled local test remained unopened. No competition predictions were
generated or uploaded, and no oversampling or class rebalancing participated.

## Evaluation contract

The screen used scikit-learn's existing `MLPClassifier` rather than adding a
large neural-network framework. This kept the experiment reproducible in the
recorded project environment; the available RTX 4070 was not used because
PyTorch, TensorFlow and skorch are not project dependencies.

Every outer fold independently fitted:

- deterministic initial feature engineering;
- median imputation and missing indicators;
- numeric standardisation;
- rare-category handling and sparse one-hot encoding;
- an Adam-trained ReLU MLP with a fixed seed;
- a 10% internal validation split used only for early stopping.

Training stopped after ten non-improving internal-validation epochs, with a
maximum of 120. Hidden-layer shape, L2 regularisation and organisation-feature
representation were the only bounded changes. The richer organisation policies
reused the previously audited fold-fitted 20-row rare groups and frequency
mappings; no aliases, thresholds or target encodings were invented for this
screen.

Blends assigned each MLP 1%, 2%, 3%, 4%, 5%, 10%, 15%, 20%, 25%, 30%, 40% or
50% weight. Promotion still required at least +0.10 percentage points of mean
accuracy, three fold wins, no fold worse than -0.25 points and no more than a
two-point repair-recall loss.

## Standalone results

| Candidate | Mean accuracy | Change | Repair recall | Mean epochs |
| --- | ---: | ---: | ---: | ---: |
| Accepted XGBoost/Random Forest vote | **81.625%** | — | **34.859%** | — |
| ReLU 128→64, current features | **78.590%** | -3.035 pp | 24.203% | 35.0 |
| ReLU 128→64, funder identity + frequency | 78.569% | -3.056 pp | 26.260% | 28.0 |
| ReLU 128→64, stronger L2 | 78.371% | -3.253 pp | 24.087% | 27.8 |
| ReLU 128→64, both identities + frequencies | 78.312% | -3.312 pp | 28.374% | 27.6 |
| ReLU 128 | 78.270% | -3.354 pp | 24.725% | 32.2 |
| ReLU 128→64, funder frequency | 78.232% | -3.392 pp | 27.360% | 31.6 |
| ReLU 64 | 78.197% | -3.428 pp | 24.638% | 53.6 |

The plain 128→64 network used about 47,100 trainable parameters per fold and
stopped after 27–44 epochs. A fold took roughly 6–10 seconds on the current CPU
runtime. Adding both organisation identities expanded the network to roughly
95,000 parameters without improving accuracy.

The richer organisation representation did change the class trade-off: both
identities plus frequencies raised standalone repair recall from 24.2% to
28.4%, but remained 6.5 points below the accepted vote and reduced overall
accuracy by another 0.28 points. This is not a competitive minority-class gain.

## Blend and diversity evidence

The best twelve blends all remained below baseline. The closest cases were:

| ANN contribution | Mean accuracy | Change | Fold wins | Worst fold | Repair recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2% ReLU 64 | **81.616%** | -0.008 pp | 1/5 | -0.074 pp | 34.801% |
| 2% ReLU 128→64 + funder frequency | 81.612% | -0.013 pp | 2/5 | -0.116 pp | 34.801% |
| 1% regularised ReLU 128→64 | 81.610% | -0.015 pp | 2/5 | -0.053 pp | 34.772% |
| 4% ReLU 128→64 | 81.608% | -0.017 pp | 2/5 | -0.116 pp | 34.627% |

The strongest standalone MLP disagreed with the accepted vote on 10.87% of
out-of-fold labels. It alone corrected 3.44% of all development rows, but the
accepted vote alone corrected 6.48%. Across the seven MLPs, ANN-only correct
shares ranged from 3.44% to 3.72%, while accepted-vote-only shares ranged from
6.48% to 7.03%. The diversity is genuine but directionally too weak to improve
the probability vote.

## Interpretation and stop rule

This result rejects a conventional scaled one-hot MLP for the current
competition recipe; it does not establish that every possible neural approach
to tabular data is ineffective. A materially different follow-up would require
an explicitly justified framework and representation, such as learned
categorical embeddings, plus a new bounded validation plan. Installing a GPU
framework merely to enlarge the same MLP search would not be evidence-led.

Stop ANN architecture, seed and blend-weight tuning on these folds. Resume the
planned extraction-type, source, quality and waterpoint hierarchy ablations
with the accepted tree ensemble, where the current evidence remains strongest.

Executable screening lives in
[`scripts/run_ann_screen.py`](../scripts/run_ann_screen.py), with reusable model
and blend evaluation in [`src/ann_evaluation.py`](../src/ann_evaluation.py).
Runtime models and CSV summaries remain ignored local artefacts under
`.runtime/ann-screen/`.

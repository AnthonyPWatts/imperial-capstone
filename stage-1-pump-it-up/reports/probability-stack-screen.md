# Probability-stack screen

## Decision

Retain the fixed arithmetic archive synthesis. A regularised multinomial
logistic combiner over the six components' centred log-memberships reaches
**81.5867%**, 0.2062 percentage points below the 81.7929% archive incumbent.
It loses all five folds, has a 0.2736-point worst-fold loss and reduces repair
recall by 2.6928 points.

The development gate therefore stops the branch before strict nested base
refits, local-test confirmation or competition inference.

## Course-aligned lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Replace uniform class-agnostic probability weights with one constrained class-specific combiner over the six proven archive components. |
| 2. Gather the data | Reuse aligned OOF probabilities for accepted/spatial XGBoost, accepted/frequency/spatial Random Forest and complete-identity CatBoost. |
| 3. Explore the data | Use one fixed centred-log representation and no component, regularisation or meta-feature grid. |
| 4. Clean and preprocess the data | Clip only numerical zero at `1e-6`, centre each component's three log-memberships, and fit standardisation inside each meta-training fold. |
| 5. Select and engineer features | Produce 18 meta-features: three centred log-memberships for each of six components. |
| 6. Define the machine-learning task | Unchanged three-class accuracy with repair recall as a gate diagnostic. |
| 7. Partition the data | Fit the meta-model on four frozen OOF folds and score the fifth; each row's base memberships were originally generated without its own label. |
| 8. Select and train candidate methods | Fit one L2 multinomial logistic model with `C=1`, `lbfgs` and at most 500 iterations. |
| 9. Evaluate and interpret the results | Compare mean accuracy, all five fold changes and repair recall with the arithmetic archive vote. |
| 10. Deploy and iterate | Stop before expensive strict nested refits or any local/competition use because the diagnostic loses every fold. |

## Combiner contract and caveat

For component membership vector $p$, the three meta-features are:

$$
z_c = \log(\max(p_c, 10^{-6})) -
\frac{1}{3}\sum_j \log(\max(p_j, 10^{-6})).
$$

This retains within-component class contrasts while removing an arbitrary
common log scale. Standardisation and logistic fitting use only the current
meta-training rows.

This is a cheap OOF meta diagnostic, not the fully nested final stack. Although
each row's base prediction excludes that row and its label, base predictions
for meta-training rows can come from base fits that included other meta-
validation rows. A strict experiment would rebuild all six components through
inner cross-fitting inside each outer meta fold. That expensive step was
predeclared only if the diagnostic showed useful signal; a 0/5-fold result does
not justify it.

## Results

| Candidate | Accuracy | Change | Fold wins | Worst fold | Repair-recall change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Existing arithmetic archive synthesis | **81.7929%** | — | — | — | — |
| Centred-log logistic probability stack | 81.5867% | **-0.2062 pp** | 0/5 | -0.2736 pp | -2.6928 pp |

The stack converges in 40–46 iterations per fold with coefficient L2 norms of
1.18–1.27. Failure is therefore not a convergence artefact. It raises
non-functional recall slightly but over-corrects the already well-calibrated
repair and functional memberships.

The result also explains why increasingly elaborate fixed blends have yielded
only tiny changes: the six components share a strong, already-useful central
signal, and estimating class-specific weights from repeatedly reused OOF rows
adds variance rather than new information.

## Reproduction

Run from `stage-1-pump-it-up`:

```powershell
..\.venv\Scripts\python.exe .\scripts\run_probability_stack_screen.py
```

Runtime evidence remains ignored under `.runtime/probability-stack-screen/`.

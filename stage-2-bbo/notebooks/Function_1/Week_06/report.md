# Function 1 — Week 6 model review

**Submission confirmation, 31 October 2026:** the project owner confirmed submission of the [eight Week 6 inputs](../../../submissions/Week_06/README.md). Results are pending; seven future queries remain per function. The analysis below retains its pre-submission evidence and query budget.

Prepared on **30 October 2026**, using **15 official observations through round 5**.
The provisional query uses **GP: separate length scales**, fitted to **official plus retained peers**.
Its proposal is **`0.608515-0.846200`**. It has not been submitted by this workflow.

**Later decision:** the prepared input is now **`0.537942-0.522217`**, following
through on the central-region hunch using the signed-log mean maximum. None of
the outputs so far is regarded as clearly non-zero in practical or
source-detection terms. See the [agreed rationale](activity-hypothesis.md).
The provisional model evidence below is retained. Nothing has been submitted.

[Executed prediction notebook](official/model_predictions.ipynb) · [Numerical review](official/model-review.json) ·
[Interactive GP/neural input explorer](official/figures/neural-network-explorer.html)

## Evidence and decision

The fixed coverage sequence ended with the fifth return. This is the first model-guided query decision for Function 1. The monotone transform h = sign(y) log10(1 + |y| / 10^-200) retains signed-score ordering, while making its extreme dynamic range numerically usable. Local and broad fixed GPs remain as comparators alongside optimised shared and separate length scales. Signed-log MAE remains the declared selection criterion. A separately fitted magnitude explanation is retained as a diagnostic and does not choose the next query. A positive signed-log mean does not establish a positive realised return, and inverse-transforming that mean would not give a raw-output posterior mean.

The declared selection criterion is **LOO MAE**, measured in **signed-log height h**. Each row below scores the same official held-out targets.
The selected GP's error is **0.904 times the mean baseline**, a **9.6% reduction**.
The neural ensemble is the mean of all three held-out seed predictions for its validation-selected architecture; no best seed is selected.

| Model | Error |
| --- | --- |
| Mean baseline (official only) | 132.332 |
| Official-only GP: GP: local (l=0.15) | 123.174 |
| Selected query GP (official + retained peer training) | 119.636 |
| Neural seed ensemble (official only): tanh-16 | 136.574 |

Official-only GP has the smaller error when both model families train on official observations only. Any selected peer-assisted GP is shown separately because it has additional training evidence. Architecture and GP selection use these validation results, so selected errors are optimistic estimates of future performance. These are predictive checks on adaptively sampled points, not a test of complete optimisation policies or evidence of a global maximum. The GP query policy remains restricted to the declared GPs because neural seed spread is not a posterior uncertainty estimate.

## Provisional query

| Quantity | Value |
| --- | --- |
| Portal input | `0.608515-0.846200` |
| Predicted target mean | 118.111 |
| Latent posterior SD | 108.324 |
| UCB, mean + 1.96 × SD | 330.425 |
| Mean minus best observed target | -66.7766 |
| Inside this fit's sampled full-input hull | True |
| Distance to nearest known input | 0.0480024 |
| Eligible candidates | 20,488 |
| Query opportunities including Week 6 | 8 |

All output quantities in this table use **signed-log height h**. Posterior SD is conditional on the GP kernel and regularisation. UCB is a selection score, not a returned value or a guarantee. The finite pool retains seeded full-space and incumbent-local candidates, corners, six-decimal bounds and known-input exclusions. The geometric comparator is a finite-pool farthest-point proxy, not a fresh continuous minimax search. The notebook shows exploration weights 0, 1, 1.96 and 3 on the same pool.

The neural mean-led alternative is **`0.631861-0.999178`**, with predicted target **377.195** and seed SD **0.531217**. This is scored on the same eligible candidates. Its seed SD describes fitted-seed variation and must not be used as GP uncertainty.

## Neural exploration

The bounded comparison retains **tanh-8**, **tanh-16** and **ReLU-8-8**, each with seeds **11, 42 and 73**. Maximum training iterations remain 400, with the existing L-BFGS settings and regularisation. Full input and target scalers are fitted only on each fold's training rows.

The offline explorer includes every architecture and seed. Change inputs or the conditional-curve axis, then compare neural means and seed variation with the selected GP's mean and latent SD. Buttons restore the measured incumbent or the provisional GP query. The validation-selected architecture is labelled separately from whichever architecture is displayed. It performs no training or course evaluations.

The selected neural models passed input-gradient checks at the measured incumbent.

![Held-out official-target comparison](official/figures/model-validation.png)

![Conditional GP and neural responses through the incumbent](official/figures/model-response-incumbent.png)

![Conditional GP and neural responses through the provisional query](official/figures/model-response-proposal.png)

## Provenance and limits

Official snapshot SHA-256: `47ab79bd8fa8764f137a63ac00d00fbec77e98b0b5343110fb37880ae802faf6`. Earlier weekly notebooks, snapshots and submitted queries remain unchanged. The observation notebook preserves the historical Week 5 forecast comparison; this notebook's held-out review is retrospective.

There are **6 retained peer rows** in the optional training comparison. Their source repositories were not refreshed. The local audit retains the pinned source hashes and verifies that official rows are unchanged. Each official target is held out while peer rows are training evidence only. Peer reports do not establish an official incumbent or increase the owner's query budget. The numerical review records both scopes' candidates on the same pool where peer data are available.

The saved official and peer-fit comparisons record **76 fitting warnings** across all folds and full fits, with their exact messages retained in the numerical JSON. These include declared convergence or kernel-bound limits; the analysis did not extend the tuning budget in response. Neural results deserve the same scrutiny as the GP assumptions.

## Course lifecycle

1. **Define the goal and scope** — support the project owner's next one-query-per-function decision; compare predictive evidence and preserve a reviewable provisional input.
2. **Gather the data** — use the confirmed official Week 6 snapshot through round 5 and, for Functions 1–4, the unchanged audited 20 October peer snapshot.
3. **Explore the data** — carry forward the measured review; inspect fitted conditional responses, candidate scores and the interactive explorer.
4. **Clean and preprocess the data** — validate the fixed schema and round counts, bounds, hashes and complete observations. Neural input and target scaling are fitted inside each fold; Function 1 retains its fixed signed-log transform.
5. **Select and engineer features** — preserve ordered inputs and the declared Function 2 projection and Function 3 quadratic comparator; no information unavailable at query time is introduced.
6. **Define the machine-learning task** — numerical regression supports sequential continuous black-box maximisation.
7. **Partition the data** — use leave-one-out checks on identical official targets. No independent final test set is available; adaptive sampling and model selection limit interpretation.
8. **Select and train candidate methods** — refit the declared baseline, regression, GP and nine neural configurations. Compare retained peer training on the same official targets; neural architecture selection averages all seeds.
9. **Evaluate and interpret the results** — compare held-out errors, neural seed sensitivity, GP regularisation and exploration weights, conditional views, exact candidate inputs and extrapolation. Fitted predictions remain distinct from measured returns.
10. **Deploy and iterate** — not applicable to deployment: prepare a provisional next query for review. No portal submission or new course evaluation is performed; eight query opportunities remain including Week 6.


![GP mean, uncertainty and acquisition planes](official/figures/model-query-planes.png)

# Measured TabPFN-2 benchmark

Run on **26 November 2026**, using 247 official observations through Round 9. The held-out reads cover 247 leave-one-coordinate-out trials and 72 sequential trials per model. Both models scored the same 8,334 eligible offline candidates. The runner produced 15 fitted landscape comparisons.

**Recommendation.** Keep the current acquisition strategy for all eight functions. The owner GP had lower RMSE and MAE on all eight chronological replays. TabPFN-2 is useful as an offline cross-check on F4 and F6 candidate ranks. Its retrospective advantage on F5, F7 and F8 did not hold when each query was predicted using earlier observations.

Each comparison below reads **GP / TabPFN-2**. Error values are in raw objective units within each function. Recall is overlap with the highest observed quartile, using the top three of nine in every replay. All predictions use the recorded inputs.

## Leave one coordinate out

| Function | Held-out rows | RMSE | MAE | Spearman ρ | Top-quartile recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | 19 | 0.03011 / 0.7485 | 0.008666 / 0.5212 | 0.1772 / 0.09298 | 0.6 / 0.4 |
| F2 | 19 | 0.1419 / 0.1555 | 0.1174 / 0.1277 | 0.8 / 0.786 | 0.8 / 0.8 |
| F3 | 24 | 0.06965 / 0.07699 | 0.03245 / 0.04719 | 0.7313 / 0.07304 | 0.6667 / 0.5 |
| F4 | 39 | 1.307 / 2.034 | 0.8506 / 1.464 | 0.9854 / 0.9534 | 1 / 1 |
| F5 | 29 | 429.4 / 189.3 | 290.7 / 131 | 0.7537 / 0.8409 | 1 / 1 |
| F6 | 29 | 0.2207 / 0.2839 | 0.1713 / 0.2219 | 0.9532 / 0.9118 | 0.875 / 0.75 |
| F7 | 39 | 0.2833 / 0.2285 | 0.1886 / 0.1803 | 0.7953 / 0.7298 | 1 / 1 |
| F8 | 49 | 0.2351 / 0.2281 | 0.1656 / 0.1882 | 0.976 / 0.9751 | 0.8462 / 0.8462 |

## Sequential replay

| Function | Held-out rows | RMSE | MAE | Spearman ρ | Top-quartile recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| F1 | 9 | 0.666 / 1.067 | 0.2231 / 0.6125 | 0.5 / 0.5167 | 0.6667 / 0.6667 |
| F2 | 9 | 0.1228 / 0.1994 | 0.1133 / 0.1585 | 0.8333 / 0.6 | 0.6667 / 0.6667 |
| F3 | 9 | 0.02633 / 0.05943 | 0.0198 / 0.05254 | 0.5 / -0.08333 | 0.6667 / 0.3333 |
| F4 | 9 | 0.8123 / 2.294 | 0.5318 / 1.554 | 0.6833 / 0.8 | 0.3333 / 0.6667 |
| F5 | 9 | 2980 / 3043 | 1524 / 1922 | 0.5167 / 0.03333 | 0.6667 / 0 |
| F6 | 9 | 0.1567 / 0.3785 | 0.1183 / 0.3466 | 0.8333 / 0.5667 | 0.6667 / 1 |
| F7 | 9 | 0.4033 / 0.8767 | 0.3002 / 0.6748 | 0.6833 / 0.35 | 0.6667 / 0.3333 |
| F8 | 9 | 0.1996 / 0.3844 | 0.1542 / 0.3067 | 0.9667 / 0.4 | 1 / 0.3333 |

## Predictive intervals on the sequential replay

The GP interval uses its modelled observation variance and a Gaussian central interval. TabPFN's interval is the 5th to 95th percentile of its forecast distribution. Coverage is an empirical count on nine selected historical inputs. Interval width and coverage need to be read together before using either model's uncertainty for acquisition.

| Function | Inside nominal 90% interval | Mean interval width |
| --- | ---: | ---: |
| F1 | 6/9 · 6/9 | 0.007572 / 0.2291 |
| F2 | 7/9 · 8/9 | 0.3796 / 0.6509 |
| F3 | 8/9 · 7/9 | 0.1315 / 0.2004 |
| F4 | 6/9 · 8/9 | 1.887 / 4.968 |
| F5 | 4/9 · 7/9 | 1201 / 5106 |
| F6 | 8/9 · 8/9 | 0.6533 / 1.133 |
| F7 | 5/9 · 6/9 | 0.4302 / 1.878 |
| F8 | 9/9 · 8/9 | 1.26 / 1.048 |

## Assessment by function

| Function | Evidence and proposed use |
| --- | --- |
| **F1** | Keep the official source-shaped GP. Both models missed the first strong return in Round 7. With that return in training, the GP forecast Round 8 at 1.99724 and TabPFN forecast 0.00807. The peak zoom shows persistent disagreement around the measured near-2 inputs. TabPFN adds no live-query case here. |
| **F2** | Keep the official GP. TabPFN has higher replay error and lower rank correlation. Its 8/9 interval coverage comes with a wider interval than the GP. Both methods put their leading offline scores below the official best of 0.723. The prepared archive centre and official fallback are excluded from this comparison. |
| **F3** | Keep the existing controlled slice decision. TabPFN shrinks its highest predicted candidate to −0.0593 while the GP's EI leader has an extrapolated positive mean. Both values show model dependence around a near-zero response. TabPFN loses both replay error and rank, so use disagreement only to flag that uncertainty. |
| **F4** | Consider an offline second ranker. TabPFN identifies two of the three strongest replay returns and its rank correlation is 0.800. The GP identifies one and correlates 0.683. TabPFN's replay RMSE is 2.294 versus 0.812. The controlled x3 decision remains the live plan. |
| **F5** | The largest retrospective TabPFN improvement is here, with RMSE 189 versus 429. Its sequential RMSE is slightly higher and it retrieves zero of the three strongest rounds. Both models' best candidate means stay below the measured upper corner. Review later evidence offline rather than spending a query on that retrospective win. |
| **F6** | Useful offline rank cross-check. TabPFN retrieves three of three strongest replay rounds versus two for the GP, with replay RMSE 0.378 versus 0.157. That rank signal warrants another look after feedback. The experimental candidate pool uses the official data and supplies no test of archival transfer. |
| **F7** | The retrospective RMSE favours TabPFN, 0.229 versus 0.283. Sequential RMSE is more than double the GP's and TabPFN retrieves one of the three strongest rounds. The bounded experiment gives no reason to change the existing policy. |
| **F8** | Keep the validated starter-only quadratic. TabPFN's slight retrospective advantage over the GP reverses in replay. Its top offline prediction exceeds 10, the fitted maximum of the separate quadratic, illustrating the advantage of testing the known structure. |

Candidate ranks are conditional on the common Sobol and local pool. Neither this pool nor the validation measured any newly suggested location. Absolute disagreement reports the difference in surrogate means. The experiment gives no calibrated probability of improvement from TabPFN or evidence about end-to-end optimisation performance.

## Local output index

Regenerating the benchmark writes the observation-level CSVs and PNGs into the ignored local output directory. [Combined run summary](../../../.runtime/tabpfn-surrogate/results/complete-summary.json).

| Function | Fitted landscapes | Candidate ranking |
| --- | --- | --- |
| F1 | [coverage grid](../../../.runtime/tabpfn-surrogate/results/f1-coverage-grid.png) · [peak zoom](../../../.runtime/tabpfn-surrogate/results/f1-peak-zoom.png) | [CSV](../../../.runtime/tabpfn-surrogate/results/f1-candidates.csv) |
| F2 | [x1 x2](../../../.runtime/tabpfn-surrogate/results/f2-x1-x2.png) | [CSV](../../../.runtime/tabpfn-surrogate/results/f2-candidates.csv) |
| F3 | [x2 x3](../../../.runtime/tabpfn-surrogate/results/f3-x2-x3.png) · [x1 x2](../../../.runtime/tabpfn-surrogate/results/f3-x1-x2.png) | [CSV](../../../.runtime/tabpfn-surrogate/results/f3-candidates.csv) |
| F4 | [x2 x3](../../../.runtime/tabpfn-surrogate/results/f4-x2-x3.png) · [x1 x4](../../../.runtime/tabpfn-surrogate/results/f4-x1-x4.png) | [CSV](../../../.runtime/tabpfn-surrogate/results/f4-candidates.csv) |
| F5 | [x3 x4](../../../.runtime/tabpfn-surrogate/results/f5-x3-x4.png) · [x1 x2](../../../.runtime/tabpfn-surrogate/results/f5-x1-x2.png) | [CSV](../../../.runtime/tabpfn-surrogate/results/f5-candidates.csv) |
| F6 | [x1 x5](../../../.runtime/tabpfn-surrogate/results/f6-x1-x5.png) · [x2 x3](../../../.runtime/tabpfn-surrogate/results/f6-x2-x3.png) | [CSV](../../../.runtime/tabpfn-surrogate/results/f6-candidates.csv) |
| F7 | [x1 x3](../../../.runtime/tabpfn-surrogate/results/f7-x1-x3.png) · [x4 x6](../../../.runtime/tabpfn-surrogate/results/f7-x4-x6.png) | [CSV](../../../.runtime/tabpfn-surrogate/results/f7-candidates.csv) |
| F8 | [x2 x5](../../../.runtime/tabpfn-surrogate/results/f8-x2-x5.png) · [x1 x4](../../../.runtime/tabpfn-surrogate/results/f8-x1-x4.png) | [CSV](../../../.runtime/tabpfn-surrogate/results/f8-candidates.csv) |

Run configuration and interpretation are in the [reproduction guide](README.md). The [runtime and alternatives notes](runtime-notes.md) record the exact TabPFN model, licensing, local and hosted limits, and PFNs4BO compatibility.

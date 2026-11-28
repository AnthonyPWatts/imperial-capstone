# Week 10 Bayesian optimisation with cohort advice

**Experimental batch for review.** The official model generates the shortlist and carries 80% of the ranking weight. The two adviser contributions are capped at 10% each. F8 uses its verified starter-only peak. The input file is ordered F1–F8.

[Copy-ready experimental inputs](advisory-submissions.txt) · [Initial proposal](initial-proposal.md) · [Current decision](README.md) · [Course data-use review](../../docs/capstone-data-policy.md)

| Function | Official-only acquisition winner | Experimental input | Official replay MAE → advisory | Nearest shortlisted candidate to 2024 best |
| --- | --- | --- | ---: | ---: |
| [F1](../../notebooks/Function_1/Week_10/cohort_advisory_review.ipynb) | `0.629188-0.627801` | `0.629188-0.627801` | 0.0013234 → 0.0017638 | 0.000795 |
| [F2](../../notebooks/Function_2/Week_10/cohort_advisory_review.ipynb) | `0.697055-0.511547` | `0.697449-0.485510` | 0.12829 → 0.11789 | 0.196505 |
| [F3](../../notebooks/Function_3/Week_10/cohort_advisory_review.ipynb) | `0.595193-0.990189-0.419529` | `0.295456-0.950070-0.428581` | 0.010855 → 0.010347 | 0.120526 |
| [F4](../../notebooks/Function_4/Week_10/cohort_advisory_review.ipynb) | `0.455297-0.441703-0.383515-0.430931` | `0.455297-0.441703-0.383515-0.430931` | 0.40489 → 0.39082 | 0.042163 |
| [F5](../../notebooks/Function_5/Week_10/cohort_advisory_review.ipynb) | `0.999999-0.999999-0.999999-0.959224` | `0.999999-0.999999-0.999999-0.959224` | 404.65 → 371.74 | 0.040775 |
| [F6](../../notebooks/Function_6/Week_10/cohort_advisory_review.ipynb) | `0.501822-0.322751-0.620839-0.821879-0.000000` | `0.496857-0.309192-0.625685-0.791371-0.000000` | 0.13173 → 0.12615 | 0.134272 |
| [F7](../../notebooks/Function_7/Week_10/cohort_advisory_review.ipynb) | `0.000000-0.135108-0.565684-0.306265-0.311144-0.649736` | `0.000000-0.135108-0.565684-0.306265-0.311144-0.649736` | 0.12332 → 0.10868 | 0.142587 |
| [F8](../../notebooks/Function_8/Week_10/cohort_advisory_review.ipynb) | `0.100000-0.150000-0.130000-0.150000-0.800000-0.500000-0.200000-0.600000` | `0.100000-0.150000-0.130000-0.150000-0.800000-0.500000-0.200000-0.600000` | 0.15495 → 0.15256 | 0.009347 |

Distances are Euclidean on the portal input axes. The replay holds out each official return in turn using the latest frozen adviser snapshots. It measures predictive error retrospectively.

The strongest archived regions for F2, F6 and F7 lie outside the official shortlist. F3's broader candidate relies on extrapolated upside. Its notebook exposes the local alternative alongside the blended winner. The notebooks retain both portal paths as prepared on 25 November. The [submission record](README.md) identifies the confirmed batch.

Regenerate from the capstone root with `./run-python.ps1 stage-2-bbo/scripts/run_week_10_advisory_reviews.py`.

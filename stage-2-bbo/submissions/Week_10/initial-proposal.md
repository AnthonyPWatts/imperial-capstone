# Week 10 initial proposal

**Historical proposal.** The points below preceded the advisory and challenger reviews.
[Current Week 10 decision](README.md) · [Week 10 notebook evidence](../../results/week-10-review.md)

The [separate cohort advisory experiment](advisory-review.md) compares this decision history with an owner-first ensemble and its experimental inputs.

| Function | Proposed input | Nearest known input distance | Rationale |
| --- | --- | ---: | --- |
| 1 | `0.629190-0.627878` | 0.001650 | Test the seven-positive axis-aligned source fit close to all three recent high returns. That raw-score fit puts its peak near 2.0066. The five-positive circular fit peaks 0.00048 away. The fitted shapes use small positive subsets and disagree about the peak farther away. |
| 2 | `0.710000-0.000114` | 0.006871 | Use the leading published x1 of 0.710000 at x2 = 0.000114, your best x2 setting. This fills a missing corner of the comparison with your two x1 = 0.700520 returns and the published (0.710000, 0.530000) return. The last controlled x2 move cost 0.02009 on one trial. |
| 3 | `0.520000-0.555000-0.450000` | 0.005000 | Keep x1 and x3 fixed and test x2 = 0.555000, just 0.005 from the best published setting. This follows up the near-zero peer return after the x2 = 0.580000 midpoint scored below both endpoints. |
| 4 | `0.407459-0.418058-0.370600-0.420348` | 0.006400 | Hold x1, x2 and x4 at the settings shared by your round 8 point and the leading peer point. Use x3 halfway between their values of 0.377000 and 0.364200. Those two points returned 0.63827 and 0.71215 respectively. |
| 5 | `0.999999-0.999999-0.999999-0.959224` | 0.020412 | Take the GP agreement and vary x4 alone on the upper face. The point clears the established 0.020 spacing rule. Both fits predict below the measured corner, so the strongest reason to use it is to measure the x4 response while exploiting the strongest known region. |
| 6 | `0.436425-0.308543-0.611598-0.707302-0.000000` | 0.057037 | Take the midpoint of the two strongest official inputs, both on x5 = 0. The official GP mean is −0.25757 and the neural ensemble mean is −0.22563, close to the second-best return of −0.22858. The last off-face query scored −0.24135. |
| 7 | `0.000000-0.135108-0.370684-0.306265-0.311144-0.700000` | 0.050264 | Increase only x6 from 0.649736 to 0.700000 at your best point. This moves towards the x6 value of the strongest published peer result while keeping your evidence for the x1 = 0 face. |
| 8 | `0.042902-0.140000-0.170186-0.000000-0.946758-0.586941-0.251883-0.480634` | 0.075002 | Reduce only x2 from 0.215002 to 0.140000 at your incumbent, towards the leading peer's x2 = 0.065000. This keeps the other seven settings and gives a cleaner comparison than another multi-coordinate move. |

All eight strings were checked for dimensions, six-decimal precision, portal
bounds, known and previously submitted points, and the per-function spacing
rule. Distances are Euclidean over all coordinates and include the current
official and pooled observations.

The choices favour measured local comparisons. The GP and neural forecasts
are conditional point estimates with substantial uncertainty. Function 1's
weekly GP and neural models predict signed-log height. Its source fit uses
a separate raw-score hypothesis.

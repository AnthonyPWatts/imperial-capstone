# Function 5 Week 10 review

[Weekly decision review](weekly_review.ipynb) covers the local peak, broader evidence and remaining queries.

Round 9 returned **8223.20909** at `0.976212-0.999999-0.999999-0.999999`. The official best is **8662.405001**. The returned score was 439.196 below that best.

[Official exploration](observation_review.ipynb) · [Official predictions](model_predictions.ipynb) · [Pooled exploration](full-cohort/observation_review.ipynb) · [Pooled predictions](full-cohort/model_predictions.ipynb)

## Initial proposed query

`0.999999-0.999999-0.999999-0.959224`

Take the GP agreement and vary x4 alone on the upper face. The point clears the established 0.020 spacing rule. Both fits predict below the measured corner, so the strongest reason to use it is to measure the x4 response while exploiting the strongest known region.

The saved Week 10 GP mean is 8219.94, with latent SD 162.166. The neural ensemble mean is 8192.44. These forecasts use raw score units. The nearest known input is 0.020412 away.

[Current Week 10 decision](../../../submissions/Week_10/README.md)

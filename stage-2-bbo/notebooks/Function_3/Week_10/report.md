# Function 3 Week 10 review

[Weekly decision review](weekly_review.ipynb) covers the local peak, broader evidence and remaining queries.

Round 9 returned **-0.008109357492** at `0.520000-0.580000-0.450000`. The official best is **-0.004952000822**. The returned score was 0.00315736 below that best.

[Official exploration](official/exploration-review.ipynb) · [Official predictions](official/model_predictions.ipynb) · [Pooled exploration](full-cohort/observation_review.ipynb) · [Pooled predictions](full-cohort/model_predictions.ipynb)

## Initial proposed query

`0.520000-0.555000-0.450000`

Keep x1 and x3 fixed and test x2 = 0.555000, just 0.005 from the best published setting. This follows up the near-zero peer return after the x2 = 0.580000 midpoint scored below both endpoints.

The saved Week 10 GP mean is -0.0163638, with latent SD 0.0144221. The neural ensemble mean is -0.010788. These forecasts use raw score units. The nearest known input is 0.005 away.

[Current Week 10 decision](../../../submissions/Week_10/README.md)

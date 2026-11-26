# Week 9 submitted inputs

**Submission confirmed on 23 November 2026.** The project owner confirmed
that all eight inputs were submitted exactly as saved. The [round 9
results](../round-09-results.json) arrived on 25 November and matched all eight.
Four queries remain per function.
[Submission confirmation](submission-confirmation.json)

The choices used the 239 official observations through round 8 and the
23 November cohort comparison. [Round 8 results](../round-08-results.json)

The [submitted input file](submissions.txt) contains one exact six-decimal input
per line in Function 1–8 order. Use it to match the returned inputs by function
when the round 9 results arrive.

| Function | Submitted input | Reason |
| --- | --- | --- |
| 1 | `0.627434-0.628714` | Refine the small region that produced the two highest official scores, both near 1.997. The distance-weighted 5-NN point is 0.000538 from the nearest official input. |
| 2 | `0.700520-0.530000` | Keep x1 at the best official setting and test x2 = 0.530000 from the highest-scoring published peer return. |
| 3 | `0.520000-0.580000-0.450000` | Test x2 halfway between the best official and peer points, holding their shared x1 and x3 fixed. |
| 4 | `0.391921-0.432079-0.420229-0.427165` | Use the original official GP point near the measured high-score cluster. The Matérn fit gives it a higher predicted mean than its own UCB pick. |
| 5 | `0.976212-0.999999-0.999999-0.999999` | Use the point chosen by both the original official GP and the tuned blend. It reduces x1 alone from the best measured upper corner. |
| 6 | `0.445816-0.323600-0.551644-0.755918-0.072632` | The SVR had the lowest error on the eight later official returns. Its point is close to the best published peer result. |
| 7 | `0.000000-0.145572-0.392685-0.242554-0.346193-0.742781` | The kernel ridge point is near the three best official returns. The original GP, which had lower error on the later returns, also predicts a higher mean here than at its own UCB point. |
| 8 | `0.000000-0.365002-0.020186-0.150000-0.796758-0.436941-0.101883-0.630634` | The Matérn GP had lower error on the later official returns. Its proposed point is closer to the best official measurement than either saved GP corner. |

Functions 2 and 3 are new local probes based on measured high scores. The other
six points come from the saved Week 9 comparison. All eight pass the dimension,
bounds, six-decimal format, spacing and duplicate checks against the official
and cohort snapshots and the Week 1–8 submission files.
[Official-only candidate comparison](../../results/week-09-official-alternatives.md) ·
[Full-cohort review](../../results/week-09-cohort-review.md)

## Function 2 — test x2 at the best official x1

**Submitted input** `0.700520-0.530000`.

The best official return was **0.723057546** at `0.700520-0.000114`.
The highest-scoring published peer return was **0.736237833** at
`0.710000-0.530000`. Both support testing near x1 = 0.70. The selected
Week 9 GPs use x1 alone. The official-only Matérn fit gives the same
predicted mean of **0.66889** at the prepared input and the official best.
The peer result supplies the x2 value to try. Holding x1 exactly fixed
makes this a direct comparison of the two x2 settings.
[Function 2 model review](../../notebooks/Function_2/Week_09/official/model_predictions.ipynb)

**When the return arrives**, subtract the earlier official score of
**0.723057546** from the new score. A positive difference gives a new
official best at the same x1. Published returns have varied at identical
inputs, so a single comparison gives preliminary evidence about x2.

## Function 3 — test the midpoint on x2

**Submitted input** `0.520000-0.580000-0.450000`.

The best official return was **−0.00495200082** at
`0.520000-0.610000-0.450000`. The strongest published peer result was
**−0.000140306768** at `0.520000-0.550000-0.450000`. Both hold x1 at
0.520000 and x3 at 0.450000. The midpoint
`(0.550000 + 0.610000) / 2 = 0.580000` is 0.030000 from each measured setting.

The full-cohort GP predicts **−0.01263** at the midpoint. Along the same line
between x2 = 0.550000 and x2 = 0.610000, its mean peaks near x2 = 0.600.
That predicted gain over the midpoint is about **0.00013**, and both
predicted means fall below the official best. The midpoint gives a balanced
test between the two measured high scores.
[Function 3 full-cohort model review](../../notebooks/Function_3/Week_09/full-cohort/model_predictions.ipynb)

**When the return arrives**, compare it with both endpoint scores. A result
above **−0.00495200082** improves the official best. A result above
**−0.000140306768** also beats the strongest peer result and gives evidence
for a higher-scoring interior setting. A result below both indicates a
weaker midpoint on this trial.

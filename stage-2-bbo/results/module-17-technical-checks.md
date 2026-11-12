# Module 17.2 technical checks

Run on **12 November 2026**. The [F6 worked example](../docs/methodology.md#worked-example-for-function-6-in-round-6)
traces the saved forecasts to the returned score. The checks here replay F5
and compare two search policies on published test functions. The
[run record](module-17-technical-checks.json) contains the settings and
per-query results. See the [reproduction guide](../docs/reproducing-the-analyses.md#module-172-technical-checks).

## F5 Sobol sensitivity

The replay uses the **25 official observations through round 5** and the
saved GP. It ranks candidates outside the incumbent's ±0.15 box by latent
standard deviation, with the existing distance and coordinate tie-breaks.
The 0.02 minimum distance, local pool, explicit corners and six-decimal
precision are fixed across cases. The broad Sobol draws use powers of two.
[Week 6 decision record](../submissions/Week_06/README.md#function-5-switch-from-local-refinement-to-global-information),
[Owen, *On dropping the first Sobol’ point*](https://arxiv.org/abs/2008.08051)

| Setting varied | Cases |
| --- | --- |
| Scramble with 16,384 broad draws | Seeds 11, 42, 43, 73 and 101 |
| Broad draws with seed 42 | 4,096, 8,192, 16,384, 32,768 and 65,536 |

All **nine distinct cases** chose `0.000000-0.000000-0.999999-0.999999`, the
recorded F5 exploration point. Maximum coordinate shift was **0**. The
selected point is an explicit corner present in every pool. This result
measures sensitivity for that saved GP and mixed candidate pool.

## Offline BBOB comparison

The benchmark uses IOHexperimenter to evaluate Sphere, rotated Ellipsoid
and rotated Rastrigin, each at **2D and 8D** on instance 1. Each scenario uses
five paired seeds. Both policies start from the same 16 observations in 2D
or 32 in 8D, and receive **13 further evaluations**. Inputs are rounded to
six decimal places in the unit domain before mapping to the BBOB bounds.
[BBOB suite](https://numbbo.github.io/coco/testsuites/bbob),
[IOHexperimenter](https://github.com/IOHprofiler/IOHexperimenter)

- **GP-UCB** refits the Stage 2 shared-length-scale RBF GP on each round's
  available observations. It applies the capstone's maximisation rule to
  negated BBOB values, scoring the mean plus 1.96 times latent standard
  deviation. The fixed candidate pools use 4,096 broad and 1,024 local Sobol
  draws plus corners. Previously evaluated coordinates are excluded.
- **Sobol** evaluates the next 13 points in the same scrambled sequence
  used for that pair's starting design.

Final gap is the lowest objective value observed minus the known optimum.
The table gives medians across the five seeds, with paired results in
**GP-UCB win / tie / Sobol win** order. Gaps within `1e-12` count as ties.

| BBOB function | Dimensions | GP-UCB median gap | Sobol median gap | Paired results |
| --- | ---: | ---: | ---: | ---: |
| f1 Sphere | 2 | 0.0001075 | 0.4424 | 5 / 0 / 0 |
| f1 Sphere | 8 | 2.131 | 21.17 | 4 / 1 / 0 |
| f10 rotated Ellipsoid | 2 | 32.45 | 3,273 | 5 / 0 / 0 |
| f10 rotated Ellipsoid | 8 | 305,068 | 437,082 | 3 / 1 / 1 |
| f15 rotated Rastrigin | 2 | 13.90 | 22.07 | 3 / 1 / 1 |
| f15 rotated Rastrigin | 8 | 148.6 | 203.9 | 4 / 1 / 0 |

GP-UCB finished closer to the optimum in **24 of 30 pairs**, with four
ties and two Sobol wins. The experiment tests one common GP configuration
on these instances. The per-function model choices and exceptions used
for course queries are outside this comparison. Of 390 GP fits, 27
reported a kernel amplitude near its upper bound and one reported an
L-BFGS convergence warning. The run record retains those warnings.

## Verification

Independent readback replayed all **60 traces and 2,220 evaluations**.
It confirmed identical starts within each pair, the 13-query budgets,
unique six-decimal inputs and the best-gap trajectories. The three
previously recorded F5 sensitivity cases matched their saved choices,
eligible candidate counts and latent standard deviations.

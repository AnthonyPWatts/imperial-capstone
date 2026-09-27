# Source code

[`initial_exploration.py`](initial_exploration.py) contains the observation loader,
short data preview, ranked location map, sampling-gap map and ordered output plots.
The loader checks only row alignment and finite values, without producing a
validation report. Plot labels use output rank, with 1 representing the highest
output; ties share a rank.

Location and gap plots accept an input pair. With more than two inputs, their
distances describe that projection rather than coverage of the complete domain.

For the Function 1 query, `maximin_candidates_2d` scores interior Voronoi
vertices, pairwise bisector intersections with the domain edges, and corners.
Coordinates use the portal's six-decimal precision, with `0.999999` as the
largest allowed value. The unrounded maximum is retained as a check on rounding;
Function 1's winning corner attains it. Candidate selection uses only inputs,
including the input whose output is omitted from the additional scale plot.

[`coverage_optimisation.py`](coverage_optimisation.py) minimises mean distance
after adding one or more points, uniformly weighted over the unit square. It clips bounded
Voronoi cells and integrates Euclidean distance using a boundary-integral formula,
so the objective does not depend on raster resolution. Two seeded differential
evolution runs and refinement from local minima on a coarse candidate grid
cross-check the numerical recommendation. These searches do not constitute a
formal proof of global optimality. Tests compare the integral with known values,
independent quadrature and dense nearest-neighbour grids.

`optimise_coverage_batch` jointly places two or three new points, including their
competition for the same area, and compares independently seeded searches with
refinement from a sequential plan. Duplicate new points do not double-count
coverage. [`batch_coverage_analysis.py`](batch_coverage_analysis.py) runs the
N = 1–3 study, records timings, and produces its tables and comparison figure.

[`function_2_selection.py`](function_2_selection.py) compares the specific
hypotheses of an x1-only response and local two-input features. It combines the
fractional average-coverage gain with squared predictive-mean separation divided
by combined predictive variance. A fixed `D / (1 + D)` transformation bounds the
latter without amplifying weak disagreement. The analysis includes assumed-noise
sensitivity, leave-one-out prediction checks, and a two-input fit with separate
length scales. This is an experimental-design heuristic, not a significance test
or an estimate of expected information gain. Its tests check noise units,
prediction invariance and the score mixture.

[`function_2_risk.py`](function_2_risk.py) evaluates expected simple regret of a
final recommendation on a finite pool. It integrates the maximum of affine
Gaussian posterior-mean updates analytically to calculate knowledge-gradient
values. Joint posterior draws estimate current risk; paired thirteen-query
simulations compare first choices under a fixed continuation policy. Fits and
noise assumptions remain fixed during those simulations. Tests check the normal
integral against quadrature and known values, score invariances and posterior
updates. These conditional results are not a model-free risk bound or a solution
of the optimal thirteen-step policy.

[`multidimensional_exploration.py`](multidimensional_exploration.py) adds the
Function 3 descriptive views: coordinate-pair projections, full-distance cube
slices, input/output plots and a correlation sensitivity table. The cube summary
uses uniform midpoint grids at two resolutions. Tests distinguish full-space
distance from projected distance and check slices against hand calculations.
The extreme output remains in the data and all coverage calculations.

[`multifunction_risk.py`](multifunction_risk.py) extends the finite-set study to
the remaining functions. It reuses the tested KG integral and posterior updates
from Function 2. Its additive Matérn kernel excludes interactions explicitly;
the other two kernels allow local interactions with shared or separate input
length scales. Held-out checks refit each fold. Candidate pools use grids up to
three dimensions and bounded global/local Sobol samples above three, with corners
and one-coordinate perturbations included.

[`risk_refinement.py`](risk_refinement.py) recomputes current risk and query value
on a larger decision pool, applies a declared 95% gain-retention tolerance, and
adds any changed proposal to the paired rollout comparison. It refuses to mix
statistics from different comparison pools. Function 1's retained coverage
proposal is an explicit exception based on weak predictive evidence. Numerical
results include fingerprints of the data, the study code and the shared
knowledge-gradient/posterior calculations in `function_2_risk.py`. Changing any
of these invalidates the base study and, in turn, its larger-pool refinement.
Caches created before the shared code was fingerprinted are recomputed on the
next study run; their fingerprints must not be updated without recalculation.

[`multifunction_report.py`](multifunction_report.py) keeps descriptive prose,
tables and plots outside the notebooks. Pairwise gap maps retain projected
distance semantics, whereas full-space coverage and KG use every coordinate.
Reports keep all supplied observations, original output units and the distinction
between one-query KG and a fixed-policy thirteen-query experiment.

[`report_provenance.py`](report_provenance.py) binds each executed report to its
source notebook, helper code, supplied arrays, requirements and runtime versions.
It also hashes the HTML, executed notebook and any numerical JSON consumed by
the pack builder. The builder validates every report before writing any output,
so it cannot attach current source hashes to stale results. Report manifests
conservatively cover all Stage 2 helpers; edits may require rendering reports
again, while unchanged numerical studies can still use their separate caches.

Later components will cover:

- one surrogate model per function;
- acquisition-function evaluation;
- candidate optimisation and duplicate avoidance;
- portal-format validation;
- progress plots and summary tables.

Prefer a small, testable implementation over a general optimisation framework.

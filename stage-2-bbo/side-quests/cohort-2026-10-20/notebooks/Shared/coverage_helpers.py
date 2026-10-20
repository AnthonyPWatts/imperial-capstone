"""Sample and optimise geometric coverage for the weekly BBO notebooks.

The course inputs occupy [0, 0.999999] in each dimension. Coverage is measured
on a finite set of assessment points, so its maximum distance is an estimate
of the largest gap in the continuous domain.
"""

from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.spatial.distance import cdist
from scipy.stats import qmc


MAX_VALUE = 0.999999

__all__ = [
    "make_coverage_space",
    "optimise_joint_minimax",
    "summarise_coverage_results",
    "select_coverage_batch",
]


def make_coverage_space(n_dimensions, power=12, seed=42):
    """Return 2**power Sobol points followed by all 2**n_dimensions corners.

    This is a sample of the input space, not a Cartesian grid. The same seed
    reproduces the sample. Use a different seed and a larger power for a
    separate assessment sample; the corners are shared by both samples.
    """
    sampler = qmc.Sobol(
        d=n_dimensions,
        scramble=True,
        rng=seed,
    )
    sampled_points = sampler.random_base2(m=power) * MAX_VALUE

    # Only 256 corners at the course's maximum of eight dimensions.
    corners = np.array(
        list(product((0.0, MAX_VALUE), repeat=n_dimensions))
    )
    return np.vstack([sampled_points, corners])


def optimise_joint_minimax(existing, q, coverage_space, seed=42):
    """Choose q points to minimise the worst nearest-point distance.

    ``existing`` and ``coverage_space`` have one column per input dimension.
    All q points are optimised together in those dimensions. The returned
    array has shape (q, n_dimensions). Inputs are not modified.

    The objective uses input locations only, not observed outputs. Assess the
    resulting design on a separate coverage sample. This stochastic search
    does not guarantee the global optimum or the continuous-domain maximum.
    """
    n_dimensions = existing.shape[1]

    distance_to_existing = cdist(
        coverage_space,
        existing,
    ).min(axis=1)

    def objective(flat_proposed):
        proposed = flat_proposed.reshape(q, n_dimensions)

        distance_to_proposed = cdist(
            coverage_space,
            proposed,
        ).min(axis=1)

        nearest_distance = np.minimum(
            distance_to_existing,
            distance_to_proposed,
        )
        return nearest_distance.max()

    result = differential_evolution(
        objective,
        bounds=[(0.0, MAX_VALUE)] * (q * n_dimensions),
        init="sobol",
        seed=seed,
        polish=True,
    )
    return result.x.reshape(q, n_dimensions)


def summarise_coverage_results(results):
    """Copy score rows into a table with relative and marginal improvements.

    Accept the results from compare_coverage_batches or an existing DataFrame.
    Preserve row order and use the single q=0 row as the coverage baseline.
    Marginal improvement compares each row with its predecessor; the first
    row has no predecessor and therefore contains NaN. Inputs are not modified.
    """
    summary = pd.DataFrame(results).copy()
    if not {"q", "max_distance"}.issubset(summary.columns):
        raise ValueError("Results must contain q and max_distance columns.")
    baseline = summary.loc[summary["q"] == 0, "max_distance"]
    if len(baseline) != 1:
        raise ValueError("Results must contain exactly one q=0 baseline row.")
    baseline_distance = baseline.iloc[0]
    summary["relative_max_distance"] = (
        summary["max_distance"] / baseline_distance
        if baseline_distance > 0 else 1.0
    )
    summary["coverage_improvement"] = 1.0 - summary["relative_max_distance"]
    summary["marginal_improvement"] = (
        summary["max_distance"].shift(1) - summary["max_distance"]
    )
    return summary


def select_coverage_batch(query_counts, max_distances, min_elbow_gap=0.05):
    """Select a batch at the largest bend in a normalised coverage curve.

    Compare improvement with the straight line joining the curve's endpoints:
    the largest positive difference is the elbow. This is a simple geometric
    heuristic, not the full Kneedle algorithm or a query-cost calculation.
    Both axes are normalised, so the rule does not depend on distance units.

    Independent stochastic searches can occasionally worsen with larger q.
    Use the best distance achieved so far for selection, leaving the supplied
    measurements unchanged. A bend below min_elbow_gap (default 0.05 of the
    normalised range) is considered weak. Then choose the smallest batch with
    the best measured coverage. If no batch improves coverage, choose q=0.
    Query counts must start at zero and increase strictly.
    """
    query_counts = np.asarray(query_counts, dtype=float)
    max_distances = np.asarray(max_distances, dtype=float)
    if (
        query_counts.ndim != 1
        or max_distances.shape != query_counts.shape
        or query_counts.size == 0
        or not np.isfinite(query_counts).all()
        or not np.isfinite(max_distances).all()
        or query_counts[0] != 0
        or np.any(query_counts != np.floor(query_counts))
        or np.any(np.diff(query_counts) <= 0)
        or np.any(max_distances < 0)
    ):
        raise ValueError("Supply finite distances and increasing integer q values starting at zero.")
    if not np.isfinite(min_elbow_gap) or not 0 <= min_elbow_gap <= 1:
        raise ValueError("min_elbow_gap must be between zero and one.")

    best_so_far = np.minimum.accumulate(max_distances)
    total_improvement = best_so_far[0] - best_so_far[-1]
    tolerance = 16 * np.finfo(float).eps * best_so_far[0]
    if total_improvement <= tolerance:
        return {
            "q": 0,
            "method": "no_improvement",
            "elbow_gap": 0.0,
            "message": "No tested batch improves coverage; selected 0 exploratory queries.",
        }

    query_fraction = query_counts / query_counts[-1]
    improvement_fraction = (best_so_far[0] - best_so_far) / total_improvement
    elbow_gaps = improvement_fraction - query_fraction
    elbow_index = int(np.argmax(elbow_gaps))
    elbow_gap = float(elbow_gaps[elbow_index])
    if 0 < elbow_index < len(query_counts) - 1 and elbow_gap >= min_elbow_gap:
        selected_q = int(query_counts[elbow_index])
        method = "elbow"
        message = (
            f"Selected {selected_q} exploratory queries at the elbow "
            f"(normalised bend: {elbow_gap:.3f})."
        )
    else:
        selected_q = int(query_counts[np.argmin(max_distances)])
        method = "best_coverage"
        message = (
            f"No clear elbow; selected {selected_q} exploratory queries: "
            "the smallest batch with the best measured coverage."
        )
    return {
        "q": selected_q,
        "method": method,
        "elbow_gap": elbow_gap,
        "message": message,
    }

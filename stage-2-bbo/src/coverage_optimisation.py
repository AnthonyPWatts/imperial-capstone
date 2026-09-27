"""Optimise average nearest-sample distance over the two-dimensional unit square."""

import numpy as np
from itertools import product
from time import perf_counter
from scipy.ndimage import minimum_filter
from scipy.optimize import differential_evolution, minimize


def _clip_polygon(polygon, normal, offset):
    """Keep the part satisfying normal @ point <= offset."""
    if len(polygon) == 0:
        return polygon
    vertices = []
    start = polygon[-1]
    start_value = float(normal @ start - offset)
    for end in polygon:
        end_value = float(normal @ end - offset)
        start_inside, end_inside = start_value <= 0, end_value <= 0
        if start_inside != end_inside:
            fraction = start_value / (start_value - end_value)
            vertices.append(start + fraction * (end - start))
        if end_inside:
            vertices.append(end)
        start, start_value = end, end_value
    return np.asarray(vertices).reshape(-1, 2)


def _distance_integral(polygon, site):
    """Integrate Euclidean distance to site over a counter-clockwise polygon.

    The divergence of (z - site) * ||z - site|| is 3 * ||z - site||.
    This turns the area integral into closed-form integrals along the edges.
    Signed edge heights also allow the site to lie outside the polygon.
    """
    total = 0.0
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = end - start
        length = float(np.linalg.norm(edge))
        if length == 0:
            continue
        direction = edge / length
        normal = np.array([direction[1], -direction[0]])
        height = float((start - site) @ normal)
        if height == 0:
            continue
        first = float((start - site) @ direction)
        last = first + length

        def primitive(value):
            return 0.5 * (value * np.hypot(value, height)
                          + height**2 * np.arcsinh(value / abs(height)))

        total += height * (primitive(last) - primitive(first)) / 3
    return max(0.0, float(total))


class MeanCoverage2D:
    """Mean distance after adding points, with uniform weight over [0, 1]^2.

    Distance is integrated over bounded Voronoi cells, avoiding a raster
    approximation to the area integral. Arithmetic remains floating-point.
    """

    def __init__(self, inputs):
        self.inputs = np.unique(np.asarray(inputs, dtype=float), axis=0)
        if self.inputs.ndim != 2 or self.inputs.shape[1] != 2 or not len(self.inputs):
            raise ValueError("Coverage optimisation requires non-empty two-dimensional inputs.")
        square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        self.cells = []
        for site in self.inputs:
            polygon = square.copy()
            for other in self.inputs:
                polygon = _clip_polygon(polygon, 2 * (other - site), other @ other - site @ site)
            self.cells.append(polygon)
        self.mean_before = sum(_distance_integral(cell, site)
                               for cell, site in zip(self.cells, self.inputs))

    def __call__(self, point):
        return self.mean_after(np.asarray(point, dtype=float).reshape(-1, 2))

    def mean_after(self, points):
        """Partition overlap between new samples so that improvement counts once."""
        points = np.unique(np.asarray(points, dtype=float).reshape(-1, 2), axis=0)
        improvement = 0.0
        for cell, site in zip(self.cells, self.inputs):
            for point in points:
                # This point must beat both the old sample and every other new point.
                gained = _clip_polygon(cell, 2 * (site - point), site @ site - point @ point)
                for other in points:
                    if not len(gained):
                        break
                    if not np.array_equal(point, other):
                        gained = _clip_polygon(gained, 2 * (other - point),
                                               other @ other - point @ point)
                improvement += (_distance_integral(gained, site)
                                - _distance_integral(gained, point))
        return float(self.mean_before - improvement)


def optimise_average_coverage(data):
    """Search globally, cross-check other basins, and return a rounded candidate.

    Multiple searches provide numerical evidence, not a formal global-optimality
    certificate. Only coordinates enter the coverage objective.
    """
    started = perf_counter()
    if [name for name in data if name.startswith("x")] != ["x1", "x2"]:
        raise ValueError("This coverage search is for two inputs, not a projection of more inputs.")
    objective = MeanCoverage2D(data[["x1", "x2"]].to_numpy())
    bounds = [(0.0, 0.999999)] * 2
    runs = []
    for seed in (7, 19):
        result = differential_evolution(
            objective, bounds, rng=seed, popsize=12, maxiter=150,
            tol=1e-10, atol=1e-12, polish=True,
        )
        if not result.success:
            raise RuntimeError(f"Coverage search did not converge: {result.message}")
        runs.append(result)

    axis = np.linspace(0, 0.999999, 41)
    landscape = np.array([[objective((first, second)) for first in axis] for second in axis])
    minima = np.argwhere(landscape == minimum_filter(landscape, size=3, mode="nearest"))
    starts = [run.x for run in runs] + [np.array([axis[column], axis[row]]) for row, column in minima]
    refined = []
    for start in starts:
        result = minimize(objective, start, method="Nelder-Mead", bounds=bounds,
                          options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 500})
        if not result.success:
            raise RuntimeError(f"Coverage refinement did not converge: {result.message}")
        refined.append(result)
    best = min(refined, key=lambda result: result.fun)
    # Check the neighbouring six-decimal coordinates instead of rounding blindly.
    lower = np.clip(np.floor(best.x * 1e6) / 1e6, 0, 0.999999)
    upper = np.clip(np.ceil(best.x * 1e6) / 1e6, 0, 0.999999)
    neighbours = [np.array([first, second]) for first in (lower[0], upper[0])
                  for second in (lower[1], upper[1])]
    point = min(neighbours, key=objective)
    return {
        "point": point,
        "unrounded_point": best.x,
        "mean_before": objective.mean_before,
        "mean_after": objective(point),
        "reduction": 1 - objective(point) / objective.mean_before,
        "rounding_penalty": objective(point) - best.fun,
        "global_search_points": np.array([run.x for run in runs]),
        "local_results": np.array([[*run.x, run.fun] for run in refined]),
        "candidate_axis": axis,
        "candidate_mean_distances": landscape,
        "elapsed_seconds": perf_counter() - started,
    }


def optimise_coverage_batch(data, count, *, initial_points=None, seeds=(7, 19)):
    """Joint numerical search for a fixed batch, rather than greedy placement.

    The cap of three points keeps this experiment bounded. Permutations describe
    the same batch; the returned coordinates are sorted only for presentation.
    """
    if count not in (1, 2, 3):
        raise ValueError("This bounded coverage experiment supports one to three new points.")
    if [name for name in data if name.startswith("x")] != ["x1", "x2"]:
        raise ValueError("This coverage search is for two inputs, not a projection of more inputs.")
    started = perf_counter()
    objective = MeanCoverage2D(data[["x1", "x2"]].to_numpy())
    bounds = [(0.0, 0.999999)] * (2 * count)
    runs, evaluations = [], 0
    for seed in seeds:
        result = differential_evolution(
            objective, bounds, rng=seed, popsize=12, maxiter=300,
            tol=1e-9, atol=1e-12, polish=True,
        )
        if not result.success:
            raise RuntimeError(f"Batch coverage search did not converge: {result.message}")
        evaluations += result.nfev
        runs.append(result)
    starts = [run.x for run in runs]
    if initial_points is not None:
        initial = np.asarray(initial_points, dtype=float)
        if initial.shape != (count, 2):
            raise ValueError("The comparison batch must contain the requested number of 2D points.")
        starts.append(initial.ravel())
    refined = []
    for start in starts:
        result = minimize(objective, start, method="Nelder-Mead", bounds=bounds,
                          options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 1200})
        if not result.success:
            raise RuntimeError(f"Batch coverage refinement did not converge: {result.message}")
        evaluations += result.nfev
        refined.append(result)
    best = min(refined, key=lambda result: result.fun)
    lower = np.clip(np.floor(best.x * 1e6) / 1e6, 0, 0.999999)
    upper = np.clip(np.ceil(best.x * 1e6) / 1e6, 0, 0.999999)
    neighbours = list(product(*zip(lower, upper)))
    rounded = np.asarray(min(neighbours, key=objective)).reshape(count, 2)
    rounded = rounded[np.lexsort((rounded[:, 1], rounded[:, 0]))]
    mean_after = objective.mean_after(rounded)
    return {
        "points": rounded,
        "mean_before": objective.mean_before,
        "mean_after": mean_after,
        "reduction": 1 - mean_after / objective.mean_before,
        "rounding_penalty": mean_after - best.fun,
        "global_run_means": np.array([run.fun for run in runs]),
        "refined_run_means": np.array([run.fun for run in refined]),
        "search_evaluations": evaluations,
        "elapsed_seconds": perf_counter() - started,
    }

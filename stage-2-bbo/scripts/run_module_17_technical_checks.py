"""Replay the round 6 F5 sensitivity check and run a paired offline BBOB comparison."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path
import warnings

import ioh
import numpy as np
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from review_week_06_queries import (
    F5_LOCAL_RADIUS, F5_MIN_DISTANCE, STAGE, fingerprint, read_review, review_f5,
)
from model_helpers import score_query_candidates
from model_review_helpers import build_review_candidates, make_regression_models


QUERY_BUDGET = 13
INSTANCE = 1
BBOB_FUNCTIONS = (1, 10, 15)
START_COUNTS = {2: 16, 8: 32}
SEEDS = (11, 19, 31, 73, 101)
UCB_WEIGHT = 1.96
CANDIDATE_SETTINGS = dict(broad_power=12, broad_seed=42, local_power=10, local_seed=43,
                          local_radius=.15, precision=6, include_corners=True)
GP_SETTINGS = dict(length_scale=.3, length_scale_bounds=(.03, 2.), amplitude=1.,
                   amplitude_bounds=(.01, 100.), noise_fraction=.10,
                   optimiser_restarts=1, seed=42)


def f5_sensitivity():
    """Vary the broad scramble and power independently, reusing the saved GP."""
    review = read_review(5)
    summary = review["summary"]
    settings = summary["search_settings"]
    seed, power = settings["broad_seed"], settings["broad_power"]
    cases = [(seed, power)]
    cases.extend((other_seed, power) for other_seed in (11, 43, 73, 101))
    cases.extend((seed, other_power) for other_power in (12, 13, 15, 16))
    selected, details, _ = review_f5(review, sensitivity_cases=cases)
    recorded_path = STAGE / "submissions/Week_06/proposals.json"
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))
    original = recorded["function_5_review"]["alternatives"]["global_information_query"]
    if not np.array_equal(selected, original["point"]):
        raise ValueError("The reference F5 replay must match the recorded exploration point")

    for case in details["search_sensitivity"]:
        point = np.asarray([float(part) for part in case["portal_input"].split("-")])
        case["broad_draw_count"] = 2 ** case["broad_power"]
        case["maximum_coordinate_shift"] = float(np.abs(point - selected).max())
    return dict(function=5, query_round=6, official_observations=len(review["data"]),
                snapshot_sha256=summary["snapshot_sha256"], decision_sha256=fingerprint(recorded_path),
                reference_settings=settings, reference_input=original["portal_input"],
                minimum_known_input_distance=F5_MIN_DISTANCE, local_exclusion_radius=F5_LOCAL_RADIUS,
                cases=details["search_sensitivity"])


def make_problem(function, dimensions):
    return ioh.get_problem(function, instance=INSTANCE, dimension=dimensions,
                           problem_class=ioh.ProblemClass.BBOB)


def sobol_schedule(dimensions, start_count, seed):
    total = start_count + QUERY_BUDGET
    # Draw a complete power-of-two pool, then evaluate only the required prefix.
    draw = qmc.Sobol(d=dimensions, scramble=True, rng=seed).random_base2((total - 1).bit_length())
    precision = CANDIDATE_SETTINGS["precision"]
    points = np.clip(draw[:total].round(precision), 0, 1 - 10. ** -precision)
    if len(np.unique(points, axis=0)) != total:
        raise ValueError("The scheduled Sobol inputs repeat at portal precision")
    return points


def evaluate(problem, points):
    lower, upper = np.asarray(problem.bounds.lb), np.asarray(problem.bounds.ub)
    return np.asarray([problem(point) for point in lower + points * (upper - lower)])


def policy_trace(problem, points, values, start_count):
    best_gap = np.minimum.accumulate(values)[start_count - 1:] - problem.optimum.y
    if (problem.state.evaluations != start_count + QUERY_BUDGET
            or len(np.unique(points, axis=0)) != len(points)
            or not np.isfinite(values).all() or (best_gap < 0).any()):
        raise ValueError("A BBOB run has invalid observations or an incorrect evaluation budget")
    precision = CANDIDATE_SETTINGS["precision"]
    return dict(evaluations=problem.state.evaluations,
                queried_inputs=["-".join(f"{value:.{precision}f}" for value in row)
                                for row in points[start_count:]],
                returns=values[start_count:].tolist(), best_gap_by_query=best_gap.tolist())


def benchmark_pair(function, dimensions, start_count, seed):
    gp_problem = make_problem(function, dimensions)
    sobol_problem = make_problem(function, dimensions)
    schedule = sobol_schedule(dimensions, start_count, seed)
    sobol_values = evaluate(sobol_problem, schedule)
    gp_points = schedule[:start_count]
    gp_values = evaluate(gp_problem, gp_points)
    np.testing.assert_array_equal(gp_values, sobol_values[:start_count])

    gp = make_regression_models(dimensions, **GP_SETTINGS)["Gaussian process"]
    columns = [f"x{axis + 1}" for axis in range(dimensions)]
    fit_warnings = Counter()
    for _ in range(QUERY_BUDGET):
        # BBOB minimises. Negation lets the capstone's maximisation rule rank the candidates.
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            gp.fit(gp_points, -gp_values)
        fit_warnings.update(str(item.message) for item in captured)
        pool = build_review_candidates(gp_points, gp_points[np.argmin(gp_values)],
                                       np.empty((0, dimensions)), input_columns=columns,
                                       **CANDIDATE_SETTINGS)
        ranked = score_query_candidates(gp, pool[columns].to_numpy(), weight=UCB_WEIGHT,
                                        input_columns=columns, precision=CANDIDATE_SETTINGS["precision"])
        point = ranked.loc[[0], columns].to_numpy(dtype=float)
        gp_points = np.vstack((gp_points, point))
        gp_values = np.concatenate((gp_values, evaluate(gp_problem, point)))

    return dict(seed=seed, gp_ucb=policy_trace(gp_problem, gp_points, gp_values, start_count),
                sobol=policy_trace(sobol_problem, schedule, sobol_values, start_count),
                gp_fit_warnings=dict(fit_warnings))


def bbob_benchmark():
    scenarios = []
    for function in BBOB_FUNCTIONS:
        for dimensions, start_count in START_COUNTS.items():
            problem = make_problem(function, dimensions)
            runs = [benchmark_pair(function, dimensions, start_count, seed) for seed in SEEDS]
            scenarios.append(dict(function=function, name=problem.meta_data.name,
                                  dimensions=dimensions, initial_observations=start_count,
                                  known_optimum=float(problem.optimum.y), runs=runs))
            print(f"BBOB f{function}, {dimensions}D: {len(runs)} paired runs", flush=True)
    return dict(instance=INSTANCE, new_evaluations_per_run=QUERY_BUDGET, seeds=SEEDS,
                gp_ucb_weight=UCB_WEIGHT, gp_settings=GP_SETTINGS,
                candidate_settings=CANDIDATE_SETTINGS, scenarios=scenarios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STAGE / "results/module-17-technical-checks.json")
    args = parser.parse_args()
    with threadpool_limits(limits=1, user_api="blas"):
        result = dict(dependencies={name: version(name) for name in ("ioh", "numpy", "scipy", "scikit-learn")},
                      f5_sobol_sensitivity=f5_sensitivity(), bbob=bbob_benchmark())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.name}")


if __name__ == "__main__":
    main()

"""Compare official GP and TabPFN on one common pool of fresh portal inputs."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import ndtr
from scipy.stats import qmc, rankdata


PORTAL_MAX = 0.999999
# Same near-incumbent radii as the Week 10 owner-first advisory study.
LOCAL_RADII = {1: .006, 2: .09, 3: .06, 4: .075, 5: .14, 6: .11, 7: .13, 8: .17}
DIAGNOSTIC_WEIGHTS = {"GP EI": .60, "TabPFN mean": .30, "Coverage": .10}


def _candidate_pool(x, y, previous, radius, seed, pool_power, minimum_distance):
    dimension = x.shape[1]
    incumbent = x[np.argmax(y)]
    global_sampler = qmc.Sobol(dimension, scramble=True, seed=seed)
    local_sampler = qmc.Sobol(dimension, scramble=True, seed=seed + 1)
    global_points = global_sampler.random_base2(pool_power - 1)
    local_offsets = 2 * local_sampler.random_base2(pool_power - 1) - 1
    local_points = incumbent + radius * local_offsets
    directions = np.vstack((np.eye(dimension), -np.eye(dimension)))
    arms = np.vstack([incumbent + step * radius * directions for step in (.25, .5, 1., 1.5)])
    sets = (global_points, local_points, arms)
    labels = np.repeat(["Global Sobol", "Local Sobol", "Coordinate arm"], [len(s) for s in sets])
    proposal = np.clip(np.vstack(sets), 0, PORTAL_MAX).round(6)
    points, first = np.unique(proposal, axis=0, return_index=True)

    # The portal sees six decimals. Use its precision for duplicates and spacing.
    known = np.clip(np.vstack((x, previous)), 0, PORTAL_MAX).round(6)
    separation = cKDTree(known).query(points)[0]
    keep = (separation > 0) & (separation + 1e-12 >= minimum_distance)
    return points[keep], labels[first[keep]]


def _prediction_vector(values, count, label):
    values = np.asarray(values, dtype=float)
    if values.shape not in ((count,), (count, 1)) or not np.isfinite(values).all():
        raise ValueError(f"{label} must supply one finite value per candidate")
    return values.reshape(-1)


def _expected_improvement(mean, sd, best):
    improvement = mean - best
    # A noiseless prediction has deterministic positive-part improvement.
    positive_sd = sd > 0
    ei = np.maximum(improvement, 0.)
    with np.errstate(over="ignore"):
        z = improvement[positive_sd] / sd[positive_sd]
        density = np.exp(-.5 * z**2) / np.sqrt(2 * np.pi)
    ei[positive_sd] = improvement[positive_sd] * ndtr(z) + sd[positive_sd] * density
    return np.maximum(ei, 0.)


def rank_candidates(function, x, y, gp_predict, tab_predict, prior_inputs, *,
                    seed=907, pool_power=10, minimum_distance=0, batch_size=128):
    """Score an owner-only candidate pool in raw output units, maximising score.

    The GP callback returns means and predictive *observation* SDs. The TabPFN
    callback returns means and manages its own batching. Coverage is distance to
    the nearest official observation. The combined diagnostic summarises ranks
    within this pool for inspection.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or not x.size or y.shape != (len(x),):
        raise ValueError("Provide a non-empty NxD official array and N outcomes")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((x < 0) | (x >= 1)):
        raise ValueError("Official outcomes must be finite and inputs within [0, 1)")
    if function not in LOCAL_RADII:
        raise ValueError("Function must be between 1 and 8")
    if not isinstance(pool_power, (int, np.integer)) or pool_power < 1:
        raise ValueError("pool_power must be a positive integer")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(minimum_distance) or minimum_distance < 0:
        raise ValueError("minimum_distance must be finite and non-negative")

    previous = np.asarray(prior_inputs, dtype=float) if prior_inputs is not None else np.empty(0)
    if not previous.size:
        previous = np.empty((0, x.shape[1]))
    if previous.ndim != 2 or previous.shape[1] != x.shape[1]:
        raise ValueError("Prior inputs must have the same dimensionality as the official inputs")
    if not np.isfinite(previous).all() or np.any((previous < 0) | (previous >= 1)):
        raise ValueError("Prior inputs must be finite and within [0, 1)")

    points, labels = _candidate_pool(
        x, y, previous, LOCAL_RADII[function], seed, pool_power, minimum_distance
    )
    if not len(points):
        raise ValueError("No candidate meets the portal precision and spacing rules")

    means, sds = [], []
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]
        mean, sd = gp_predict(batch)
        means.append(_prediction_vector(mean, len(batch), "GP mean"))
        sds.append(_prediction_vector(sd, len(batch), "GP observation SD"))
    gp_mean, gp_sd = np.concatenate(means), np.concatenate(sds)
    if np.any(gp_sd < 0):
        raise ValueError("GP observation SD must be non-negative")
    tab_mean = _prediction_vector(tab_predict(points), len(points), "TabPFN mean")
    coverage = cKDTree(x).query(points)[0]

    table = pd.DataFrame(points, columns=[f"x{i + 1}" for i in range(x.shape[1])])
    table.insert(0, "Portal input", ["-".join(f"{value:.6f}" for value in point) for point in points])
    table["Candidate source"] = labels
    table["GP mean"] = gp_mean
    table["GP observation SD"] = gp_sd
    table["GP EI"] = _expected_improvement(gp_mean, gp_sd, y.max())
    table["TabPFN mean"] = tab_mean
    table["Nearest official distance"] = coverage
    table["Absolute mean disagreement"] = np.abs(gp_mean - tab_mean)
    indicators = {"GP mean": gp_mean, "GP EI": table["GP EI"], "TabPFN mean": tab_mean,
                  "Coverage": coverage, "Disagreement": table["Absolute mean disagreement"]}
    for metric, values in indicators.items():
        table[f"{metric} rank"] = rankdata(-values, method="min")
        table[f"{metric} percentile"] = (
            (rankdata(values, method="average") - 1) / (len(points) - 1) if len(points) > 1 else 1.
        )

    table["Combined diagnostic"] = sum(
        weight * table[f"{metric} percentile"] for metric, weight in DIAGNOSTIC_WEIGHTS.items()
    )
    table["Combined rank"] = rankdata(-table["Combined diagnostic"], method="min")
    table = table.sort_values(["GP EI rank", "Portal input"], ignore_index=True)
    table.attrs["combined_diagnostic_weights"] = DIAGNOSTIC_WEIGHTS.copy()
    return table

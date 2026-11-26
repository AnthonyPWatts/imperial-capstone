
"""Owner-first Bayesian optimisation with two bounded, local residual advisers."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from function_1_source_helpers import fit_gaussian_source, predict_source_log10, source_peak
from weekly_peak_review import PORTAL_MAX, _expected_improvement, _make_model, portal_point, separable_quadratic


# Function-specific starting scales, noise bounds and near-incumbent radii from Week 10.
SETTINGS = {
    1: ([.035, .035], [.04, 1e-6, .4], .006, 0., [(0, 1)]),
    2: ([.12, .45], [.09, .001, .6], .09, 0., [(0, 1)]),
    3: ([.17, .17, .20], [.04, 5e-5, .4], .06, 0., [(1, 2), (0, 1)]),
    4: ([.14] * 4, [.03, 1e-5, .4], .075, 0., [(1, 2), (0, 3)]),
    5: ([.32] * 4, [.001, 1e-8, .1], .14, .02, [(2, 3), (0, 1)]),
    6: ([.23, .23, .22, .22, .15], [.05, 1e-5, .5], .11, 0., [(0, 4), (1, 2)]),
    7: ([.19, .22, .26, .23, .24, .29], [.018, 1e-5, .35], .13, 0., [(0, 2), (3, 5)]),
    8: ([.35] * 8, [.0001, 1e-8, .12], .17, 0., [(1, 4), (0, 3)]),
}

# Frozen comparison inputs from the owner-first review of the Week 10 proposal.
PREVIOUS = (
    "0.629190-0.627878",
    "0.695564-0.569722",
    "0.520000-0.650000-0.450000",
    "0.419096-0.406060-0.386528-0.426466",
    "0.999999-0.999999-0.999999-0.959224",
    "0.528534-0.395701-0.607892-0.792820-0.038603",
    "0.120000-0.135108-0.370684-0.306265-0.311144-0.649736",
    "0.100000-0.150000-0.130000-0.150000-0.800000-0.500000-0.200000-0.600000",
)
WEIGHTS = {
    "Official only": (1., 0., 0.),
    "Current only": (.9, .1, 0.),
    "2024 only": (.9, 0., .1),
    "5% each": (.9, .05, .05),
    "10% each": (.8, .1, .1),
}


def read_snapshot(path, expected_hash=None):
    data_hash = sha256(path.read_bytes()).hexdigest()
    if expected_hash is not None and data_hash != expected_hash:
        raise ValueError(f"Snapshot does not match its intake fingerprint: {path}")
    return pd.read_csv(path, float_precision="round_trip"), data_hash


def load_data(stage, function):
    """Pin the input files, then verify the owner and pooled views agree."""
    stage = Path(stage)
    dimension = len(SETTINGS[function][0])
    columns = [f"x{i + 1}" for i in range(dimension)]
    own_path = stage / f"data/accumulated/function_{function}/observations.csv"
    pool_root = stage / "data/cohort/2026-11-25"
    archive_root = stage / "data/historical-cohort/2024"
    snapshot = json.loads((pool_root / "dataset-audit.json").read_text(encoding="utf-8"))
    pool_record = next(row for row in snapshot["summary"] if row["function"] == function)
    archive_audit = json.loads((archive_root / "intake-audit.json").read_text(encoding="utf-8"))
    archive_record = archive_audit["functions"][str(function)]
    # The pool's hash for official rows is against the CSV at intake. Both
    # snapshots also carry row identifiers, so check the values after parsing.
    owner, own_hash = read_snapshot(own_path, pool_record["official_sha256"])
    pool, pool_hash = read_snapshot(pool_root / f"function_{function}/observations.csv", pool_record["sha256"])
    archive, archive_hash = read_snapshot(archive_root / f"function_{function}.csv", archive_record["sha256"])
    fields = ["observation_id", "source_round", *columns, "y"]
    peer_mask = pool["observation_id"].str.startswith("peer-")
    pd.testing.assert_frame_equal(pool.loc[~peer_mask, fields].reset_index(drop=True), owner[fields])
    valid_archive = archive[columns].ge(0).all(axis=1) & archive[columns].lt(1).all(axis=1)
    eligible_archive = archive.loc[valid_archive].copy()
    trust = {
        "Current intake date": snapshot["intake_date"],
        "Archive revision": archive_audit["commit"],
        "Official SHA-256": own_hash,
        "Current SHA-256": pool_hash,
        "2024 SHA-256": archive_hash,
        "2024 out-of-bounds rows excluded": int((~valid_archive).sum()),
    }
    return owner, pool.loc[peer_mask].copy(), eligible_archive, columns, trust


def coordinate_groups(frame, columns, exclude):
    """Collapse repeated coordinates and retain the within-coordinate variance."""
    rounded = frame[columns].round(6).to_numpy()
    allowed = [tuple(row) not in exclude for row in rounded]
    rows = pd.DataFrame(rounded, columns=columns).assign(y=frame["y"].to_numpy()).loc[allowed]
    return rows.groupby(columns, as_index=False, sort=True).y.agg(
        y="median", repeat_variance="var", evaluations="size"
    ).fillna({"repeat_variance": 0.})


def advisory_groups(owner, peer, archive, columns):
    """Give current data precedence at overlaps to avoid two votes for one location."""
    owner_keys = set(map(tuple, owner[columns].round(6).to_numpy()))
    current = coordinate_groups(peer, columns, owner_keys)
    current_keys = set(map(tuple, current[columns].to_numpy()))
    historical = coordinate_groups(archive, columns, owner_keys | current_keys)
    return current, historical


def source_mean(source, x):
    if source is None:
        return np.zeros(len(x))
    return 10 ** np.clip(predict_source_log10(source, x), -310., np.log10(source["cap"]))


def fit_owner(function, x, y):
    """Use an official-only GP, with the existing raw-source mean for F1."""
    source = None
    if function == 1 and np.count_nonzero(y > 1.) >= 2:
        try:
            proposed = fit_gaussian_source(x, y, positive_count=7, anisotropic=True)
            peak = source_peak(proposed)
        except ValueError:
            peak = None
        if (peak and peak["stationary_point_inside_domain"] and not peak["curvature_at_floor"]
                and peak["predicted_height"] is not None and peak["predicted_height"] <= 1.15 * y.max()
                and np.linalg.norm(np.asarray(peak["point"]) - x[y.argmax()]) <= .015):
            source = proposed | {"cap": 1.15 * float(y.max()), "peak": peak["point"]}
    scales, noise, _, _, _ = SETTINGS[function]
    fitted = _make_model(x, y - source_mean(source, x), scales, "Matérn 3/2", noise)
    return fitted, source


def owner_posterior(owner_model, points):
    """Predict raw output and latent standard deviation from the owner model."""
    (gp, centre, scale, _), source = owner_model
    mean, observation_sd = gp.predict(points, return_std=True)
    latent_var = np.maximum(observation_sd**2 - gp.kernel_.k2.noise_level, 0.)
    return centre + scale * mean + source_mean(source, points), scale * np.sqrt(latent_var)


def fit_advisers(function, owner_model, groups, columns, y):
    """Fit two shrinkage models to residuals on distinct external coordinates."""
    scales = np.asarray(owner_model[0][0].kernel_.k1.k2.length_scale)
    radius = SETTINGS[function][2]
    scales = np.clip(scales, radius / 3., 2 * radius)
    scale = max(float(np.std(y)), 1e-12)
    advisers = []
    for group in groups:
        if group.empty:
            advisers.append(None)
            continue
        x = group[columns].to_numpy()
        mean, _ = owner_posterior(owner_model, x)
        residual = np.clip((group["y"].to_numpy() - mean) / scale, -4., 4.)
        alpha = .35 + group["repeat_variance"].to_numpy() / scale**2
        gp = GaussianProcessRegressor(kernel=Matern(scales, nu=1.5), alpha=alpha, optimizer=None)
        advisers.append((gp.fit(x, residual), scale))
    return advisers


def residuals(advisers, points):
    """Return the two cohort corrections in output units, with zero distant prior."""
    return np.stack([np.zeros(len(points)) if model is None else model[1] * model[0].predict(points)
                     for model in advisers], axis=1)


def owner_candidates(function, x, y, model, prior_inputs, comparison, structural):
    """Produce and freeze an owner-only shortlist with local and broader options."""
    _, _, radius, spacing, _ = SETTINGS[function]
    dimension = x.shape[1]
    incumbent = x[y.argmax()]
    anchors = x[np.argsort(y, kind="stable")[-3:]]
    broad = qmc.Sobol(dimension, scramble=True, seed=907).random_base2(12)
    local = qmc.Sobol(dimension, scramble=True, seed=903).random_base2(9) * 2 - 1
    per_anchor = [anchors[:, None, :] + radius * factor * local for factor in (.35, 1., 2.)]
    arms = incumbent + np.vstack((np.eye(dimension), -np.eye(dimension))) * np.array(
        [.25, .5, 1., 1.5]
    )[:, None, None] * radius
    special = [comparison]
    if structural is not None:
        special.append(structural["point"])
    if model[1] is not None:
        special.append(model[1]["peak"])
    scan = np.vstack([broad, arms.reshape(-1, dimension), special,
                      *(grid.reshape(-1, dimension) for grid in per_anchor)])
    candidates = np.unique(np.clip(scan, 0, PORTAL_MAX).round(6), axis=0)
    min_dist = cdist(candidates, np.vstack([x.round(6), prior_inputs])).min(axis=1)
    candidates = candidates[(min_dist > 0) & (min_dist + 1e-12 >= spacing)]
    mean, sd = owner_posterior(model, candidates)
    noise = model[0][2] * np.sqrt(model[0][0].kernel_.k2.noise_level)
    # Acquisition uses predictive observation SD because feedback can vary.
    sd_obs = np.hypot(sd, noise)
    ei = _expected_improvement(mean, sd_obs, y.max())
    nearby = np.linalg.norm(candidates - incumbent, axis=1) <= radius
    overall = np.argsort(-ei, kind="stable")[:20]
    strata = [np.flatnonzero(mask) for mask in (nearby, ~nearby)]
    strata_best = [rows[np.argsort(-ei[rows], kind="stable")[:8]] for rows in strata]
    # Retain different exploration directions when a smooth GP puts its best
    # values on one boundary or along one coordinate arm.
    arms_best = []
    for axis in range(dimension):
        eligible = np.flatnonzero(np.abs(candidates[:, axis] - incumbent[axis]) >= radius / 2)
        if len(eligible):
            arms_best.append(eligible[np.argmax(ei[eligible])])
    special_ids = [np.flatnonzero(np.all(candidates == np.asarray(point), axis=1)) for point in special]
    keep = np.unique(np.concatenate([overall, *strata_best, *special_ids, np.asarray(arms_best, dtype=int)]))
    return candidates[keep], mean[keep], sd_obs[keep], ei[keep], int(len(candidates))


def rerank(ei, changes, weights):
    """Bound each external contribution to its declared weight."""
    base = ei / max(float(np.max(ei)), np.finfo(float).tiny)
    return weights[0] * base + np.clip(changes, -1., 1.) @ np.asarray(weights[1:])


def replay(function, owner, columns, groups):
    """Hold out official returns by coordinate, refitting owner and adviser models."""
    x, y = owner[columns].to_numpy(), owner["y"].to_numpy()
    records = []
    for i in np.flatnonzero(owner["source_round"].to_numpy() > 0):
        train = np.any(x.round(6) != x[i].round(6), axis=1)
        model = fit_owner(function, x[train], y[train])
        advisers = fit_advisers(function, model, groups, columns, y[train])
        point = x[i:i + 1]
        mean, sd = owner_posterior(model, point)
        current, archive = residuals(advisers, point)[0]
        records.append({"Round": int(owner.iloc[i]["source_round"]), "Actual": y[i],
                        "Official only": mean[0], "Current only": mean[0] + .1 * current,
                        "2024 only": mean[0] + .1 * archive, "10% each": mean[0] + .1 * (current + archive),
                        "Owner latent SD": sd[0]})
    return pd.DataFrame(records)


def review(stage, function, owner, current, archive, columns):
    x, y = owner[columns].to_numpy(), owner["y"].to_numpy()
    model = fit_owner(function, x, y)
    previous = np.array(PREVIOUS[function - 1].split("-"), dtype=float)
    structural = None
    if function == 8:
        fit = separable_quadratic(x, y, owner["source_round"].gt(0))
        if fit["later_max_abs_error"] < 1e-10:
            structural = fit
    prior_inputs = np.array([
        np.array((stage / f"submissions/Week_{week:02d}/submissions.txt")
                 .read_text(encoding="utf-8").splitlines()[function - 1].split("-"), dtype=float)
        for week in range(1, 10)
    ])
    points, mean, sd, ei, pool_size = owner_candidates(function, x, y, model, prior_inputs, previous, structural)
    groups = advisory_groups(owner, current, archive, columns)
    advisers = fit_advisers(function, model, groups, columns, y)
    corrections = residuals(advisers, points)
    adjusted_ei = _expected_improvement(mean[:, None] + corrections, sd[:, None], y.max())
    effects = np.clip((adjusted_ei - ei[:, None]) / max(float(ei.max()), np.finfo(float).tiny), -1., 1.)
    scores = {name: rerank(ei, effects, weights) for name, weights in WEIGHTS.items()}
    winners = {name: portal_point(points[np.argmax(values)]) for name, values in scores.items()}
    chosen = structural["portal_input"] if structural is not None else winners["10% each"]
    radius = SETTINGS[function][2]
    distance = np.linalg.norm(points - x[y.argmax()], axis=1)
    local = np.flatnonzero(distance <= radius)
    local_winner = portal_point(points[local[np.argmax(scores["10% each"][local])]]) if len(local) else None
    if chosen not in {portal_point(point) for point in points}:
        raise ValueError("Selected point is outside the eligible owner shortlist")
    table = pd.DataFrame({
        "Input": [portal_point(point) for point in points], "Owner mean": mean, "Owner return SD": sd,
        "Within owner radius": distance <= radius,
        "Owner EI": ei, "Current shift (y)": corrections[:, 0], "2024 shift (y)": corrections[:, 1],
        "Current score contribution": .1 * effects[:, 0], "2024 score contribution": .1 * effects[:, 1],
        **scores,
    }).sort_values("10% each", ascending=False, kind="stable").reset_index(drop=True)
    comparisons = []
    for name, raw, group in zip(("Current cohort", "2024 archive"), (current, archive), groups, strict=True):
        best = raw.loc[raw["y"].idxmax()]
        point = best[columns].to_numpy(dtype=float)
        gap = np.linalg.norm(points - point, axis=1)
        nearest = int(np.argmin(gap))
        comparisons.append({"Source": name, "Supplied rows": len(raw), "Used rows": int(group.evaluations.sum()),
                            "Used coordinates": len(group), "Best observed": float(best["y"]),
                            "Best observed input": portal_point(point),
                            "Distance to nearest shortlist point": float(gap[nearest]),
                            "Nearest shortlist point": portal_point(points[nearest]),
                            "Nearest point ensemble rank": int(np.flatnonzero(
                                table["Input"].eq(portal_point(points[nearest])).to_numpy())[0] + 1)})
    return {"table": table, "winners": winners, "chosen": chosen, "previous": portal_point(previous),
            "local_winner": local_winner,
            "pool_size": pool_size, "incumbent": x[y.argmax()], "incumbent_y": float(y.max()),
            "model": model, "advisers": advisers, "structural": structural,
            "comparison": pd.DataFrame(comparisons),
            "replay": replay(function, owner, columns, groups)}

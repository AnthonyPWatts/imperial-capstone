"""Finite-domain peak checks and short-horizon acquisition comparisons."""

from itertools import product
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import ndtr, ndtri
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression


PORTAL_MAX = 0.999999
SAMPLE_COUNT = 1024
SEED = 1042


def portal_point(point):
    return "-".join(f"{value:.6f}" for value in point)


def _make_model(x, y, scales, family, noise):
    centre, spread = float(y.mean()), float(y.std()) or 1.0
    if family == "Matérn 3/2":
        shape = Matern(scales, (0.003, 2.0), nu=1.5)
    else:
        shape = RBF(scales, (0.003, 2.0))
    kernel = ConstantKernel(1.0, (0.05, 20.0)) * shape + WhiteKernel(noise[0], noise[1:])
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, random_state=SEED).fit(x, (y - centre) / spread)
    return gp, centre, spread, sum(isinstance(m.message, ConvergenceWarning) for m in messages)


def _latent(fitted, points):
    gp, centre, scale, _ = fitted
    # The fitted WhiteKernel enters the training solve only. The sampled
    # covariance is for the underlying response and excludes evaluation noise.
    signal = gp.kernel_.k1
    cross = signal(gp.X_train_, points)
    projection = solve_triangular(gp.L_, cross, lower=True, check_finite=False)
    mean = centre + scale * (cross.T @ gp.alpha_)
    cov = scale**2 * (signal(points) - projection.T @ projection)
    return mean, (cov + cov.T) * 0.5


def _return_validation(x, y, later, scales, family, noise):
    portal = x.round(6)
    residuals = []
    in_band = []
    for index in np.flatnonzero(later):
        train = np.any(portal != portal[index], axis=1)
        gp, centre, scale, _ = _make_model(x[train], y[train], scales, family, noise)
        mean, sd = gp.predict(x[index:index + 1], return_std=True)
        error = float(y[index] - (centre + scale * mean[0]))
        residuals.append(error)
        in_band.append(abs(error) <= 1.64485362695147 * scale * sd[0])
    e = np.asarray(residuals)
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2))), int(sum(in_band))


def _pool(x, incumbent, radius, fitted):
    d = x.shape[1]
    sample = qmc.Sobol(d + 1, scramble=True, seed=SEED + 1).random_base2(7)
    directions = ndtri(sample[:, 1:])
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    offsets = unit * sample[:, :1] ** (1 / d)
    local = np.vstack([incumbent + radius * offsets,
                       incumbent + np.vstack([np.eye(d), -np.eye(d)]) * radius * 0.98,
                       incumbent + np.vstack([np.eye(d), -np.eye(d)]) * radius / 4])
    broad = qmc.Sobol(d, scramble=True, seed=SEED).random_base2(12) * PORTAL_MAX
    corners = np.asarray(list(product((0.0, PORTAL_MAX), repeat=d)))
    scan = np.unique(np.clip(np.vstack([broad, local, corners]), 0, PORTAL_MAX).round(6), axis=0)
    additions = []
    for gp, _, _, _ in fitted.values():
        mean, sd = gp.predict(scan, return_std=True)
        for score in (mean, mean + 1.96 * sd):
            additions.append(scan[np.argsort(score, kind="stable")[-12:]])
    pts = np.unique(np.clip(np.vstack([x, broad[:256], local, corners, *additions]),
                            0, PORTAL_MAX).round(6), axis=0)
    index = int(np.flatnonzero(np.all(pts == incumbent, axis=1))[0])
    local_mask = np.linalg.norm(pts - incumbent, axis=1) <= radius + 1e-8
    return pts, index, local_mask


def _expected_improvement(mean, sd, best):
    gain = mean - best
    safe = np.maximum(sd, np.finfo(float).tiny)
    z = gain / safe
    return gain * ndtr(z) + sd * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def _short_horizon(points, mean, cov, noise_variance, record, eligible_mask, local, minimum_distance):
    eligible = np.flatnonzero(eligible_mask)
    if len(eligible) < 2:
        raise ValueError("Need two distinct fresh query candidates")
    var = np.maximum(np.diag(cov), 0.0) + noise_variance
    ei = _expected_improvement(mean, np.sqrt(var), record)
    order = eligible[np.argsort(ei[eligible], kind="stable")[::-1]]
    shortlist = list(order[:8])
    for nearby in (local, ~local):
        subset = order[nearby[order]]
        if len(subset):
            shortlist.append(int(subset[0]))
    shortlist.append(int(eligible[np.argmax(var[eligible])]))
    shortlist = np.asarray(list(dict.fromkeys(shortlist)), dtype=int)
    nodes, weights = np.polynomial.hermite.hermgauss(12)
    weights /= np.sqrt(np.pi)
    rows = []
    for index in shortlist:
        distance = np.linalg.norm(points[shortlist] - points[index], axis=1)
        alternatives = shortlist[(shortlist != index) & (distance >= minimum_distance)]
        first = mean[index] + np.sqrt(2 * var[index]) * nodes
        best_after_first = np.maximum(first, record)
        two_step = ei[index]
        if len(alternatives):
            regression = cov[alternatives, index] / var[index]
            conditional_mean = mean[alternatives][None, :] + (first - mean[index])[:, None] * regression
            conditional_sd = np.sqrt(np.maximum(var[alternatives] - regression * cov[alternatives, index], 0))
            second_ei = _expected_improvement(conditional_mean, conditional_sd, best_after_first[:, None])
            two_step = ei[index] + weights @ second_ei.max(axis=1)
        rows.append({"Input": portal_point(points[index]), "Region": "Local" if local[index] else "Elsewhere",
                     "Mean": float(mean[index]), "Observed SD": float(np.sqrt(var[index])),
                     "One-query EI": float(ei[index]), "Two-query gain": float(two_step)})
    return pd.DataFrame(rows).sort_values("Two-query gain", ascending=False)


def peak_review(inputs, target, is_return, *, radius, tolerance, length_scales, noise, minimum_distance=0.0):
    """Compare latent incumbent regret on fixed local and broader candidate sets."""
    x, y = np.asarray(inputs, float), np.asarray(target, float)
    later = np.asarray(is_return, bool)
    if x.ndim != 2 or y.shape != (len(x),) or later.shape != y.shape or not later.any():
        raise ValueError("Peak review requires aligned training arrays and official returns")
    if (not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(x < 0) or np.any(x >= 1)
            or radius <= 0 or tolerance <= 0 or minimum_distance < 0 or len(length_scales) != x.shape[1]):
        raise ValueError("Invalid inputs, local region, tolerance or length scales")
    incumbent = np.clip(x[np.argmax(y)].round(6), 0, PORTAL_MAX)
    families = ("Matérn 3/2", "RBF")
    fitted = {name: _make_model(x, y, length_scales, name, noise) for name in families}
    points, best, local = _pool(x, incumbent, radius, fitted)
    distance = np.linalg.norm(points[:, None, :] - x.round(6)[None, :, :], axis=-1).min(axis=1)
    eligible = (distance > 0) & (distance >= minimum_distance)
    stats, acquisitions = [], {}
    rng = np.random.default_rng(SEED)
    normals = rng.standard_normal((len(points), SAMPLE_COUNT))
    for name, model in fitted.items():
        mean, covariance = _latent(model, points)
        # Stabilise the joint solve without altering the score-unit tolerance.
        jitter = max(float(np.max(np.diag(covariance))), 1.0) * 1e-10
        lower = np.linalg.cholesky(covariance + jitter * np.eye(len(points)))
        draws = mean[:, None] + lower @ normals
        local_gap = draws[local].max(axis=0) - draws[best]
        global_gap = draws.max(axis=0) - draws[best]
        mae, rmse, covered = _return_validation(x, y, later, length_scales, name, noise)
        noise_sd = float(np.sqrt(model[0].kernel_.k2.noise_level) * model[2])
        row = {"Model": name, "Local set within tolerance": float(np.mean(local_gap <= tolerance)),
               "Broader set within tolerance": float(np.mean(global_gap <= tolerance)),
               "Broader gap median": float(np.median(global_gap)),
               "Broader gap 90th pct": float(np.quantile(global_gap, 0.9)),
               "Return MAE": mae, "Return RMSE": rmse, "In 90% observation band": covered,
               "Return holdouts": int(later.sum()), "Fitted noise SD": noise_sd,
               "Fit warnings": int(model[3])}
        stats.append(row)
        acquisitions[name] = _short_horizon(points, mean, covariance, noise_sd**2, y.max(),
                                            eligible, local, minimum_distance)
    return {"summary": pd.DataFrame(stats).set_index("Model"), "acquisitions": acquisitions,
            "incumbent": incumbent, "candidate_count": len(points), "local_count": int(local.sum()),
            "probability_draws": SAMPLE_COUNT}


def separable_quadratic(inputs, target, is_return):
    """Fit only the starters, verify later returns, and maximise the fitted quadratic."""
    x, y = np.asarray(inputs, float), np.asarray(target, float)
    later = np.asarray(is_return, bool)
    design = np.column_stack([x, x**2])
    fit = LinearRegression().fit(design[~later], y[~later])
    linear, quadratic = np.split(fit.coef_, 2)
    if fit.rank_ != design.shape[1] or not np.all(quadratic < 0):
        raise ValueError("The starter fit must be identifiable and strictly concave")
    point = np.clip(-linear / (2 * quadratic), 0, PORTAL_MAX).round(6)
    return {"point": point.tolist(), "portal_input": portal_point(point), "curvature": (-quadratic).tolist(),
            "predicted_maximum": float(fit.predict(np.r_[point, point**2][None])[0]),
            "later_max_abs_error": float(np.max(np.abs(fit.predict(design[later]) - y[later]))),
            "later_rows": int(later.sum()), "starter_rows": int((~later).sum())}

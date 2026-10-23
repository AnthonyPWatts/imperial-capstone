"""Configurable regression reviews and exploratory proposals for any input count."""

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from coverage_helpers import make_coverage_space
from model_helpers import compare_models, make_query_candidates, score_query_candidates


def make_regression_models(n_inputs, *, feature_sets=None, noise_fraction=.10,
                           ridge_alpha=1., kernel_ridge_alpha=.1, kernel_ridge_gamma=1.,
                           polynomial_degree=None, polynomial_ridge_alpha=1.,
                           length_scale=.3, length_scale_bounds=(.03, 2.),
                           amplitude=1., amplitude_bounds=(.01, 100.),
                           optimiser_restarts=3, seed=42):
    """Build declared models; projected GPs select coordinates inside every fold.

    feature_sets maps a model label to ordered input-axis indices. Separate GP
    length scales are created from n_inputs rather than a fixed-size vector.
    Optional polynomial features and scaling are fitted inside each training fold.
    """
    if not isinstance(n_inputs, (int, np.integer)) or n_inputs < 1:
        raise ValueError("n_inputs must be a positive integer")
    if not np.isfinite(noise_fraction) or noise_fraction <= 0:
        raise ValueError("noise_fraction must be finite and positive")
    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(amplitude, amplitude_bounds) * RBF(length_scale, length_scale_bounds),
        alpha=noise_fraction ** 2, normalize_y=True,
        n_restarts_optimizer=optimiser_restarts, random_state=seed,
    )
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear regression": LinearRegression(),
        "Ridge": Ridge(alpha=ridge_alpha),
        "Kernel ridge": KernelRidge(alpha=kernel_ridge_alpha, kernel="rbf", gamma=kernel_ridge_gamma),
    }
    if polynomial_degree is not None:
        if not isinstance(polynomial_degree, (int, np.integer)) or polynomial_degree < 1:
            raise ValueError("polynomial_degree must be a positive integer")
        label = "Quadratic ridge" if polynomial_degree == 2 else "Polynomial ridge"
        models[label] = make_pipeline(PolynomialFeatures(degree=polynomial_degree, include_bias=False),
                                      StandardScaler(), Ridge(alpha=polynomial_ridge_alpha))
    models["Gaussian process"] = gp
    for name, axes in (feature_sets or {}).items():
        axes = list(axes)
        if (name in models or name == "GP: separate length scales" or not axes
                or len(set(axes)) != len(axes)
                or any(axis not in range(n_inputs) for axis in axes)):
            raise ValueError("Projected models need distinct names and valid input axes")
        models[name] = make_pipeline(
            ColumnTransformer([("coordinates", "passthrough", axes)], remainder="drop"), clone(gp),
        )
    models["GP: separate length scales"] = clone(gp).set_params(
        kernel=ConstantKernel(amplitude, amplitude_bounds)
               * RBF([length_scale] * n_inputs, length_scale_bounds),
    )
    return models


def fit_model_review(models, inputs, target, *, gp_names, baseline_name="Mean baseline"):
    """Run held-out comparisons once and select the lowest-RMSE declared GP."""
    comparison, predictions, fitted = compare_models(models, inputs, target)
    baseline = float(comparison.loc[baseline_name, "LOO RMSE"])
    selected_name = comparison.loc[list(gp_names), "LOO RMSE"].idxmin()
    table = comparison.reset_index()
    table["RMSE reduction (%)"] = 100 * (1 - table["LOO RMSE"] / baseline) if baseline else np.nan
    table = table.sort_values("LOO RMSE").reset_index(drop=True)
    selected = fitted[selected_name]
    gp = selected.steps[-1][1] if hasattr(selected, "steps") else selected
    return dict(table=table, predictions=predictions, fitted=fitted,
                selected_name=selected_name, selected_model=selected, fitted_kernel=str(gp.kernel_),
                selected_rmse=float(comparison.loc[selected_name, "LOO RMSE"]),
                baseline_rmse=baseline)


def review_latest_predictions(predictions, target, latest_mask):
    """Score the one latest held-out result, with safe undefined percentages.

    Other-row RMSE excludes that row from scoring only; those rows' training
    folds still include it. No historical forecast is reconstructed here.
    """
    target, latest_mask = np.asarray(target), np.asarray(latest_mask)
    if (target.ndim != 1 or len(target) < 2 or latest_mask.shape != target.shape
            or latest_mask.dtype != bool or latest_mask.sum() != 1):
        raise ValueError("Supply a target vector and exactly one latest observation")
    position = int(np.flatnonzero(latest_mask)[0])
    latest = float(target[position])
    rows = []
    for name, values in predictions.items():
        values = np.asarray(values)
        if values.shape != target.shape or not np.isfinite(values).all():
            raise ValueError("Held-out predictions must match the finite target vector")
        error = float(values[position] - latest)
        total_squared = float(np.sum((values - target) ** 2))
        rows.append({
            "Model": name, "Latest held-out prediction": float(values[position]),
            "Latest error": error,
            "Latest absolute error (%)": 100 * abs(error) / abs(latest) if latest else np.nan,
            "Latest share of squared error (%)": 100 * error ** 2 / total_squared if total_squared else np.nan,
            "Other rows LOO RMSE": root_mean_squared_error(target[~latest_mask], values[~latest_mask]),
        })
    table = pd.DataFrame(rows)
    return table.loc[table["Latest error"].abs().sort_values().index].reset_index(drop=True)


def build_review_candidates(inputs, anchor, coverage_points, *, input_columns,
                            broad_power=14, broad_seed=42, local_power=12,
                            local_seed=43, local_radius=.15, precision=6,
                            include_corners=True):
    """Combine broad, local and supplied coverage points without hidden searches."""
    inputs, anchor = np.asarray(inputs), np.asarray(anchor)
    dimensions = inputs.shape[1]
    if (len(input_columns) != dimensions or anchor.shape != (dimensions,)
            or not np.isfinite(local_radius) or local_radius < 0):
        raise ValueError("Input columns, anchor and local radius must match the input space")
    broad = make_query_candidates(inputs, power=broad_power, seed=broad_seed,
                                  precision=precision, include_corners=include_corners)
    offsets = make_coverage_space(dimensions, power=local_power, seed=local_seed)
    upper = 1 - 10.0 ** -precision
    local = np.clip(anchor + 2 * local_radius * (offsets - .5), 0, upper)
    frames = [pd.DataFrame(points, columns=input_columns).assign(candidate_source=label)
              for points, label in [(broad, "Full input space"), (local, "Near incumbent"),
                                     (coverage_points, "Single coverage query")]]
    candidates = pd.concat(frames, ignore_index=True)
    candidates[list(input_columns)] = candidates[list(input_columns)].round(precision).clip(0, upper)
    candidates = candidates.drop_duplicates(subset=list(input_columns))
    observed = set(map(tuple, inputs.round(precision)))
    keep = [tuple(p) not in observed for p in candidates[list(input_columns)].to_numpy()]
    return candidates.loc[keep].copy()


def score_review_candidates(model, candidates, *, input_columns, weight=1.96, precision=6):
    """Score exact inputs and preserve their broad/local/coverage source labels."""
    scores = score_query_candidates(model, candidates[list(input_columns)].to_numpy(),
                                    weight=weight, input_columns=input_columns, precision=precision)
    return scores.merge(candidates, on=list(input_columns), how="left", validate="one_to_one")


def resolve_unmodelled_coordinates(point, observed, *, modelled_axes,
                                   grid_size=4097, precision=6, seed=44):
    """Use full-input distance to break ties only along unmodelled axes.

    One free coordinate uses an evenly spaced grid, preserving the original
    Function 2 calculation. Multiple free coordinates use a seeded Sobol sample.
    Modelled coordinates remain fixed, and observed inputs are excluded at
    submission precision. Output values are never consulted.
    """
    point, observed = np.asarray(point, dtype=float).copy(), np.asarray(observed, dtype=float)
    axes = list(modelled_axes)
    if (point.shape != (observed.shape[1],) or len(set(axes)) != len(axes)
            or any(axis not in range(len(point)) for axis in axes)
            or not isinstance(grid_size, (int, np.integer)) or grid_size < 2):
        raise ValueError("Supply matching input dimensions, valid modelled axes and grid_size >= 2")
    free = [axis for axis in range(len(point)) if axis not in axes]
    if not free:
        return point
    upper = 1 - 10.0 ** -precision
    if len(free) == 1:
        locations = np.linspace(0, upper, grid_size).round(precision)[:, None]
    else:
        power = int(np.ceil(np.log2(grid_size)))
        locations = qmc.Sobol(d=len(free), scramble=True, seed=seed).random_base2(power)[:grid_size]
        locations = (locations * upper).round(precision)
    alternatives = np.tile(point, (len(locations), 1))
    alternatives[:, free] = locations
    observed_keys = set(map(tuple, observed.round(precision)))
    alternatives = alternatives[[tuple(p) not in observed_keys for p in alternatives]]
    if not len(alternatives):
        raise ValueError("No unobserved tie-break location remains")
    return alternatives[int(np.argmax(cdist(alternatives, observed).min(axis=1)))]


def query_prediction(model, point, inputs, target, *, weight=1.96, precision=6):
    """Validate a proposal and report its prediction separately from its score."""
    point, inputs = np.asarray(point), np.asarray(inputs)
    if (point.shape != (inputs.shape[1],) or not np.isfinite(point).all()
            or not ((point >= 0) & (point < 1)).all()
            or not np.array_equal(point, point.round(precision))
            or tuple(point) in set(map(tuple, inputs.round(precision)))):
        raise ValueError("Proposal must be a new finite input at submission precision")
    mean, std = model.predict(point[None, :], return_std=True)
    try:
        inside = (bool(inputs[:, 0].min() <= point[0] <= inputs[:, 0].max()) if len(point) == 1
                  else bool(Delaunay(inputs).find_simplex(point) >= 0))
    except QhullError:
        inside = None  # A lower-rank sample cannot define a full-dimensional hull.
    return dict(point=point.tolist(), predicted_output=float(mean[0]), posterior_std=float(std[0]),
                ucb=float(mean[0] + weight * std[0]),
                predicted_mean_improvement=float(mean[0] - np.max(target)), inside_sampled_hull=inside,
                portal_input="-".join(f"{value:.{precision}f}" for value in point))


def noise_sensitivity(model, inputs, target, point, *, fractions, fitted_noise_fraction):
    """Refit only changed noise assumptions and score the same proposed point."""
    parameter = f"{model.steps[-1][0]}__alpha" if hasattr(model, "steps") else "alpha"
    rows = []
    for fraction in fractions:
        if not np.isfinite(fraction) or fraction <= 0:
            raise ValueError("Noise fractions must be finite and positive")
        fitted = (model if fraction == fitted_noise_fraction else
                  clone(model).set_params(**{parameter: fraction ** 2}).fit(inputs, target))
        mean, std = fitted.predict(np.asarray(point)[None, :], return_std=True)
        rows.append({"Assumed noise fraction": fraction, "Predicted output": mean[0], "Posterior SD": std[0]})
    return pd.DataFrame(rows)


def weight_sensitivity(scored, inputs, *, input_columns, modelled_axes, weights,
                        grid_size=4097, precision=6, seed=44):
    """Reuse the same scored pool, resolving only the declared unmodelled axes."""
    rows = []
    for weight in weights:
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Exploration weights must be finite and non-negative")
        scores = scored.posterior_mean + weight * scored.posterior_sd
        top = scored.loc[scores.idxmax()]
        point = resolve_unmodelled_coordinates(top[list(input_columns)].to_numpy(dtype=float), inputs,
            modelled_axes=modelled_axes, grid_size=grid_size, precision=precision, seed=seed)
        rows.append({"Weight": weight, **dict(zip(input_columns, point)),
                     "Predicted output": top.posterior_mean, "Posterior SD": top.posterior_sd})
    return pd.DataFrame(rows)

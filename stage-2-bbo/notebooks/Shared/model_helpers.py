"""Shared model comparisons, GP teaching plots and hypothetical query scoring.

These routines work with different input dimensions. No helper evaluates a
course function, changes observations or selects an actual submission.
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def make_gp(length_scale=0.15, noise_fraction=0.10):
    """A fixed RBF assumption, with noise SD relative to training target SD.

    Normalisation happens inside each fit. With an RBF kernel and alpha noise,
    predict(return_std=True) describes latent-function uncertainty; it does not
    add fresh observation noise. No hyperparameter optimisation is performed.
    """
    return GaussianProcessRegressor(
        kernel=RBF(length_scale, length_scale_bounds="fixed"),
        alpha=noise_fraction ** 2, normalize_y=True, optimizer=None,
    )


def make_models():
    """A small declared comparison, without a search for favourable settings."""
    return {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Quadratic ridge": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            StandardScaler(), Ridge(alpha=1.0),
        ),
        "GP: local (l=0.15)": make_gp(0.15),
        "GP: broad (l=0.35)": make_gp(0.35),
    }


def compare_models(models, inputs, target, *, floor_zero=False):
    """Refit each complete estimator per held-out row, then fit all rows.

    floor_zero is explicit post-processing, applied consistently to held-out
    predictions and training-error diagnostics for the modified-target study.
    Caller-owned estimators, inputs and targets are left unchanged.
    """
    inputs, target = np.asarray(inputs), np.asarray(target)
    rows, predictions, fitted = [], {}, {}
    for name, estimator in models.items():
        held_out = cross_val_predict(clone(estimator), inputs, target, cv=LeaveOneOut())
        model = clone(estimator).fit(inputs, target)
        training = model.predict(inputs)
        if floor_zero:
            held_out, training = np.maximum(held_out, 0), np.maximum(training, 0)
        rows.append({"Model": name, "Training MAE": mean_absolute_error(target, training),
                     "LOO MAE": mean_absolute_error(target, held_out),
                     "LOO RMSE": root_mean_squared_error(target, held_out)})
        predictions[name], fitted[name] = held_out, model
    table = pd.DataFrame(rows).set_index("Model")
    baseline = table.loc["Mean baseline", "LOO MAE"]
    table["MAE / baseline"] = table["LOO MAE"] / baseline if baseline else np.nan
    return table, predictions, fitted


def make_query_candidates(observed_inputs, *, power=12, seed=42, precision=6,
                          include_corners=True):
    """A finite input-dimensional pool, excluding observed coordinates.

    Include all allowed extreme corners unless disabled. Six-decimal equality follows the
    portal input convention, so a nearby point that rounds to an observation is
    not treated as a new query. No course evaluations are performed.
    """
    observed = np.asarray(observed_inputs, dtype=float)
    if observed.ndim != 2 or observed.shape[1] < 1 or not np.isfinite(observed).all():
        raise ValueError("Expected finite observations with at least one input column")
    if not isinstance(precision, (int, np.integer)) or precision < 1:
        raise ValueError("precision must be a positive integer")
    dimensions = observed.shape[1]
    upper = 1 - 10.0 ** -precision
    candidates = qmc.Sobol(d=dimensions, scramble=True, seed=seed).random_base2(power)
    if include_corners:
        candidates = np.vstack([candidates, np.array(list(product((0, upper), repeat=dimensions)))])
    candidates = np.unique(np.clip(np.round(candidates, precision), 0, upper), axis=0)
    observed_keys = set(map(tuple, np.round(observed, precision)))
    keep = [tuple(point) not in observed_keys for point in candidates]
    return candidates[keep]


def score_query_candidates(model, candidates, *, weight=1.96, input_columns=None,
                           precision=6):
    """Rank exact candidate coordinates by GP UCB; the maximum is finite-pool only."""
    candidates = np.asarray(candidates, dtype=float)
    if not isinstance(precision, (int, np.integer)) or precision < 1:
        raise ValueError("precision must be a positive integer")
    if (candidates.ndim != 2 or candidates.shape[1] < 1 or len(candidates) == 0
            or not np.isfinite(candidates).all()
            or not ((candidates >= 0) & (candidates <= 1 - 10.0 ** -precision)).all()):
        raise ValueError("Expected non-empty finite candidates within submission bounds")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and non-negative")
    mean, std = model.predict(candidates, return_std=True)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("Expected finite GP means and non-negative standard deviations")
    columns = ([f"x{index + 1}" for index in range(candidates.shape[1])]
               if input_columns is None else list(input_columns))
    if (len(columns) != candidates.shape[1] or len(set(columns)) != len(columns)
            or set(columns) & {"posterior_mean", "posterior_sd", "ucb"}):
        raise ValueError("Supply one distinct input column name per coordinate")
    result = pd.DataFrame(candidates, columns=columns)
    result["posterior_mean"], result["posterior_sd"] = mean, std
    result["ucb"] = mean + weight * std
    return result.sort_values(["ucb", *columns], ascending=[False] + [True] * len(columns)).reset_index(drop=True)


def finish_figure(fig, save_to=None):
    """Save when requested and return a closed figure for one notebook display."""
    if save_to is not None:
        fig.savefig(Path(save_to), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_posterior_curve(ax, coordinates, mean, std, observed_x, observed_y, *, truth=None):
    """Repeat the same mean / conditional band / observations visual vocabulary."""
    ax.fill_between(coordinates, mean - 1.96 * std, mean + 1.96 * std,
                    color="tab:blue", alpha=0.18, label="95% model band (latent function)")
    ax.plot(coordinates, mean, color="tab:blue", linewidth=2, label="GP mean")
    if truth is not None:
        ax.plot(coordinates, truth, "--", color="tab:orange", label="Known toy function")
    ax.scatter(observed_x, observed_y, marker="x", s=65, linewidths=2,
               color="tab:green", zorder=4, label="Observations on this curve")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    return ax


def plot_toy_sequence(save_to=None):
    """An invented 1D example; its queries do not use the course allowance."""
    def truth(x):
        return np.sin(2 * np.pi * x) + 0.35 * np.cos(5 * np.pi * x)

    observed_x = np.array([0.08, 0.46, 0.88, 0.25, 0.67, 0.96, 0.16, 0.37, 0.57, 0.78])
    grid = np.linspace(0, 1, 250)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, count in zip(axes, (3, 6, 10)):
        gp = make_gp(0.18, 0.03).set_params(normalize_y=False)
        gp.fit(observed_x[:count, None], truth(observed_x[:count]))
        mean, std = gp.predict(grid[:, None], return_std=True)
        plot_posterior_curve(ax, grid, mean, std, observed_x[:count], truth(observed_x[:count]), truth=truth(grid))
        ax.set(title=f"After {count} invented observations", xlabel="One toy input", xlim=(0, 1))
    axes[0].set_ylabel("Toy output")
    lower = min(ax.get_ylim()[0] for ax in axes)
    upper = max(ax.get_ylim()[1] for ax in axes)
    axes[0].set_ylim(lower, upper)
    fig.suptitle("Learning the picture: a known toy function, separate from the course black boxes")
    fig.tight_layout()
    return finish_figure(fig, save_to)


def gp_slice(model, inputs, target, fixed_point, varying_axis, *, side=250):
    """Vary one input, holding every other coordinate fixed for a true slice."""
    inputs, target = np.asarray(inputs), np.asarray(target)
    if varying_axis not in range(inputs.shape[1]):
        raise ValueError("varying_axis must name an input coordinate")
    coordinates = np.linspace(0, 0.999999, side)
    points = np.tile(np.asarray(fixed_point, dtype=float), (side, 1))
    points[:, varying_axis] = coordinates
    mean, std = model.predict(points, return_std=True)
    fixed_axes = [axis for axis in range(inputs.shape[1]) if axis != varying_axis]
    on_slice = np.isclose(inputs[:, fixed_axes], np.asarray(fixed_point)[fixed_axes],
                          rtol=0, atol=1e-10).all(axis=1)
    return dict(coordinates=coordinates, points=points, mean=mean, std=std,
                observed_x=inputs[on_slice, varying_axis], observed_y=target[on_slice])


def plot_validation(table, predictions, target, *, target_label="Signed-log height h", save_to=None):
    names = list(table.index)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    positions = np.arange(len(names))
    axes[0].barh(positions - 0.17, table["Training MAE"], height=0.32, label="Training error")
    axes[0].barh(positions + 0.17, table["LOO MAE"], height=0.32, label="Held-out error")
    axes[0].set(yticks=positions, yticklabels=names, xlabel=f"MAE ({target_label})", title="Fitting is easier than predicting a hidden result")
    axes[0].legend(fontsize=9)
    for name in ("Quadratic ridge", "GP: local (l=0.15)"):
        axes[1].scatter(target, predictions[name], s=45, alpha=0.7, label=name)
    values = np.r_[target, predictions["Quadratic ridge"], predictions["GP: local (l=0.15)"]]
    padding = max(float(np.ptp(values)), np.finfo(float).eps) * 0.05
    limits = (values.min() - padding, values.max() + padding)
    axes[1].plot(limits, limits, "k--", linewidth=1, label="Perfect prediction")
    axes[1].set(xlim=limits, ylim=limits, xlabel=f"Observed {target_label}",
                ylabel=f"Held-out predicted {target_label}", title="Every prediction hides its own observation")
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return finish_figure(fig, save_to)

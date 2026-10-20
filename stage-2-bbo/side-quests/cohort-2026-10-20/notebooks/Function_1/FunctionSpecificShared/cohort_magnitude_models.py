"""Side-quest plots comparing signed and magnitude targets on the same inputs."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from model_helpers import finish_figure, gp_slice
from two_input_model_helpers import outline_hull, prediction_grid


MODEL_NAMES = ("Quadratic ridge", "GP: local (l=0.15)", "GP: broad (l=0.35)")


def _observations(ax, inputs, raw_y):
    for sign, marker in ((1, "o"), (-1, "s"), (0, "x")):
        keep = np.sign(raw_y) == sign
        ax.scatter(*inputs[keep].T, marker=marker, color="black", s=23, zorder=4)
    outline_hull(ax, inputs)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x1", ylabel="x2", aspect="equal")


def plot_target_shapes(inputs, raw_y, signed_fitted, magnitude_fitted, *, save_to=None):
    """Use the same grid, hull mask, axes and colour scale in all six panels."""
    grids = [[prediction_grid(models[name], inputs) for models in (signed_fitted, magnitude_fitted)]
             for name in MODEL_NAMES]
    values = np.concatenate([grid["mean"][grid["inside"]] for row in grids for grid in row])
    extent = np.ceil(np.max(np.abs(values)) / 25) * 25
    levels = np.linspace(-extent, extent, 25)
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), layout="constrained")
    for row, name in enumerate(MODEL_NAMES):
        for column, target in enumerate(("Signed h(y)", "Magnitude g(y)")):
            ax, grid = axes[row, column], grids[row][column]
            heights = np.where(grid["inside"], grid["mean"], np.nan)
            fill = ax.contourf(grid["xx"], grid["yy"], heights, levels=levels, cmap="coolwarm")
            contours = ax.contour(grid["xx"], grid["yy"], heights, levels=levels[::2],
                                  colors="black", linewidths=0.5, alpha=0.6)
            ax.clabel(contours, fmt="%.0f", fontsize=8)
            _observations(ax, inputs, raw_y)
            ax.set_title(f"{name}\n{target}")
    fig.colorbar(fill, ax=axes.ravel().tolist(), shrink=0.6, label="Fitted transformed height (shared scale)")
    fig.suptitle("Changing the target: signed response versus magnitude", fontsize=15)
    fig.supxlabel("Black circles: positive readings; squares: negative readings. Dotted line: sampled hull.\n"
                  "Identical colour limits throughout; blank regions have no surrounding observations.", fontsize=10)
    return finish_figure(fig, save_to)


def plot_target_slices(inputs, signed_target, signed_fitted, magnitude_fitted, fixed_point, *, save_to=None):
    """Compare actual 1D slices at the same held coordinate, with common limits."""
    sections = []
    for name in MODEL_NAMES[1:]:
        sections.append([gp_slice(models[name], inputs, target, fixed_point, 0)
                         for models, target in ((signed_fitted, signed_target),
                                                (magnitude_fitted, np.abs(signed_target)))])
    low = min(np.min(s["mean"] - 1.96 * s["std"]) for row in sections for s in row)
    high = max(np.max(s["mean"] + 1.96 * s["std"]) for row in sections for s in row)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True, sharey=True, layout="constrained")
    for ax, name, row in zip(axes, MODEL_NAMES[1:], sections):
        for section, label, colour in zip(row, ("Signed h", "Magnitude g"), ("#2475b0", "#bf681d")):
            ax.plot(section["coordinates"], section["mean"], label=label, color=colour)
            ax.fill_between(section["coordinates"], section["mean"] - 1.96 * section["std"],
                            section["mean"] + 1.96 * section["std"], color=colour, alpha=0.13)
            ax.scatter(section["observed_x"], section["observed_y"], color=colour, marker="x", s=45, zorder=4)
        ax.axvline(fixed_point[0], color="grey", linestyle=":", label="Best signed input's x1")
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set(title=name, xlabel="Vary x1", xlim=(0, 1), ylim=(low, high))
        ax.grid(alpha=0.15)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Predicted transformed height")
    fig.suptitle(f"The same cross-section: hold x2 = {fixed_point[1]:.6f}")
    fig.supxlabel("Shading: conditional 95% latent-function bands, not calibrated detection probabilities.\n"
                  "Each curve uses all 19 observations; they are not projections of off-slice readings.", fontsize=10)
    return finish_figure(fig, save_to)


def plot_activity_search(grid, inputs, raw_y, selected_point, local_point, *, weight=1.96, save_to=None):
    """Show the selected magnitude GP, retaining the alternative local-GP query."""
    mean, std = grid["mean"], grid["std"]
    fields = (mean, std, mean + weight * std)
    labels = ("Predicted activity g", "Model uncertainty: SD", f"Activity UCB: mean + {weight:g} × SD")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.subplots_adjust(left=0.045, right=0.98, bottom=0.24, top=0.86, wspace=0.34)
    for ax, field, label in zip(axes, fields, labels):
        fill = ax.contourf(grid["xx"], grid["yy"], field, levels=16, cmap="viridis")
        fig.colorbar(fill, ax=ax, shrink=0.75)
        _observations(ax, inputs, raw_y)
        ax.scatter(*selected_point, marker="X", color="#ee7788", edgecolor="black", s=110, zorder=5)
        ax.scatter(*local_point, marker="D", color="#cc88ff", edgecolor="black", s=70, zorder=5)
        ax.set_title(label, fontsize=11)
    fig.suptitle("Magnitude exploration: selected GP and sensitivity to a shorter length scale")
    fig.legend(handles=[
        Line2D([], [], marker="X", linestyle="none", color="#bb3355", label="Selected magnitude GP: UCB point"),
        Line2D([], [], marker="D", linestyle="none", color="#8855bb", label="Local magnitude GP: alternative UCB point"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.105), ncols=2, fontsize=10)
    fig.text(0.5, 0.025, "Black circles/squares retain positive/negative signs. Dotted boundary: sampled hull.\n"
             "Colours outside it are extrapolations. Each panel has its own colour scale; both markers are unevaluated.",
             ha="center", fontsize=10)
    return finish_figure(fig, save_to)


def plot_relative_validation(signed, magnitude, *, save_to=None):
    """Normalise each target against its own held-out mean baseline."""
    names = list(signed.index)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    ax.barh(positions - 0.18, signed["MAE / baseline"], height=0.34, label="Signed h")
    ax.barh(positions + 0.18, magnitude["MAE / baseline"], height=0.34, label="Magnitude g")
    ax.axvline(1, color="black", linestyle=":", label="Own-target mean baseline")
    ax.set(yticks=positions, yticklabels=names, xlabel="LOO MAE / own-target baseline MAE (lower is better)",
           title="How predictable is each target relative to its own simple benchmark?")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    return finish_figure(fig, save_to)

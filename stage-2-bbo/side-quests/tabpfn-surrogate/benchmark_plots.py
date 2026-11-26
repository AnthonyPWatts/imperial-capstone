"""Compare fitted surrogate means on a common benchmark slice."""

from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np


def plot_surrogates(
    gp_mean,
    tab_mean,
    axes_pair,
    points,
    observed_x,
    observed_y,
    function_number,
    output_path,
    *,
    anchor=None,
    coordinate_distances=None,
    score_label="Score",
):
    """Save aligned mean and disagreement panels on a complete Cartesian grid.

    The two axes use zero-based indices. Every other feature is fixed to the
    corresponding anchor value, inferred from the grid when omitted. Optional
    distances measure each observation's Euclidean distance in the fixed features.
    """
    points = np.asarray(points, dtype=float)
    observed_x = np.asarray(observed_x, dtype=float)
    observed_y = np.asarray(observed_y, dtype=float).reshape(-1)
    gp_mean = np.asarray(gp_mean, dtype=float).reshape(-1)
    tab_mean = np.asarray(tab_mean, dtype=float).reshape(-1)
    if points.ndim != 2 or not len(points):
        raise ValueError("points must contain a non-empty array of full-dimensional grid inputs")
    n, dimension = points.shape
    if observed_x.shape != (len(observed_y), dimension):
        raise ValueError("observed_x and observed_y must describe aligned, full-dimensional observations")
    if len(gp_mean) != n or len(tab_mean) != n:
        raise ValueError("each surrogate must predict every grid input")
    if any(not np.isfinite(values).all() for values in (points, observed_x, observed_y, gp_mean, tab_mean)):
        raise ValueError("all coordinates and scores must be finite")

    i, j = axes_pair
    if i == j or not (0 <= i < dimension and 0 <= j < dimension):
        raise ValueError("axes_pair must specify two distinct valid feature indices")
    fixed = [k for k in range(dimension) if k not in axes_pair]
    anchor = points[0] if anchor is None else np.asarray(anchor, dtype=float)
    if anchor.shape != (dimension,) or not np.isfinite(anchor).all():
        raise ValueError("anchor must contain a finite value for each feature")
    if not np.allclose(points[:, fixed], anchor[fixed], atol=1e-10, rtol=0):
        raise ValueError("grid inputs must hold all other features at the anchor")
    distances = (
        np.linalg.norm(observed_x[:, fixed] - anchor[fixed], axis=1)
        if coordinate_distances is None
        else np.asarray(coordinate_distances, dtype=float)
    )
    if distances.shape != (len(observed_y),) or not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError("coordinate_distances must contain one finite, non-negative distance per observation")
    # Six-decimal portal inputs agree on the slice to within rounding precision.
    on_slice = distances <= 5e-7
    x = np.unique(points[:, i])
    y = np.unique(points[:, j])
    order = np.lexsort((points[:, i], points[:, j]))
    if (
        len(x) < 2 or len(y) < 2 or len(x) * len(y) != n
        or not np.array_equal(points[order, i], np.tile(x, len(y)))
        or not np.array_equal(points[order, j], np.repeat(y, len(x)))
    ):
        raise ValueError("displayed features must form a complete Cartesian grid")
    shown = (
        (observed_x[:, i] >= x[0]) & (observed_x[:, i] <= x[-1])
        & (observed_x[:, j] >= y[0]) & (observed_x[:, j] <= y[-1])
    )
    measured = shown & on_slice
    projected = shown & ~on_slice
    shape = (len(y), len(x))
    raw = np.concatenate((gp_mean, tab_mean, observed_y[measured]))
    lower, upper = raw.min(), raw.max()
    if lower == upper:
        pad = max(abs(lower) * .01, 1e-12)
        lower, upper = lower - pad, upper + pad
    score_norm = Normalize(vmin=lower, vmax=upper)
    gap = np.abs(gp_mean - tab_mean)
    gap_maximum = gap.max()
    marker_halo = [path_effects.withStroke(linewidth=2.2, foreground="white")]

    fig = plt.figure(figsize=(11.4, 4.65), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=(1, .045, .14), hspace=.12, wspace=.07)
    first = fig.add_subplot(grid[0, 0])
    axes = [first, *(fig.add_subplot(grid[0, k], sharex=first, sharey=first) for k in (1, 2))]
    try:
        for axis, mean, title in zip(axes[:2], (gp_mean, tab_mean), ("Gaussian process mean", "TabPFN mean"), strict=True):
            surface = axis.pcolormesh(
                x, y, mean[order].reshape(shape), cmap="viridis", norm=score_norm, shading="nearest"
            )
            if measured.any():
                axis.scatter(
                    observed_x[measured, i], observed_x[measured, j], c=observed_y[measured],
                    cmap="viridis", norm=score_norm, s=22, linewidths=.65, edgecolors="#1d2428",
                    path_effects=marker_halo, clip_on=False, zorder=3,
                )
            axis.set_title(title, loc="left", fontsize=11)
        difference = axes[2].pcolormesh(
            x, y, gap[order].reshape(shape), cmap="magma", vmin=0,
            vmax=gap_maximum if gap_maximum else 1, shading="nearest",
        )
        if measured.any():
            axes[2].scatter(
                observed_x[measured, i], observed_x[measured, j], s=22, facecolors="white", linewidths=.65,
                edgecolors="#1d2428", clip_on=False, zorder=3,
            )
        axes[2].set_title("Absolute mean difference", loc="left", fontsize=11)
        for axis in axes:
            if projected.any():
                axis.scatter(
                    observed_x[projected, i], observed_x[projected, j], s=17, marker="x", color="#374151",
                    linewidths=.65, path_effects=marker_halo, clip_on=False, zorder=3,
                )
            axis.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), xlabel=f"x{i + 1}", aspect="equal")
        axes[0].set_ylabel(f"x{j + 1}")
        for axis in axes[1:]:
            axis.tick_params(labelleft=False)
        score_bar = fig.colorbar(surface, cax=fig.add_subplot(grid[1, :2]), orientation="horizontal", format="%.3g")
        score_bar.set_label(score_label, fontsize=9)
        delta_bar = fig.colorbar(
            difference, cax=fig.add_subplot(grid[1, 2]), orientation="horizontal",
            ticks=[0] if gap_maximum == 0 else None, format="%.3g",
        )
        delta_bar.set_label(f"Difference in {score_label.lower()}", fontsize=9)
        score_bar.ax.tick_params(labelsize=9)
        delta_bar.ax.tick_params(labelsize=9)
        fig.suptitle(f"Function {function_number} · x{i + 1} and x{j + 1} surrogate comparison", fontsize=14)
        footer = fig.add_subplot(grid[2, :])
        footer.set_axis_off()

        markers = []
        if measured.any():
            markers.append(Line2D(
                [], [], marker="o", linestyle="none", color="#1d2428", markerfacecolor="#798c98",
                label="Observed on slice",
            ))
        if projected.any():
            markers.append(Line2D(
                [], [], marker="x", linestyle="none", color="#374151",
                label="Observation projected from outside the slice",
            ))
        if markers:
            footer.legend(
                handles=markers, loc="upper center" if fixed else "center", borderaxespad=0,
                ncols=len(markers), frameon=False, fontsize=8.5,
            )
        if fixed:
            conditions = ", ".join(f"x{k + 1} = {anchor[k]:.6f}" for k in fixed)
            footer.text(.5, .05, f"Other coordinates fixed at {conditions}.", ha="center", fontsize=8.5)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=170, facecolor="white")
    finally:
        plt.close(fig)
    return path

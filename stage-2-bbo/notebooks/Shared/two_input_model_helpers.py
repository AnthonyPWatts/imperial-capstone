"""Two-input model slices, contour maps and fitted terrains.

Fitting and validation live in model_helpers; these plots use the input square.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull, Delaunay

from model_helpers import finish_figure, gp_slice, plot_posterior_curve
from plot_helpers import PLOT_CONFIG, TERRAIN_COLOURS


def outline_hull(ax, inputs):
    hull = ConvexHull(inputs)
    boundary = inputs[np.r_[hull.vertices, hull.vertices[0]]]
    ax.plot(boundary[:, 0], boundary[:, 1], ":", color="grey", linewidth=1)


def plot_gp_slice(model, inputs, target, fixed_point, varying_axis, *, title, height_limits=None,
                  target_label="Signed-log height h", save_to=None):
    inputs, target = np.asarray(inputs), np.asarray(target)
    section = gp_slice(model, inputs, target, fixed_point, varying_axis)
    fixed_axis = 1 - varying_axis
    fig, (map_ax, curve_ax) = plt.subplots(1, 2, figsize=(12, 4.7), gridspec_kw={"width_ratios": [1, 1.7]})
    map_ax.scatter(inputs[:, 0], inputs[:, 1], color="black", s=30, label="All observed inputs")
    map_ax.plot(section["points"][:, 0], section["points"][:, 1], color="tab:blue", label="Slice path")
    map_ax.scatter(*fixed_point, marker="*", color="gold", edgecolor="black", s=180, zorder=5, label="Best observed input")
    outline_hull(map_ax, inputs)
    map_ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x1", ylabel="x2", title="Where this curve lies in the input square")
    map_ax.set_aspect("equal")
    map_ax.legend(fontsize=8)
    plot_posterior_curve(curve_ax, section["coordinates"], section["mean"], section["std"],
                         section["observed_x"], section["observed_y"])
    curve_ax.set(xlabel=f"Vary x{varying_axis + 1}", ylabel=target_label,
                 title=f"Hold x{fixed_axis + 1} = {fixed_point[fixed_axis]:.4f}", xlim=(0, 1))
    if height_limits is not None:
        curve_ax.set_ylim(height_limits)
    fig.suptitle(title)
    fig.text(0.5, 0.01, "The GP uses every observation. Green crosses match the held coordinate; other measurements stay on the input map.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return finish_figure(fig, save_to)


def prediction_grid(model, inputs, *, side=61, uncertainty=False, floor_zero=False):
    axis = np.linspace(0, 0.999999, side)
    xx, yy = np.meshgrid(axis, axis)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    if uncertainty:
        mean, std = model.predict(points, return_std=True)
    else:
        mean, std = model.predict(points), None
    if floor_zero:
        mean = np.maximum(mean, 0)
    inside = Delaunay(inputs).find_simplex(points) >= 0
    return dict(axis=axis, xx=xx, yy=yy, points=points, mean=mean.reshape(xx.shape),
                std=None if std is None else std.reshape(xx.shape), inside=inside.reshape(xx.shape))


def plot_contour_comparison(inputs, panels, *, title, target_label="Signed-log height h",
                            contour_format="%.0f", save_to=None):
    """Panels are (label, prediction_grid, measured target); colours are shared."""
    values = np.concatenate([np.r_[grid["mean"][grid["inside"]], observed] for _, grid, observed in panels])
    levels = np.linspace(values.min(), values.max(), 15)
    fig, axes = plt.subplots(1, len(panels), figsize=(7 * len(panels), 6), squeeze=False, layout="constrained")
    for ax, (name, grid, observed) in zip(axes[0], panels):
        heights = np.where(grid["inside"], grid["mean"], np.nan)
        fill = ax.contourf(grid["xx"], grid["yy"], heights, levels=levels, cmap="terrain")
        lines = ax.contour(grid["xx"], grid["yy"], heights, levels=levels, colors="black", linewidths=0.5, alpha=0.6)
        ax.clabel(lines, inline=True, fontsize=8, fmt=contour_format)
        ax.scatter(inputs[:, 0], inputs[:, 1], c=observed, cmap="terrain", vmin=levels[0], vmax=levels[-1],
                   s=70, edgecolors="black", linewidths=1.2)
        best = np.argmax(observed)
        ax.scatter(*inputs[best], marker="*", color="gold", edgecolor="black", s=180)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x1", ylabel="x2", title=name)
        ax.set_aspect("equal")
    fig.colorbar(fill, ax=axes[0], shrink=0.85, label=f"{target_label} (shared scale)")
    fig.suptitle(title)
    fig.supxlabel("Background and lines: fitted heights; dot colours: measured target; gold star: best observation.\n"
                  "Blank areas are outside the sampled hull. Contours remain model assumptions.", fontsize=10)
    return finish_figure(fig, save_to)


def plot_gp_maps(grid, inputs, planned_inputs, *, weight=1.96, target_label="h",
                 suggestion=None, title=None, save_to=None):
    mean, std = grid["mean"], grid["std"]
    if std is None:
        raise ValueError("A grid with uncertainty=True is required")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    for ax, field, name in zip(axes, (mean, std, mean + weight * std),
                                (f"GP mean: predicted {target_label}", "GP standard deviation: uncertainty", f"UCB = mean + {weight:g} x SD")):
        fill = ax.contourf(grid["xx"], grid["yy"], field, levels=15, cmap="viridis")
        fig.colorbar(fill, ax=ax, shrink=0.8)
        outline_hull(ax, inputs)
        ax.scatter(inputs[:, 0], inputs[:, 1], color="black", s=15)
        ax.scatter(planned_inputs[0, 0], planned_inputs[0, 1], marker="*", color="gold", edgecolor="black", s=170)
        if suggestion is not None:
            marker = ax.scatter(*suggestion, marker="X", color="magenta", edgecolor="black",
                                s=90, linewidths=0.7, clip_on=False, zorder=5)
            marker.set_in_layout(False)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x1", ylabel="x2", title=name)
        ax.set_aspect("equal")
    if title is not None:
        fig.suptitle(title)
    marker_note = "Magenta X: hypothetical model-led point. " if suggestion is not None else ""
    fig.supxlabel("Gold star: next saved coverage point. " + marker_note + "Dotted boundary: sampled hull.\n"
                  "Coloured regions outside the hull are extrapolations. "
                  "Each panel has its own units and colour scale. The UCB illustration does not replace the coverage plan.", fontsize=10)
    return finish_figure(fig, save_to)


def fitted_terrain(grid, inputs, target, raw_outputs, *, title, target_label="Signed-log height h",
                    number_format=".1f", save_to=None):
    """A small draggable surface, preserving the earlier 0.16 visual relief."""
    heights = np.where(grid["inside"], grid["mean"], np.nan)
    fig = go.Figure(go.Surface(x=grid["axis"], y=grid["axis"], z=heights,
        colorscale=TERRAIN_COLOURS, colorbar=dict(title=target_label), connectgaps=False,
        contours=dict(z=dict(show=True, usecolormap=True)),
        hovertemplate=f"x1=%{{x:.3f}}<br>x2=%{{y:.3f}}<br>Fitted {target_label}=%{{z:{number_format}}}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=inputs[:, 0], y=inputs[:, 1], z=target, mode="markers",
        marker=dict(size=4, color="black"), customdata=raw_outputs, name="Measured target heights",
        hovertemplate=f"x1=%{{x:.4f}}<br>x2=%{{y:.4f}}<br>Original y=%{{customdata:.3e}}<br>Target=%{{z:{number_format}}}<extra></extra>"))
    best = np.argmax(target)
    fig.add_trace(go.Scatter3d(x=[inputs[best, 0]], y=[inputs[best, 1]], z=[target[best]], mode="markers",
        marker=dict(size=7, color="gold", symbol="diamond"), name="Best observed target"))
    fig.update_layout(title=title, height=650, margin=dict(l=10, r=60, t=80, b=40),
        scene=dict(xaxis=dict(title="x1", range=[0, 1]), yaxis=dict(title="x2", range=[0, 1]),
                   zaxis=dict(title=target_label, nticks=3), aspectmode="manual",
                   aspectratio=dict(x=1, y=1, z=0.16), dragmode="orbit"))
    if save_to is not None:
        fig.write_html(Path(save_to), include_plotlyjs=True, config=PLOT_CONFIG)
    return fig


def save_two_input_model_views(model, reference_model, inputs, target, *, title,
                               figures_dir, reference_label="Kernel ridge",
                               grid_side=61, slice_padding=.05):
    """Export the declared two-input contours, common-scale slices and terrain."""
    inputs, target = np.asarray(inputs), np.asarray(target)
    if inputs.ndim != 2 or inputs.shape[1] != 2:
        raise ValueError("These model views require exactly two input coordinates")
    figures_dir = Path(figures_dir)
    anchor = inputs[int(np.argmax(target))].copy()
    grid = prediction_grid(model, inputs, side=grid_side, uncertainty=True)
    reference_grid = prediction_grid(reference_model, inputs, side=grid_side)
    figures = [plot_contour_comparison(inputs, [
        (reference_label, reference_grid, target), ("Fitted GP mean", grid, target),
    ], title=f"{title}: fitted output comparisons", target_label="Output y",
       contour_format="%.2f", save_to=figures_dir / "fitted-output-contours.png")]
    sections = [gp_slice(model, inputs, target, anchor, axis) for axis in range(2)]
    lower = min(np.min(section["mean"] - 1.96 * section["std"]) for section in sections)
    upper = max(np.max(section["mean"] + 1.96 * section["std"]) for section in sections)
    pad = slice_padding * (upper - lower)
    for axis in range(2):
        figures.append(plot_gp_slice(model, inputs, target, anchor, axis,
            title=f"{title}: fitted GP, vary x{axis + 1}", height_limits=(lower-pad, upper+pad),
            target_label="Output y", save_to=figures_dir / f"gp-slice-x{axis + 1}.png"))
    terrain = fitted_terrain(grid, inputs, target, target,
        title=f"{title}: fitted GP mean (raw output)", target_label="Output y",
        number_format=".3f", save_to=figures_dir / "fitted-gp-terrain.html")
    return dict(figures=figures, terrain=terrain, grid=grid, anchor=anchor)


def plot_query_proposal(grid, inputs, anchor, proposal, *, title, weight=1.96,
                        figsize=(15, 5), save_to=None):
    """Locate the incumbent and proposal on distinct mean, uncertainty and UCB maps."""
    fields = [grid["mean"], grid["std"], grid["mean"] + weight * grid["std"]]
    labels = ["GP mean: predicted output y", "GP standard deviation: uncertainty",
              f"UCB = mean + {weight:g} × SD"]
    fig, axes = plt.subplots(1, 3, figsize=figsize, layout="constrained")
    for ax, field, label in zip(axes, fields, labels):
        fill = ax.contourf(grid["xx"], grid["yy"], field, levels=15, cmap="viridis")
        fig.colorbar(fill, ax=ax, shrink=.8)
        outline_hull(ax, inputs)
        ax.scatter(*np.asarray(inputs).T, color="black", s=18, label="Observed inputs")
        ax.scatter(*anchor, marker="*", color="gold", edgecolor="black", s=170, label="Incumbent")
        ax.scatter(*proposal, marker="X", color="magenta", edgecolor="black", s=100,
                   clip_on=False, zorder=5, label="Proposed query")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="x1", ylabel="x2", title=label)
        ax.set_aspect("equal")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title)
    fig.supxlabel("Gold star: incumbent. Magenta X: proposed query. Dotted line: sampled hull. Outside-hull predictions are extrapolations.", fontsize=10)
    return finish_figure(fig, save_to)


def plot_cohort_comparison(paired, own_inputs, peer_inputs, own_proposal, pooled_proposal,
                            *, title, figsize=(13, 5), save_to=None):
    """Paired target errors beside the explicit two-input observation/proposal map."""
    own_inputs, peer_inputs = np.asarray(own_inputs), np.asarray(peer_inputs)
    if own_inputs.ndim != 2 or peer_inputs.ndim != 2 or own_inputs.shape[1] != 2 or peer_inputs.shape[1] != 2:
        raise ValueError("The cohort input map requires exactly two coordinates")
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    positions = np.arange(len(paired))
    axes[0].barh(positions-.18, paired.iloc[:, 1], height=.34, label="Own training")
    axes[0].barh(positions+.18, paired.iloc[:, 2], height=.34, label="Own + peer training")
    axes[0].set(yticks=positions, yticklabels=paired.Model,
                xlabel=f"RMSE on the same {len(own_inputs)} official targets", title="Paired held-out comparison")
    axes[0].legend(fontsize=8)
    axes[1].scatter(*own_inputs.T, color="#4678a0", label="Official observations", s=35)
    axes[1].scatter(*peer_inputs.T, color="#bd7720", marker="D", label="Added peer observations", s=45)
    axes[1].scatter(*own_proposal, color="purple", marker="X", s=110, label="Official provisional query")
    axes[1].scatter(*pooled_proposal, color="crimson", marker="*", s=150, label="Cohort exploratory query")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="x1", ylabel="x2", title="Observed inputs and the two proposals")
    axes[1].set_aspect("equal")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    return finish_figure(fig, save_to)

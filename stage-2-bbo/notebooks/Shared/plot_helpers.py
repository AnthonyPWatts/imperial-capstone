"""Shared exploration plots for the function-by-week notebooks.

Pass a separate plotting DataFrame with a ``rank`` column for point labels;
inputs are not modified. Coordinate plots use a selected pair of inputs in the unit square.
For functions with more than two inputs, sampling gaps show a projection,
not distances in the full input space.
"""

from itertools import combinations
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


__all__ = [
    "PLOT_CONFIG",
    "TERRAIN_COLOURS",
    "plot_observed_terrain",
    "plot_fixed_exploration_projections",
    "plot_ranked_locations",
    "plot_pairwise_ranked_locations",
    "plot_sampling_gaps",
    "plot_fixed_exploration_progression",
    "plot_pairwise_sampling_gaps",
    "plot_sampling_gap_projections",
    "plot_coverage_comparison",
    "plot_pairwise_coverage_comparisons",
    "plot_coverage_tradeoff",
    "plot_output_number_line",
    "plot_output_distribution",
]


TERRAIN_COLOURS = [[0, "#2475b0"], [0.4, "#42b983"],
                   [0.65, "#b6d77a"], [0.85, "#d9c69c"], [1, "#fffde0"]]
PLOT_CONFIG = dict(scrollZoom=True, displaylogo=False, responsive=True)


def plot_observed_terrain(inputs, outputs, *, title, heights=None, height_axis=None,
                          height_description="measured outputs", relief=0.16, save_to=None,
                          width=None, height=700, camera=None, show_reset=False):
    """Join observed two-input heights without fitting or extrapolating a model.

    Supply transformed heights and axis ticks when a function needs a different
    display scale. Hover labels and the best-output marker always use raw outputs.
    Relief changes the visual aspect ratio only. Returns a draggable Plotly figure.
    Supply width for a fixed-size panel, camera for its initial view, and
    show_reset=True for a button restoring that view, including zoom and pan.
    When displaying a fixed-width figure, also pass responsive=False to show().
    """
    import plotly.graph_objects as go
    from scipy.spatial import Delaunay

    inputs, outputs = np.asarray(inputs, dtype=float), np.asarray(outputs, dtype=float)
    heights = outputs if heights is None else np.asarray(heights, dtype=float)
    initial_camera = dict(eye=dict(x=-1.4, y=-1.8, z=1.2)) if camera is None else camera
    if show_reset:
        # Explicit defaults ensure reset also clears any interactive pan or roll.
        initial_camera = {"center": dict(x=0, y=0, z=0), "up": dict(x=0, y=0, z=1),
                          "projection": dict(type="perspective"), **initial_camera}
    triangles = Delaunay(inputs).simplices
    fig = go.Figure(go.Mesh3d(
        x=inputs[:, 0], y=inputs[:, 1], z=heights,
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        intensity=heights, intensitymode="vertex", flatshading=True,
        colorscale=TERRAIN_COLOURS, showscale=False, hoverinfo="skip", name="Joined observations",
    ))
    fig.add_trace(go.Scatter3d(
        x=inputs[:, 0], y=inputs[:, 1], z=heights, mode="markers",
        marker=dict(color="black", size=4), customdata=outputs,
        hovertemplate="x1=%{x:.4f}<br>x2=%{y:.4f}<br>Output y=%{customdata:.3e}<extra></extra>",
        name="Observed outputs",
    ))
    best = np.argmax(outputs)
    fig.add_trace(go.Scatter3d(
        x=[inputs[best, 0]], y=[inputs[best, 1]], z=[heights[best]], mode="markers",
        marker=dict(color="gold", size=7, symbol="diamond", line=dict(color="black", width=2)),
        customdata=[outputs[best]],
        hovertemplate="Best observed output: %{customdata:.3e}<extra></extra>",
        name="Best observed output",
    ))
    fig.update_layout(
        title=dict(text=title + ("<br><sup>Drag to rotate; scroll to zoom; Reset view restores the starting view.</sup>"
                               if show_reset else "<br><sup>Drag to rotate; scroll to zoom; use the toolbar to pan or reset.</sup>")),
        width=width, height=height, margin=dict(l=10, r=10, t=90, b=90),
        scene=dict(
            xaxis=dict(title="x1", range=[0, 1]),
            yaxis=dict(title="x2", range=[0, 1]),
            zaxis=height_axis if height_axis is not None else dict(title="Output y", tickmode="auto", nticks=3),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=relief),
            dragmode="orbit", camera=initial_camera,
        ),
        legend=dict(x=0, y=1),
        annotations=[dict(
            x=0.5, y=-0.08, xref="paper", yref="paper", showarrow=False,
            text=f"Faces linearly join {height_description}; they are not the known function surface."
                 "<br>Blank areas are outside the sampled region. Higher still means a larger output.",
        )],
    )
    if width is not None:
        fig.update_layout(autosize=False)
    if show_reset:
        fig.update_layout(updatemenus=[dict(
            type="buttons", showactive=False, x=1, xanchor="right", y=1.12, yanchor="top",
            buttons=[dict(label="Reset view", method="relayout",
                          args=[{"scene.camera": fig.layout.scene.camera.to_plotly_json()}])],
        )])
    if save_to is not None:
        fig.write_html(Path(save_to), include_plotlyjs=True,
                       config={**PLOT_CONFIG, "responsive": width is None})
    return fig


def _finish(figure, save_to):
    if save_to is not None:
        destination = Path(save_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
    # Returning a closed figure displays it once in a notebook, without duplicates.
    plt.close(figure)
    return figure


def _coordinate_axes(axis, pair):
    axis.set(xlabel=pair[0], ylabel=pair[1], xlim=(0, 1), ylim=(0, 1), aspect="equal")
    axis.set_xticks(np.linspace(0, 1, 6))
    axis.set_yticks(np.linspace(0, 1, 6))


def _input_pairs(input_columns, pairs):
    if len(input_columns) < 2:
        raise ValueError("Pairwise plots need at least two input columns.")
    pairs = (
        list(combinations(input_columns, 2)) if pairs is None
        else [tuple(pair) for pair in pairs]
    )
    if not pairs or any(
        len(pair) != 2 or pair[0] == pair[1]
        or any(name not in input_columns for name in pair)
        for pair in pairs
    ):
        raise ValueError("pairs must select two different input columns per pair.")
    if len({frozenset(pair) for pair in pairs}) != len(pairs):
        raise ValueError("Each unordered input pair must appear only once.")
    return pairs


def _pair_destination(save_to, index, pair):
    if save_to is None:
        return None
    destination = Path(save_to)
    if index > 0:
        destination = destination.with_name(
            f"{destination.stem}-{pair[0]}-{pair[1]}{destination.suffix}"
        )
    return destination


def _projection_note(input_columns, pair):
    return (
        f"\nProjection: {pair[0]} and {pair[1]} ({len(input_columns)}D inputs)"
        if len(input_columns) > 2 else ""
    )


def plot_ranked_locations(data: pd.DataFrame, pair=("x1", "x2"), *, save_to=None):
    """Plot numeric output ranks for an input pair, with 1 denoting the best."""
    figure, axis = plt.subplots(figsize=(7.6, 6.1), layout="constrained")
    normalise = Normalize(vmin=1, vmax=max(2, len(data)))
    colours = plt.get_cmap("viridis_r")
    points = axis.scatter(
        data[pair[0]], data[pair[1]], c=data["rank"], cmap=colours,
        norm=normalise, s=360, edgecolors="#263238", linewidths=0.9, zorder=3,
    )
    for first, second, rank in data[[*pair, "rank"]].itertuples(index=False, name=None):
        red, green, blue, _ = colours(normalise(rank))
        ink = "#172126" if 0.299 * red + 0.587 * green + 0.114 * blue > 0.5 else "white"
        axis.text(first, second, str(rank), ha="center", va="center", color=ink,
                  fontsize=10, fontweight="bold", zorder=4)
    _coordinate_axes(axis, pair)
    axis.set_title("Observed locations, coloured and labelled by output rank")
    colourbar = figure.colorbar(points, ax=axis, shrink=0.82, label="Rank: 1 = highest output")
    colourbar.set_ticks(sorted(data["rank"].unique()))
    colourbar.ax.invert_yaxis()
    return _finish(figure, save_to)


def plot_pairwise_ranked_locations(data: pd.DataFrame, *, input_columns,
                                   pairs=None, save_to=None):
    """Return ranked-location figures for every unordered pair of inputs.

    Select inputs explicitly to exclude output and metadata columns. Supply
    pairs to show a subset. All figures use the same observation ranks and
    rank colour scale; data is not modified.

    Return closed figures keyed by pair; display each value in the notebook.
    Save the first pair to save_to, appending input names for subsequent pairs.
    """
    input_columns = list(input_columns)
    figures = {}
    for index, pair in enumerate(_input_pairs(input_columns, pairs)):
        figure = plot_ranked_locations(data, pair=pair)
        axis = figure.axes[0]
        axis.set_title(axis.get_title() + _projection_note(input_columns, pair))
        figures[pair] = _finish(figure, _pair_destination(save_to, index, pair))
    return figures


def _sampling_gap_surface(data, pair):
    coordinates = data[list(pair)].to_numpy()
    edges = np.linspace(0, 1, 202)
    centres = (edges[:-1] + edges[1:]) / 2
    first, second = np.meshgrid(centres, centres)
    mesh = np.column_stack([first.ravel(), second.ravel()])
    distances = cKDTree(coordinates).query(mesh)[0].reshape(first.shape)
    return coordinates, edges, distances


def plot_sampling_gaps(data: pd.DataFrame, pair=("x1", "x2"), *, save_to=None,
                       candidate=None, central_alternative=None, ax=None, vmax=None,
                       show_points=True, show_labels=True, highlight=None,
                       highlight_label="Highlighted observations", show_legend=True,
                       show_colorbar=True, figsize=(7.6, 6.1)):
    """Plot coverage gaps; ax embeds a subplot and vmax fixes the colour scale.

    With ax supplied, the caller owns the figure; save_to saves that whole figure.
    show_points=False hides all point overlays, including labels and candidates.
    show_labels=False keeps markers but hides their labels; rank is then optional.
    Labelled circles have a key distinguishing observed ranks from P1, P2, ...
    planned samples, which have no observed output yet.
    highlight is an optional boolean mask in data row order (one value per row).
    Selected observations use red stars, retaining ranks when labels are shown.
    Hide the legend or colour bar when a caller supplies a shared figure key.
    figsize controls standalone figures; embedded plots retain their caller's size.
    """
    coordinates, edges, distances = _sampling_gap_surface(data, pair)
    highlighted = np.zeros(len(data), dtype=bool) if highlight is None else np.asarray(highlight)
    if highlighted.shape != (len(data),) or highlighted.dtype != np.dtype(bool):
        raise ValueError("highlight must be a boolean mask with one value per data row")
    if ax is None:
        figure, axis = plt.subplots(figsize=figsize, layout="constrained")
    else:
        axis = ax
        figure = axis.figure
    heatmap = axis.pcolormesh(edges, edges, distances, cmap="viridis", vmin=0, vmax=vmax,
                            shading="flat", rasterized=True)
    if show_points:
        axis.scatter(*coordinates[~highlighted].T, s=310, facecolors="white", edgecolors="#263238",
                     linewidths=0.9, zorder=3)
        if highlighted.any():
            stars = axis.scatter(*coordinates[highlighted].T, s=650, marker="*", color="tab:red",
                                 edgecolors="#263238", linewidths=0.9, zorder=3, clip_on=False)
            stars.set_in_layout(False)
        if show_labels:
            for (first, second), rank, selected in zip(coordinates, data["rank"], highlighted):
                axis.text(first, second, str(rank), ha="center", va="center",
                          color="white" if selected else "#172126",
                          fontsize=10, fontweight="bold", zorder=4)
    _coordinate_axes(axis, pair)
    projected = len([name for name in data if name.startswith("x")]) > 2
    axis.set_title("Sampling gaps in this two-input projection" if projected else "Sampling gaps")
    handles = []
    if show_points and show_labels:
        planned = data["rank"].astype(str).str.fullmatch(r"P\d+")
        labels = []
        if (~planned).any():
            labels.append("Numbers: observed output rank (1 = highest output)")
        if planned.any():
            labels.append("P1, P2, ...: planned samples (not yet evaluated)")
        handles = [Line2D([], [], marker="o", linestyle="none", markersize=10,
                          markerfacecolor="white", markeredgecolor="#263238", label=label)
                   for label in labels]
    if show_points and highlighted.any():
        handles.append(Line2D([], [], marker="*", linestyle="none", markersize=15,
                              markerfacecolor="tab:red", markeredgecolor="#263238",
                              label=highlight_label))
    if handles and show_legend:
        axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12),
                    frameon=False, fontsize=10)
    if show_points and candidate is not None:
        point = np.asarray(candidate)
        nearest = coordinates[cKDTree(coordinates).query(point)[1]]
        axis.plot(*np.vstack([point, nearest]).T, color="white", linestyle="--",
                  linewidth=1.6, zorder=4)
        axis.scatter(*point, s=260, marker="*", color="#ffbf3f", edgecolors="#172126",
                     linewidths=1.2, zorder=6, clip_on=False)
        if show_labels:
            axis.annotate(f"Maximin choice\n({point[0]:.6f}, {point[1]:.6f})", xy=point,
                          xytext=(0.14, 0.96), ha="left", va="top", fontsize=10,
                          bbox={"facecolor": "white", "alpha": 0.95, "edgecolor": "none"},
                          arrowprops={"arrowstyle": "-", "color": "#172126"}, zorder=7)
        axis.set_title("Choosing the next point by coverage")
    if show_points and central_alternative is not None:
        axis.scatter(*central_alternative, s=95, marker="D", facecolors="none",
                     edgecolors="#172126", linewidths=1.7, zorder=5)
        if show_labels:
            axis.annotate("Central alternative", xy=central_alternative, xytext=(0.15, 0.55),
                          fontsize=10, color="#172126",
                          bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
                          arrowprops={"arrowstyle": "-", "color": "#172126"}, zorder=6)
    if show_colorbar:
        figure.colorbar(heatmap, ax=axis, shrink=0.82,
                       label="Distance to nearest observation in this plane")
    if ax is None:
        return _finish(figure, save_to)
    if save_to is not None:
        destination = Path(save_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
    return figure


def plot_fixed_exploration_progression(observations, plan, *, current_week,
                                      first_plan_week=2, save_to=None):
    """Show the fixed input-coverage plan, framing the state before this week's query.

    Earlier stages use recorded coordinates; later stages add the saved inputs.
    This is geometric coverage only: no search, fitted outputs or invented ranks.
    """
    columns = list(plan.columns)
    points = plan.to_numpy(dtype=float)
    if len(columns) != 2 or len(points) == 0:
        raise ValueError("Supply a non-empty two-input plan")
    completed = current_week - first_plan_week
    if completed not in range(len(points) + 1):
        raise ValueError("current_week must identify a stage of this saved plan")
    if (observations["source_round"] >= current_week).any():
        raise ValueError("The observations must precede the current week's query")

    baseline = observations.loc[observations["source_round"] < first_plan_week, columns].copy()
    if baseline.empty:
        raise ValueError("The plan needs its earlier observations as a starting point")
    stages = [baseline]
    for index, point in enumerate(points):
        week = first_plan_week + index
        if index < completed:
            recorded = observations.loc[observations["source_round"] == week, columns]
            if len(recorded) != 1 or not np.allclose(recorded.to_numpy()[0], point, rtol=0, atol=5e-7):
                raise ValueError(f"Week {week}'s recorded input does not match the saved plan")
            addition = recorded.copy()
        else:
            addition = pd.DataFrame([point], columns=columns)
        stages.append(pd.concat([stages[-1], addition], ignore_index=True))

    # Reuse the observed sampling-gap grid and distance calculation.
    surfaces = [_sampling_gap_surface(stage, columns) for stage in stages]
    edges = surfaces[0][1]
    gaps = [surface[2] for surface in surfaces]
    norm = Normalize(vmin=0, vmax=float(gaps[0].max()))

    figure, axes = plt.subplots(1, len(stages), figsize=(14, 3.4),
                                sharex=True, sharey=True, layout="constrained")
    last_week = first_plan_week + len(points) - 1
    for index, (axis, stage, distances) in enumerate(zip(axes, stages, gaps)):
        heatmap = axis.pcolormesh(edges, edges, distances, cmap="viridis", norm=norm,
                                 shading="flat", rasterized=True)
        axis.scatter(stage[columns[0]], stage[columns[1]], s=17, facecolors="white",
                     edgecolors="#263238", linewidths=0.6, zorder=3)
        axis.set(xlim=(0, 1), ylim=(0, 1), xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
        axis.set_aspect("equal")
        axis.tick_params(labelsize=8)
        title = f"Week {first_plan_week + index} start" if index < len(points) else f"After Week {last_week}"
        if index == completed:
            title += "\nCurrent Position"
            for spine in axis.spines.values():
                spine.set_edgecolor("#c2410c")
                spine.set_linewidth(2.5)
        elif index > completed:
            title += "\nPlanned"
        axis.set_title(title, fontsize=10,
                       color="#9a3412" if index == completed else "#172126",
                       fontweight="bold" if index == completed else "normal")
    figure.supxlabel(columns[0], fontsize=10)
    figure.supylabel(columns[1], fontsize=10)
    figure.colorbar(heatmap, ax=axes, shrink=0.8, fraction=0.025, pad=0.02,
                    label="Distance to nearest point")
    if save_to is not None:
        figure.savefig(save_to, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return figure


def plot_pairwise_sampling_gaps(data: pd.DataFrame, *, input_columns,
                               pairs=None, vmax=None, save_to=None):
    """Return sampling-gap figures for every unordered pair of inputs.

    These show nearest-observation distances in each two-input projection,
    rather than distances in the full input space. Select inputs explicitly;
    supply pairs to show a subset. Inputs and rank labels are not modified.

    All figures share a colour scale. With vmax=None, use the largest gap
    across the selected pairs. Return closed figures keyed by pair; display
    each value separately. Save the first pair to save_to, appending both
    input names for subsequent pairs.
    """
    input_columns = list(input_columns)
    pairs = _input_pairs(input_columns, pairs)
    shared_max = vmax if vmax is not None else max(
        _sampling_gap_surface(data, pair)[2].max() for pair in pairs
    )
    figures = {}
    for index, pair in enumerate(pairs):
        figure = plot_sampling_gaps(data, pair=pair, vmax=shared_max)
        figure.axes[0].set_title("Sampling gaps" + _projection_note(input_columns, pair))
        figures[pair] = _finish(figure, _pair_destination(save_to, index, pair))
    return figures


def plot_sampling_gap_projections(data: pd.DataFrame, *, input_columns, pairs=None,
                                  vmax=None, highlight=None,
                                  highlight_label="Highlighted observations", save_to=None):
    """Show a few input-pair projections in one row with a shared legend and scale.

    All unordered input pairs are included unless pairs selects a subset. With
    vmax=None, use the largest projected gap across these pairs. Distances are
    measured in each two-input plane, not in the full input space. Highlight is
    a boolean mask in data row order, as in plot_sampling_gaps.
    Return one closed figure; the existing pairwise helper returns separate figures.
    """
    input_columns = list(input_columns)
    pairs = _input_pairs(input_columns, pairs)
    shared_max = vmax if vmax is not None else max(
        _sampling_gap_surface(data, pair)[2].max() for pair in pairs
    )
    figure, grid = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs) + 0.5, 4.6),
                                sharex=True, sharey=True, squeeze=False, layout="constrained")
    axes = grid[0]
    for index, (axis, pair) in enumerate(zip(axes, pairs)):
        plot_sampling_gaps(data, pair=pair, ax=axis, vmax=shared_max,
                           highlight=highlight, highlight_label=highlight_label,
                           show_legend=index == 0, show_colorbar=False)
        # Keep markers intact where an input lies near the unit-square boundary.
        for points in axis.collections[1:]:
            points.set_clip_on(False)
            points.set_in_layout(False)
        axis.set_title(f"{pair[0]} and {pair[1]}", fontsize=11)
    # Reuse the point renderer's key once for the whole row.
    legend = axes[0].get_legend()
    if legend is not None:
        handles = legend.legend_handles
        labels = [text.get_text() for text in legend.get_texts()]
        legend.remove()
        figure.legend(handles, labels, loc="outside lower center", ncol=2,
                      frameon=False, fontsize=9)
    title = (f"Sampling gaps: 2D projections of {len(input_columns)}D inputs"
             if len(input_columns) > 2 else "Sampling gaps")
    figure.suptitle(title, fontsize=12)
    figure.colorbar(axes[0].collections[0], ax=axes, shrink=0.86, fraction=0.025, pad=0.02,
                    label="Distance to nearest observation\nin this projection")
    return _finish(figure, save_to)


def plot_coverage_comparison(data: pd.DataFrame, proposed: pd.DataFrame, *,
                             pair=None, vmax=None, save_to=None):
    """Compare coverage before/after proposals, with and without point overlays.

    data contains observed inputs and rank labels; proposed contains only the
    input columns, including when it has zero rows. Neither is modified.
    Use the first two proposed columns unless pair selects another input pair.
    Higher-dimensional views are labelled as two-input projections.

    All four panels share one colour scale. vmax=None uses the before panel's
    maximum; supply a value to compare across weeks. Optionally save the figure
    and return it closed for a single inline notebook display.
    """
    input_columns = list(proposed.columns)
    if len(input_columns) < 2:
        raise ValueError("Coverage comparison needs at least two input columns.")
    pair = tuple(input_columns[:2]) if pair is None else tuple(pair)
    if len(pair) != 2 or pair[0] == pair[1] or any(name not in input_columns for name in pair):
        raise ValueError("pair must select two different input columns from proposed.")

    proposed_for_plot = proposed.assign(
        rank=[f"P{i + 1}" for i in range(len(proposed))]
    )
    combined_for_plot = pd.concat(
        [data[[*input_columns, "rank"]], proposed_for_plot],
        ignore_index=True,
    )
    projection_note = _projection_note(input_columns, pair)

    figure, axes = plt.subplots(
        2, 2, figsize=(15, 12), sharex=True, sharey=True, layout="constrained"
    )
    plot_sampling_gaps(data, pair=pair, ax=axes[0, 0], vmax=vmax)
    shared_max = axes[0, 0].collections[0].get_clim()[1]
    plot_sampling_gaps(combined_for_plot, pair=pair, ax=axes[0, 1], vmax=shared_max)
    plot_sampling_gaps(
        data, pair=pair, ax=axes[1, 0], vmax=shared_max, show_points=False
    )
    plot_sampling_gaps(
        combined_for_plot, pair=pair, ax=axes[1, 1], vmax=shared_max,
        show_points=False,
    )

    axes[0, 0].set_title("Before: observed points" + projection_note)
    axes[0, 1].set_title(
        f"After: including {len(proposed)} proposed points" + projection_note
    )
    axes[1, 0].set_title("Before: heatmap only" + projection_note)
    axes[1, 1].set_title("After: heatmap only" + projection_note)
    colour_label = (
        "Nearest-point distance (projection)" if len(input_columns) > 2
        else "Nearest-point distance"
    )
    for axis in axes.flat:
        axis.collections[0].colorbar.set_label(colour_label)

    return _finish(figure, save_to)


def plot_pairwise_coverage_comparisons(data: pd.DataFrame, proposed: pd.DataFrame, *,
                                      pairs=None, vmax=None, save_to=None):
    """Return one four-panel comparison figure for each distinct input pair.

    By default, use every unordered pair of proposed columns: three pairs for
    three inputs, six for four inputs. Supply pairs to select a smaller set.
    Each figure has before/after panels with labels, then heatmaps alone.
    These are projections; the coverage search still uses every input dimension.

    All panels across all figures share one colour scale. With vmax=None, use
    the largest gap in any before projection. Return a dictionary keyed by pair,
    containing closed figures; display each value separately in a notebook.

    The first pair is saved to save_to; additional filenames append both input
    names, e.g. sampling-gaps-comparison-x1-x3.png. Inputs are not modified.
    """
    input_columns = list(proposed.columns)
    pairs = _input_pairs(input_columns, pairs)

    shared_max = vmax if vmax is not None else max(
        _sampling_gap_surface(data, pair)[2].max() for pair in pairs
    )
    figures = {}
    for index, pair in enumerate(pairs):
        figures[pair] = plot_coverage_comparison(
            data, proposed, pair=pair, vmax=shared_max,
            save_to=_pair_destination(save_to, index, pair),
        )
    return figures


def plot_coverage_tradeoff(coverage_results: pd.DataFrame, selected_q, *, save_to=None):
    """Plot relative worst gaps and marginal improvements on a shared query axis.

    Use the table returned by summarise_coverage_results. Both charts mark
    selected_q, aligning the selection vertically. Optionally save the figure
    and return it closed for a single inline notebook display.
    """
    figure, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, layout="constrained")
    axes[0].plot(
        coverage_results["q"],
        coverage_results["relative_max_distance"],
        marker="o",
    )
    axes[0].set_title("Coverage versus number of exploratory queries")
    axes[0].set_ylabel("Worst gap relative to current coverage")

    axes[1].bar(
        coverage_results["q"].iloc[1:],
        coverage_results["marginal_improvement"].iloc[1:],
    )
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].set_ylabel("Reduction from the preceding batch")
    axes[1].set_xlabel("Number of additional exploratory queries")
    axes[1].set_xticks(coverage_results["q"])

    for axis in axes:
        axis.axvline(
            selected_q, linestyle="--", color="tab:red",
            label=f"Selected batch: {selected_q} queries",
        )
    axes[0].legend()
    return _finish(figure, save_to)


def plot_output_number_line(review, *, title=None, output_limits=None, save_to=None):
    """Show measured outputs on one linear baseline, highlighting latest results.

    Accept the review returned by summarise_output_observations. Both signs use
    blue circles for earlier observations; exact zeros use grey diamonds and
    the latest round uses red stars. Values and input row order are unchanged.
    Use this for moderate output ranges; Function 1 has its own mirrored log view.
    output_limits zooms the view without filtering observations or changing ranks.
    Return a closed figure for a single inline notebook display.
    """
    data = review["data"]
    values = data[review["output_column"]].to_numpy(dtype=float)
    latest = np.asarray(review["latest_mask"], dtype=bool)
    zeros = values == 0
    earlier = ~latest & ~zeros
    positions = np.zeros(len(values))
    count = int(latest.sum())
    figure, axis = plt.subplots(figsize=(12, 2.9 + 0.2 * max(0, count - 1)))
    if earlier.any():
        axis.scatter(values[earlier], positions[earlier], s=55, color="tab:blue",
                     label="Previous observations", zorder=3)
    if zeros.any():
        axis.scatter(values[zeros], positions[zeros], s=50, marker="D", color="grey",
                     label="Exact zero", zorder=3)
    if latest.any():
        axis.scatter(values[latest], positions[latest], s=190, marker="*", color="tab:red",
                     edgecolors="#263238", linewidths=0.7,
                     label=f"Latest submission (round {review['latest_round']})", zorder=4)

    span = float(values.max() - values.min())
    padding = 0.06 * span if span > 0 else max(0.05, abs(float(values[0])) * 0.05)
    axis.set(xlim=(values.min() - padding, values.max() + padding), ylim=(-0.45, 0.75),
             yticks=[], xlabel=f"Output {review['output_column']} (linear scale)")
    if output_limits is not None:
        if (len(output_limits) != 2 or not np.isfinite(output_limits).all()
                or output_limits[0] >= output_limits[1]):
            raise ValueError("output_limits must contain two increasing finite values")
        axis.set_xlim(*output_limits)
    if axis.get_xlim()[0] <= 0 <= axis.get_xlim()[1]:
        axis.axvline(0, color="grey", linestyle=":", linewidth=0.8, zorder=1)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useOffset=False)
    axis.yaxis.set_visible(False)
    axis.grid(False)
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.spines["bottom"].set_position(("data", 0))
    axis.spines["bottom"].set_color("#64748b")
    figure.legend(loc="upper left", bbox_to_anchor=(0.06, 0.74), ncol=3,
                  frameon=False, fontsize=9)
    figure.suptitle(title or "Observed outputs on a number line", fontsize=15, y=0.98)
    labels = (data["observation_id"].astype(str).to_numpy() if "observation_id" in data
              else np.array([f"Row {index + 1}" for index in range(len(values))]))
    ranks = pd.Series(values).rank(ascending=False, method="min").to_numpy(dtype=int)
    notes = [f"{labels[index]}: {review['output_column']} = {values[index]:.6g}; "
             f"rank {ranks[index]}/{len(values)} (1 = highest output)"
             for index in np.flatnonzero(latest)]
    figure.text(0.5, 0.85, "\n".join(notes), ha="center", va="top", fontsize=10)
    figure.text(0.5, 0.04, "Better signed outputs run left to right.", ha="center", fontsize=9)
    figure.subplots_adjust(left=0.06, right=0.98, top=0.67, bottom=0.22)
    return _finish(figure, save_to)


def plot_output_distribution(review, *, title=None, save_to=None):
    """Show every output in a cumulative distribution, highlighting new results.

    Accept the review returned by summarise_output_observations. Include its
    coordinates and comparison notes in the saved figure. A second panel shows
    non-zero absolute magnitudes on a log scale when they span at least four
    orders of magnitude; marker shapes preserve their signs. Exact zeros remain
    in the signed distribution and are explicitly counted when omitted on log.
    Return a closed figure for a single inline notebook display.
    """
    data = review["data"]
    values = data[review["output_column"]].to_numpy(dtype=float)
    latest = review["latest_mask"]
    magnitudes = np.abs(values)
    nonzero = magnitudes > 0
    show_magnitudes = (
        nonzero.any()
        and np.ptp(np.log10(magnitudes[nonzero])) >= 4
    )
    panel_count = 2 if show_magnitudes else 1
    caption = "\n".join(fill(line, width=105, break_long_words=False) for line in review["report"].splitlines())
    caption_height = max(1.0, 0.22 * len(caption.splitlines()))
    figure = plt.figure(figsize=(12, 5.2 + caption_height), layout="constrained")
    figure.suptitle(title or "Observed output distribution", fontsize=15)
    grid = figure.add_gridspec(2, panel_count, height_ratios=[caption_height, 4.5])
    summary_axis = figure.add_subplot(grid[0, :])
    summary_axis.set_axis_off()
    summary_axis.text(0, 1, caption, transform=summary_axis.transAxes,
                      va="top", fontsize=9.5, fontfamily="monospace")

    axis = figure.add_subplot(grid[1, 0])
    ordered = np.sort(values)
    percentages = np.arange(1, len(values) + 1) / len(values) * 100
    positions = np.searchsorted(ordered, values, side="right") / len(values) * 100
    axis.step(np.r_[ordered[0], ordered], np.r_[0, percentages], where="post",
              color="#64748b", linewidth=1.3, label=f"All {len(values)} outputs")
    axis.scatter(values[~latest], positions[~latest], s=38, color="tab:blue",
                 label=f"Earlier observations ({review['prior_count']})", zorder=3)
    if latest.any():
        axis.scatter(values[latest], positions[latest], s=170, marker="*",
                     color="tab:red", edgecolors="#263238", linewidths=0.6,
                     label=f"Latest submission (round {review['latest_round']})", zorder=4)
    if ordered[0] <= 0 <= ordered[-1]:
        axis.axvline(0, color="grey", linestyle=":", linewidth=0.8)
    axis.set_title(
        "Signed outputs: where the latest result sits" if latest.any()
        else "Signed outputs: initial observations"
    )
    axis.set_xlabel(f"Observed {review['output_column']} (linear scale)")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4), useOffset=False)
    axis.set_ylabel("Share of outputs at or below this value (%)")
    axis.set_ylim(0, 105)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower right", fontsize=9)

    if show_magnitudes:
        magnitude_axis = figure.add_subplot(grid[1, 1])
        ordered_magnitudes = np.sort(magnitudes[nonzero])
        magnitude_percentages = np.arange(1, nonzero.sum() + 1) / nonzero.sum() * 100
        magnitude_positions = np.searchsorted(
            ordered_magnitudes, magnitudes, side="right"
        ) / nonzero.sum() * 100
        magnitude_axis.step(
            np.r_[ordered_magnitudes[0], ordered_magnitudes],
            np.r_[0, magnitude_percentages], where="post", color="#64748b", linewidth=1.3,
        )
        for label, sign_mask, colour, marker in (
            ("Earlier positive", values > 0, "tab:blue", "o"),
            ("Earlier negative", values < 0, "tab:orange", "^"),
        ):
            mask = sign_mask & ~latest
            if mask.any():
                magnitude_axis.scatter(magnitudes[mask], magnitude_positions[mask],
                                       s=45, color=colour, marker=marker, label=label, zorder=3)
        latest_nonzero = latest & nonzero
        if latest_nonzero.any():
            magnitude_axis.scatter(
                magnitudes[latest_nonzero], magnitude_positions[latest_nonzero],
                s=170, marker="*", color="tab:red", edgecolors="#263238", linewidths=0.6,
                label="Latest submission", zorder=4,
            )
        magnitude_axis.set_xscale("log")
        magnitude_axis.set_title("Absolute magnitudes: small values stay visible")
        magnitude_axis.set_xlabel(
            f"Absolute {review['output_column']} (log scale)\n"
            f"{int((~nonzero).sum())} exact zeros omitted from this panel"
        )
        magnitude_axis.set_ylabel("Share of non-zero magnitudes at or below this value (%)")
        magnitude_axis.set_ylim(0, 105)
        magnitude_axis.set_yticks([0, 25, 50, 75, 100])
        magnitude_axis.grid(axis="y", alpha=0.2)
        magnitude_axis.legend(loc="lower right", fontsize=9)

    return _finish(figure, save_to)


def plot_fixed_exploration_projections(observations, plan, *, current_week,
                                   first_plan_week=2, pairs=None, figsize=None, save_to=None):
    """Show a recorded fixed plan in selected input-pair projections for any input count."""
    columns = list(plan.columns)
    completed = current_week - first_plan_week
    if len(columns) < 2 or plan.empty:
        raise ValueError("Supply a non-empty plan with at least two inputs.")
    if completed not in range(len(plan) + 1):
        raise ValueError("current_week must identify a stage of the saved plan.")
    if (observations["source_round"] >= current_week).any():
        raise ValueError("Observed submissions must precede the current week's query.")
    baseline = observations.loc[observations["source_round"] < first_plan_week, columns].copy()
    if baseline.empty:
        raise ValueError("The plan needs earlier observations as a starting point.")

    # Completed steps use recorded inputs, including their submission precision.
    # Only unevaluated steps use the saved planning coordinates.
    stages = [baseline]
    for index, point in enumerate(plan.to_numpy(dtype=float)):
        week = first_plan_week + index
        if index < completed:
            addition = observations.loc[observations["source_round"] == week, columns].copy()
            if len(addition) != 1 or not np.allclose(addition.to_numpy()[0], point, rtol=0, atol=5e-7):
                raise ValueError(f"Week {week}'s recorded input does not match the saved plan.")
        else:
            addition = pd.DataFrame([point], columns=columns)
        stages.append(pd.concat([stages[-1], addition], ignore_index=True))

    pairs = _input_pairs(columns, pairs)
    figsize = figsize or (14, 3 * len(pairs) + 0.4)
    figure, axes = plt.subplots(len(pairs), len(stages), figsize=figsize,
                                sharex=True, sharey=True, squeeze=False, layout="constrained")
    meshes = []
    for row, pair in enumerate(pairs):
        for column, stage in enumerate(stages):
            axis = axes[row, column]
            plot_sampling_gaps(stage, pair=pair, ax=axis, show_labels=False,
                               show_legend=False, show_colorbar=False)
            meshes.append(axis.collections[0])
            dots = axis.collections[1]
            dots.set_sizes([18])
            dots.set_linewidths([0.6])
            dots.set_clip_on(False)
            dots.set_in_layout(False)
            axis.set(xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
            axis.tick_params(labelsize=8)
            axis.set_ylabel(pair[1] if column == 0 else "")
            if row == 0:
                last_week = first_plan_week + len(plan) - 1
                title = (f"Week {first_plan_week + column} start" if column < len(plan)
                         else f"After Week {last_week}")
                if column == completed:
                    title += "\nCurrent Position"
                elif column > completed:
                    title += "\nPlanned"
                axis.set_title(title, fontsize=10,
                               color="#9a3412" if column == completed else "#172126",
                               fontweight="bold" if column == completed else "normal")
            else:
                axis.set_title("")
            if column == completed:
                for spine in axis.spines.values():
                    spine.set_edgecolor("#c2410c")
                    spine.set_linewidth(2.5)

    # Adding observations cannot increase nearest-point distances, so the
    # largest starting gap across the projections fixes the scale everywhere.
    starting_meshes = meshes[::len(stages)]
    norm = Normalize(vmin=0, vmax=max(float(mesh.get_array().max()) for mesh in starting_meshes))
    for mesh in meshes:
        mesh.set_norm(norm)
    figure.colorbar(meshes[0], ax=axes.ravel().tolist(), shrink=0.85, fraction=0.025, pad=0.02,
                    label="Distance to nearest observation in this projection")
    if save_to is not None:
        figure.savefig(save_to, dpi=160)
    plt.close(figure)
    return figure

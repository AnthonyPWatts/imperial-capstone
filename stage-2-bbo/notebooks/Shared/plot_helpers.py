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
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


__all__ = [
    "plot_ranked_locations",
    "plot_pairwise_ranked_locations",
    "plot_sampling_gaps",
    "plot_pairwise_sampling_gaps",
    "plot_coverage_comparison",
    "plot_pairwise_coverage_comparisons",
    "plot_coverage_tradeoff",
    "plot_output_distribution",
]


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
                       show_points=True, show_labels=True):
    """Plot coverage gaps; ax embeds a subplot and vmax fixes the colour scale.

    With ax supplied, the caller owns the figure; save_to saves that whole figure.
    show_points=False hides all point overlays, including labels and candidates.
    show_labels=False keeps markers but hides their labels; rank is then optional.
    """
    coordinates, edges, distances = _sampling_gap_surface(data, pair)
    if ax is None:
        figure, axis = plt.subplots(figsize=(7.6, 6.1), layout="constrained")
    else:
        axis = ax
        figure = axis.figure
    heatmap = axis.pcolormesh(edges, edges, distances, cmap="viridis", vmin=0, vmax=vmax,
                            shading="flat", rasterized=True)
    if show_points:
        axis.scatter(*coordinates.T, s=310, facecolors="white", edgecolors="#263238",
                     linewidths=0.9, zorder=3)
        if show_labels:
            for (first, second), rank in zip(coordinates, data["rank"]):
                axis.text(first, second, str(rank), ha="center", va="center", color="#172126",
                          fontsize=10, fontweight="bold", zorder=4)
    _coordinate_axes(axis, pair)
    projected = len([name for name in data if name.startswith("x")]) > 2
    axis.set_title("Sampling gaps in this two-input projection" if projected else "Sampling gaps")
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
    figure.colorbar(heatmap, ax=axis, shrink=0.82,
                   label="Distance to nearest observation in this plane")
    if ax is None:
        return _finish(figure, save_to)
    if save_to is not None:
        destination = Path(save_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=160)
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

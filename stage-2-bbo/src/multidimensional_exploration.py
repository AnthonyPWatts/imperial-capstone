"""Descriptive projections and full-space coverage for the three-input case."""

from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from initial_exploration import _coordinate_axes, _finish


def nearest_distances(observations, locations):
    """Euclidean distance using every supplied coordinate, without projection."""
    return cKDTree(np.asarray(observations)).query(np.asarray(locations))[0]


def slice_distances(observations, levels=(0.1, 0.5, 0.9), resolution=121):
    """Distances on x1/x2 planes at fixed x3; all three coordinates count."""
    edges = np.linspace(0, 1, resolution + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    first, second = np.meshgrid(centres, centres)
    planes = []
    for level in levels:
        locations = np.column_stack((first.ravel(), second.ravel(),
                                     np.full(first.size, level)))
        planes.append(nearest_distances(observations, locations).reshape(first.shape))
    return edges, planes


def _rank_markers(axis, data, pair, *, colour=True, label_ranks=None):
    normalise = Normalize(1, len(data))
    palette = plt.get_cmap("viridis_r")
    colours = palette(normalise(data["rank"])) if colour else "white"
    axis.scatter(data[pair[0]], data[pair[1]], c=colours, s=215,
                 edgecolors="#263238", linewidths=0.8, zorder=3, clip_on=False)
    for first, second, rank in data[[*pair, "rank"]].itertuples(index=False, name=None):
        if label_ranks is not None and rank not in label_ranks:
            continue
        red, green, blue, _ = palette(normalise(rank))
        dark = not colour or 0.299 * red + 0.587 * green + 0.114 * blue > 0.5
        axis.text(first, second, str(rank), ha="center", va="center",
                  color="#172126" if dark else "white", fontsize=10,
                  fontweight="bold", zorder=4)


def plot_coordinate_pairs(data, *, gaps=False, save_to=None, pairs=None,
                          distance_max=None, label_ranks=None):
    """All coordinate pairs; gap distances deliberately use only that pair."""
    columns = [column for column in data if column.startswith("x")]
    pairs = list(combinations(columns, 2)) if pairs is None else list(pairs)
    # Three panels per row also keeps later pair collections legible.
    count = len(pairs)
    panel_columns = 2 if count == 4 else min(count, 3)
    panel_rows = (count + panel_columns - 1) // panel_columns
    with plt.rc_context({"font.size": 12}):
        figure, axes = plt.subplots(panel_rows, panel_columns,
                                    figsize=(14.6 * panel_columns / 3, 5.1 * panel_rows),
                                    squeeze=False, layout="constrained")
        panels = axes.ravel()
        edges = np.linspace(0, 1, 122)
        centres = (edges[:-1] + edges[1:]) / 2
        first, second = np.meshgrid(centres, centres)
        locations = np.column_stack((first.ravel(), second.ravel()))
        surfaces = [nearest_distances(data[list(pair)], locations).reshape(first.shape)
                    for pair in pairs] if gaps else []
        maximum = max(surface.max() for surface in surfaces) if gaps else len(data)
        if gaps and distance_max is not None:
            maximum = distance_max
        for index, (axis, pair) in enumerate(zip(panels, pairs)):
            if gaps:
                artist = axis.pcolormesh(edges, edges, surfaces[index], shading="flat",
                                         cmap="viridis", vmin=0, vmax=maximum,
                                         rasterized=True)
            _rank_markers(axis, data, pair, colour=not gaps, label_ranks=label_ranks)
            _coordinate_axes(axis, pair)
            axis.set_title(f"{pair[0]} and {pair[1]}")
        for axis in panels[count:]:
            axis.set_visible(False)
        if not gaps:
            artist = plt.cm.ScalarMappable(norm=Normalize(1, len(data)), cmap="viridis_r")
        colourbar = figure.colorbar(artist, ax=list(panels[:count]), shrink=0.82,
                                    pad=0.02, aspect=25)
        if gaps:
            colourbar.set_label("Nearest distance in this projection")
        else:
            colourbar.set_label("Output rank: 1 = highest")
            colourbar.set_ticks(np.unique(np.rint(np.linspace(1, len(data), 8)).astype(int)))
            colourbar.ax.invert_yaxis()
        return _finish(figure, save_to)


def plot_cube_slices(data, *, save_to=None):
    levels = (0.1, 0.5, 0.9)
    edges, surfaces = slice_distances(data[["x1", "x2", "x3"]], levels)
    with plt.rc_context({"font.size": 12}):
        figure, axes = plt.subplots(1, 3, figsize=(14.6, 5.1), layout="constrained")
        for axis, level, surface in zip(axes, levels, surfaces):
            artist = axis.pcolormesh(edges, edges, surface, shading="flat", cmap="viridis",
                                     vmin=0, vmax=max(plane.max() for plane in surfaces),
                                     rasterized=True)
            _coordinate_axes(axis, ("x1", "x2"))
            axis.set_title(f"Slice at x3 = {level:.1f}")
        figure.colorbar(artist, ax=list(axes), shrink=0.82, pad=0.02, aspect=25,
                        label="Nearest distance in the full cube")
        return _finish(figure, save_to)


def plot_outputs_by_input(data, *, save_to=None):
    with plt.rc_context({"font.size": 12}):
        figure, axes = plt.subplots(1, 3, figsize=(14.6, 4.8), sharey=True,
                                    layout="constrained")
        for axis, column in zip(axes, ("x1", "x2", "x3")):
            axis.scatter(data[column], data.y, c=data["rank"], cmap="viridis_r",
                         vmin=1, vmax=len(data), s=70, edgecolors="#263238",
                         linewidths=0.7, zorder=3, clip_on=False)
            for _, row in data.loc[data["rank"].isin([1, 2, len(data)])].iterrows():
                near_right_edge = row[column] > 0.92
                axis.annotate(str(int(row["rank"])), (row[column], row.y),
                              xytext=(-6 if near_right_edge else 6, -2),
                              ha="right" if near_right_edge else "left",
                              textcoords="offset points", fontsize=11)
            axis.set(xlabel=column, xlim=(0, 1), ylim=(data.y.min() - 0.025, 0),
                     title=f"Output against {column}")
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        axes[0].set_ylabel("Observed y (higher is better)")
        return _finish(figure, save_to)


def correlation_sensitivity(data):
    """A descriptive association check, with one extreme omitted for comparison."""
    reduced = data.drop(index=data["y"].idxmin())
    rows = []
    for column in ("x1", "x2", "x3"):
        rows.append({
            "Input": column,
            "Pearson: all 15": data[column].corr(data.y),
            "Pearson: without rank 15": reduced[column].corr(reduced.y),
            "Spearman: all 15": data[column].corr(data.y, method="spearman"),
        })
    return pd.DataFrame(rows).style.format(precision=2).hide(axis="index")


def exploration_metrics(data, *, save_to=None):
    """Small, reproducible evidence record; no data-validation report."""
    coordinates = data[["x1", "x2", "x3"]].to_numpy()
    distances, neighbours = cKDTree(coordinates).query(coordinates, k=2)
    closest = int(np.argmin(distances[:, 1]))
    other = int(neighbours[closest, 1])
    coverage = []
    for resolution in (41, 81):
        centres = (np.arange(resolution) + 0.5) / resolution
        grid = np.column_stack([part.ravel() for part in
                                np.meshgrid(centres, centres, centres, indexing="ij")])
        distance = nearest_distances(coordinates, grid)
        coverage.append({"resolution": resolution, "locations": len(grid),
                         "mean_distance": float(distance.mean()),
                         "p90_distance": float(np.quantile(distance, 0.9))})
    result = {
        "observations": len(data), "dimensions": 3,
        "best": data.loc[data.y.idxmax()].to_dict(),
        "worst": data.loc[data.y.idxmin()].to_dict(),
        "closest_pair_ranks": [int(data.iloc[closest]["rank"]), int(data.iloc[other]["rank"])],
        "closest_pair_distance": float(distances[closest, 1]),
        "closest_pair_output_difference": float(abs(data.iloc[closest].y - data.iloc[other].y)),
        "cube_coverage": coverage,
        "input_minima": data[["x1", "x2", "x3"]].min().to_dict(),
        "magnitude_ratio": float(data.y.abs().max() / data.y.abs().min()),
    }
    if save_to is not None:
        destination = Path(save_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result

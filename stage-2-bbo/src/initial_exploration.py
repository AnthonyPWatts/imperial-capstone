"""Explore the supplied observations and select a first 2D coverage query."""

from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, cKDTree


def load_initial_data(stage: Path, function_id: int = 1) -> pd.DataFrame:
    directory = stage / "data" / "initial_data" / f"function_{function_id}"
    inputs = np.load(directory / "initial_inputs.npy", allow_pickle=False)
    outputs = np.load(directory / "initial_outputs.npy", allow_pickle=False)
    if inputs.ndim != 2 or outputs.shape != (len(inputs),):
        raise ValueError("Expected one output for each row of input coordinates.")
    if not np.isfinite(inputs).all() or not np.isfinite(outputs).all():
        raise ValueError("The supplied observations contain non-finite values.")
    data = pd.DataFrame(inputs, columns=[f"x{i + 1}" for i in range(inputs.shape[1])])
    data["y"] = outputs
    # Equal outputs share their rank; file order is never used to break ties.
    data["rank"] = data["y"].rank(ascending=False, method="min").astype(int)
    return data


def preview(data: pd.DataFrame, rows: int = 5, *, output_format="{:.8e}"):
    formats = {column: "{:.6f}" for column in data if column.startswith("x")}
    formats.update({"y": output_format, "rank": "{:d}"})
    return data.head(rows).style.format(formats).hide(axis="index")


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


def plot_ranked_locations(data: pd.DataFrame, pair=("x1", "x2"), *, save_to=None):
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


def plot_sampling_gaps(data: pd.DataFrame, pair=("x1", "x2"), *, save_to=None,
                       candidate=None, central_alternative=None):
    coordinates = data[list(pair)].to_numpy()
    edges = np.linspace(0, 1, 202)
    centres = (edges[:-1] + edges[1:]) / 2
    first, second = np.meshgrid(centres, centres)
    mesh = np.column_stack([first.ravel(), second.ravel()])
    distances = cKDTree(coordinates).query(mesh)[0].reshape(first.shape)
    figure, axis = plt.subplots(figsize=(7.6, 6.1), layout="constrained")
    heatmap = axis.pcolormesh(edges, edges, distances, cmap="viridis", vmin=0,
                            shading="flat", rasterized=True)
    axis.scatter(*coordinates.T, s=310, facecolors="white", edgecolors="#263238",
                 linewidths=0.9, zorder=3)
    for (first, second), rank in zip(coordinates, data["rank"]):
        axis.text(first, second, str(rank), ha="center", va="center", color="#172126",
                  fontsize=10, fontweight="bold", zorder=4)
    _coordinate_axes(axis, pair)
    projected = len([name for name in data if name.startswith("x")]) > 2
    axis.set_title("Sampling gaps in this two-input projection" if projected else "Sampling gaps")
    if candidate is not None:
        point = np.asarray(candidate)
        nearest = coordinates[cKDTree(coordinates).query(point)[1]]
        axis.plot(*np.vstack([point, nearest]).T, color="white", linestyle="--",
                  linewidth=1.6, zorder=4)
        axis.scatter(*point, s=260, marker="*", color="#ffbf3f", edgecolors="#172126",
                     linewidths=1.2, zorder=6, clip_on=False)
        axis.annotate(f"Maximin choice\n({point[0]:.6f}, {point[1]:.6f})", xy=point,
                      xytext=(0.14, 0.96), ha="left", va="top", fontsize=10,
                      bbox={"facecolor": "white", "alpha": 0.95, "edgecolor": "none"},
                      arrowprops={"arrowstyle": "-", "color": "#172126"}, zorder=7)
        axis.set_title("Choosing the next point by coverage")
    if central_alternative is not None:
        axis.scatter(*central_alternative, s=95, marker="D", facecolors="none",
                     edgecolors="#172126", linewidths=1.7, zorder=5)
        axis.annotate("Central alternative", xy=central_alternative, xytext=(0.15, 0.55),
                      fontsize=10, color="#172126",
                      bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
                      arrowprops={"arrowstyle": "-", "color": "#172126"}, zorder=6)
    figure.colorbar(heatmap, ax=axis, shrink=0.82,
                   label="Distance to nearest observation in this plane")
    return _finish(figure, save_to)


def maximin_candidates_2d(data: pd.DataFrame) -> pd.DataFrame:
    """Score 2D geometric candidates, using six-decimal portal coordinates.

    On each clipped Voronoi cell, distance to its observation is convex, so
    a maximum lies at a cell vertex: an interior Voronoi vertex, an edge
    intersection or a box corner. All pairwise bisector/edge intersections
    include the needed edge vertices, plus harmless additional candidates.
    The unrounded maximum is also retained as an upper bound for checking
    whether the rounded winner attains the continuous optimum.
    """
    inputs = [column for column in data if column.startswith("x")]
    if inputs != ["x1", "x2"]:
        raise ValueError("This candidate construction is for two inputs only.")
    coordinates = data[inputs].to_numpy()
    upper = 0.999999
    candidates = [np.array(point) for point in
                  ((0, 0), (0, upper), (upper, 0), (upper, upper))]
    candidates.extend(Voronoi(coordinates).vertices)
    for first, second in combinations(coordinates, 2):
        normal = 2 * (second - first)
        offset = second @ second - first @ first
        for fixed in (0, 1):
            other = 1 - fixed
            if normal[other] == 0:
                continue
            for edge in (0, upper):
                point = np.zeros(2)
                point[fixed] = edge
                point[other] = (offset - normal[fixed] * edge) / normal[other]
                candidates.append(point)
    candidates = np.asarray(candidates)
    candidates = candidates[((candidates >= 0) & (candidates <= upper)).all(axis=1)]
    tree = cKDTree(coordinates)
    continuous_maximum = float(tree.query(candidates)[0].max())
    candidates = np.unique(np.round(candidates, 6), axis=0)
    scores = tree.query(candidates)[0]
    boundaries = ((candidates == 0) | (candidates == upper)).sum(axis=1)
    result = pd.DataFrame(candidates, columns=inputs)
    result["nearest_distance"] = scores
    result["location"] = np.select([boundaries == 2, boundaries == 1],
                                    ["corner", "edge"], default="interior")
    result = result.sort_values(["nearest_distance", "x1", "x2"],
                                ascending=[False, True, True]).reset_index(drop=True)
    result.attrs["continuous_maximum"] = continuous_maximum
    return result


def maximin_comparison(candidates: pd.DataFrame):
    interior = candidates.loc[candidates["location"] == "interior"].head(1)
    comparison = pd.concat([candidates.head(1), interior]).copy()
    comparison.insert(0, "Candidate", ["Maximin choice", "Central alternative"])
    comparison = comparison.drop(columns="location").rename(
        columns={"nearest_distance": "Distance to nearest observation"})
    return comparison.reset_index(drop=True).style.format({
        "x1": "{:.6f}", "x2": "{:.6f}", "Distance to nearest observation": "{:.6f}",
    }).hide(axis="index")


def plot_maximin_choice(data: pd.DataFrame, candidates: pd.DataFrame, *, save_to=None):
    chosen = candidates.iloc[0][["x1", "x2"]].to_numpy(dtype=float)
    interior = candidates.loc[candidates["location"] == "interior"].iloc[0]
    return plot_sampling_gaps(
        data, candidate=chosen,
        central_alternative=interior[["x1", "x2"]].to_numpy(dtype=float), save_to=save_to,
    )


def coverage_value_comparison(data: pd.DataFrame, candidates: pd.DataFrame,
                              inset=(0.07, 0.93), resolution=701):
    """Compare coverage of the usable square, with uniform area weighting."""
    chosen = candidates.iloc[0][["x1", "x2"]].to_numpy(dtype=float)
    interior = candidates.loc[candidates["location"] == "interior"].iloc[0]
    central = interior[["x1", "x2"]].to_numpy(dtype=float)
    centres = (np.arange(resolution) + 0.5) / resolution
    first, second = np.meshgrid(centres, centres)
    grid = np.column_stack([first.ravel(), second.ravel()])
    tree = cKDTree(data[["x1", "x2"]].to_numpy())
    before = tree.query(grid)[0]
    rows = []
    for name, point in (("Maximin corner", chosen),
                        (f"Inset ({inset[0]:.2f}, {inset[1]:.2f})", inset),
                        ("Central gap", central)):
        distance = np.linalg.norm(grid - point, axis=1)
        after = np.minimum(before, distance)
        rows.append({
            "Candidate": name,
            "Distance from nearest sample": tree.query(point)[0],
            "Reduction in mean sampling distance": 1 - after.mean() / before.mean(),
            "Area brought closer to a sample": np.mean(distance < before),
        })
    return pd.DataFrame(rows).style.format({
        "Distance from nearest sample": "{:.3f}",
        "Reduction in mean sampling distance": "{:.2%}",
        "Area brought closer to a sample": "{:.2%}",
    }).hide(axis="index")


def average_coverage_summary(result):
    summary = pd.DataFrame([{
        "x1": result["point"][0], "x2": result["point"][1],
        "Mean distance before": result["mean_before"],
        "Mean distance after": result["mean_after"],
        "Reduction": result["reduction"],
    }])
    return summary.style.format({
        "x1": "{:.6f}", "x2": "{:.6f}",
        "Mean distance before": "{:.6f}", "Mean distance after": "{:.6f}",
        "Reduction": "{:.2%}",
    }).hide(axis="index")


def plot_average_coverage(data: pd.DataFrame, result, *, save_to=None):
    gains = 100 * (1 - result["candidate_mean_distances"] / result["mean_before"])
    figure, axis = plt.subplots(figsize=(7.6, 6.1), layout="constrained")
    heatmap = axis.imshow(gains, origin="lower", extent=(0, 1, 0, 1),
                          cmap="viridis", vmin=0, interpolation="bilinear")
    axis.scatter(data["x1"], data["x2"], s=250, facecolors="white",
                 edgecolors="#263238", linewidths=0.9, zorder=3)
    for first, second, rank in data[["x1", "x2", "rank"]].itertuples(index=False, name=None):
        axis.text(first, second, str(rank), ha="center", va="center", color="#172126",
                  fontsize=9, fontweight="bold", zorder=4)
    point = result["point"]
    axis.scatter(*point, s=260, marker="*", color="#ffbf3f", edgecolors="#172126",
                 linewidths=1.2, zorder=5)
    axis.annotate(f"Optimised coverage point\n({point[0]:.6f}, {point[1]:.6f})", xy=point,
                  xytext=(0.15, 0.56), fontsize=10,
                  bbox={"facecolor": "white", "alpha": 0.95, "edgecolor": "none"},
                  arrowprops={"arrowstyle": "-", "color": "#172126"}, zorder=6)
    _coordinate_axes(axis, ("x1", "x2"))
    axis.set_title("How much would adding a point here improve average coverage?")
    figure.colorbar(heatmap, ax=axis, shrink=0.82,
                   label="Reduction in mean sampling distance (%)")
    return _finish(figure, save_to)


def plot_ordered_outputs(data: pd.DataFrame, *, exclude_rank=None, save_to=None):
    ordered = data.sort_values("y", ascending=False, kind="stable")
    if exclude_rank is not None:
        ordered = ordered[ordered["rank"] != exclude_rank]
    ranks = ordered["rank"].to_numpy()
    outputs = ordered["y"].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), layout="constrained")
    if exclude_rank is not None:
        figure.suptitle(f"Rank {exclude_rank} excluded from this view")
    axes[0].axhline(0, color="#969da2", linewidth=0.8)
    for mask, colour, marker, label in (
        (outputs > 0, "#236b91", "o", "Positive output"),
        (outputs < 0, "#bd531d", "v", "Negative output"),
        (outputs == 0, "#666666", "x", "Exact zero"),
    ):
        if not mask.any():
            continue
        axes[0].scatter(ranks[mask], outputs[mask], color=colour, marker=marker,
                        s=65, label=label, zorder=3)
        nonzero = mask & (outputs != 0)
        if nonzero.any():
            axes[1].scatter(ranks[nonzero], np.log10(np.abs(outputs[nonzero])),
                            color=colour, marker=marker, s=65, label=label, zorder=3)
    axes[0].set(title="Original scale", ylabel="Observed y")
    tiny_scale = 0 < np.max(np.abs(outputs)) < 1e-4
    axes[0].ticklabel_format(axis="y", style="sci" if tiny_scale else "plain",
                             scilimits=(0, 0), useOffset=False, useMathText=True)
    if (outputs > 0).all() and outputs.min() > 0.25 * outputs.max():
        padding = max(float(np.ptp(outputs)) * 0.08, outputs.max() * 0.01)
        axes[0].set_ylim(outputs.min() - padding, outputs.max() + padding)
    axes[0].legend(loc="best", fontsize=9)
    axes[1].set(title="Logarithmic magnitude; markers retain the sign",
                ylabel="log₁₀ |y| (each +1 means ×10 in magnitude)")
    # Keep the log axis comparable between the complete and excluded-rank views.
    all_nonzero = np.abs(data.loc[data["y"] != 0, "y"].to_numpy())
    if all_nonzero.size:
        logarithms = np.log10(all_nonzero)
        padding = max(float(np.ptp(logarithms)) * 0.05, 0.5)
        if np.ptp(logarithms) < 0.5:
            padding = 0.05
        axes[1].set_ylim(logarithms.min() - padding, logarithms.max() + padding)
    if (outputs == 0).any():
        axes[1].text(0.02, 0.02, "Exact zeros have no finite logarithm and are omitted here.",
                     transform=axes[1].transAxes, fontsize=8)
    for axis in axes:
        axis.set_xlabel("Output rank (1 = highest y)")
        ticks = sorted(data["rank"].unique())
        if len(ticks) > 20:
            ticks = np.unique(np.r_[ticks[0], np.arange(5, ticks[-1] + 1, 5), ticks[-1]])
        axis.set_xticks(ticks)
        axis.set_xlim(data["rank"].min() - 0.5, data["rank"].max() + 0.5)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    return _finish(figure, save_to)

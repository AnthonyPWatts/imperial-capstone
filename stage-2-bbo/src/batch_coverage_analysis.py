"""Tables and figures for planning one, two or three coverage-only queries."""

import json
from math import comb
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from coverage_optimisation import MeanCoverage2D, optimise_average_coverage, optimise_coverage_batch
from initial_exploration import _finish


def analyse_coverage_batches(data, output=None):
    started = perf_counter()
    inputs = data[["x1", "x2"]].copy()
    original = MeanCoverage2D(inputs.to_numpy())
    one = optimise_average_coverage(inputs)
    greedy_points = [one["point"]]
    greedy = {}
    for count in (2, 3):
        augmented = pd.DataFrame(np.vstack([inputs.to_numpy(), greedy_points]), columns=["x1", "x2"])
        step = optimise_coverage_batch(augmented, 1)
        greedy_points.append(step["points"][0])
        points = np.array(greedy_points)
        mean_after = original.mean_after(points)
        greedy[count] = {
            "points": points.copy(), "mean_after": mean_after,
            "reduction": 1 - mean_after / original.mean_before,
            "last_step_seconds": step["elapsed_seconds"],
        }
    joint = {1: {
        "points": one["point"].reshape(1, 2), "mean_after": one["mean_after"],
        "mean_before": one["mean_before"], "reduction": one["reduction"],
        "elapsed_seconds": one["elapsed_seconds"],
    }}
    for count in (2, 3):
        joint[count] = optimise_coverage_batch(inputs, count, initial_points=greedy[count]["points"])
    study = {
        "mean_before": original.mean_before,
        "joint": joint, "greedy": greedy,
        "total_seconds": perf_counter() - started,
    }
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "study.json").write_text(
            json.dumps(study, indent=2, default=lambda value: value.tolist(), allow_nan=False),
            encoding="utf-8",
        )
    return study


def batch_coverage_summary(study):
    rows = [{"New points": 0, "Mean sampling distance": study["mean_before"],
             "Total reduction": 0.0, "Extra reduction (percentage points)": 0.0}]
    previous = 0.0
    for count, result in study["joint"].items():
        rows.append({
            "New points": count, "Mean sampling distance": result["mean_after"],
            "Total reduction": result["reduction"],
            "Extra reduction (percentage points)": 100 * (result["reduction"] - previous),
        })
        previous = result["reduction"]
    return pd.DataFrame(rows).style.format({
        "Mean sampling distance": "{:.6f}", "Total reduction": "{:.2%}",
        "Extra reduction (percentage points)": "{:.2f}",
    }).hide(axis="index")


def batch_coordinate_table(study):
    rows = []
    for count, result in study["joint"].items():
        for index, point in enumerate(result["points"]):
            rows.append({"N": count, "Point": chr(65 + index), "x1": point[0], "x2": point[1],
                         "Portal form": f"{point[0]:.6f}-{point[1]:.6f}"})
    return pd.DataFrame(rows).style.format({"x1": "{:.6f}", "x2": "{:.6f}"}).hide(axis="index")


def batch_strategy_comparison(study):
    rows = []
    for count in (2, 3):
        joint, greedy = study["joint"][count], study["greedy"][count]
        rows.append({
            "N": count, "Sequential reduction": greedy["reduction"],
            "Joint reduction": joint["reduction"],
            "Joint advantage (percentage points)": 100 * (joint["reduction"] - greedy["reduction"]),
        })
    return pd.DataFrame(rows).style.format({
        "Sequential reduction": "{:.4%}", "Joint reduction": "{:.4%}",
        "Joint advantage (percentage points)": "{:.4f}",
    }).hide(axis="index")


def batch_computation_table(study):
    rows = []
    for count, result in study["joint"].items():
        rows.append({
            "N": count, "Coordinates optimised": 2 * count,
            "Measured search time (seconds)": result["elapsed_seconds"],
            "Sets on a 41 x 41 candidate grid": comb(41**2, count),
        })
    return pd.DataFrame(rows).style.format({
        "Measured search time (seconds)": "{:.2f}", "Sets on a 41 x 41 candidate grid": "{:,}",
    }).hide(axis="index")


def plot_batch_coverage(data, study, *, save_to=None):
    inputs = data[["x1", "x2"]].to_numpy()
    edges = np.linspace(0, 1, 202)
    centres = (edges[:-1] + edges[1:]) / 2
    first, second = np.meshgrid(centres, centres)
    grid = np.column_stack([first.ravel(), second.ravel()])
    original_distances = cKDTree(inputs).query(grid)[0]
    # The same scale is used in every panel, including the original observations.
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), layout="constrained")
    for count, axis in enumerate(axes.ravel()):
        points = study["joint"][count]["points"] if count else np.empty((0, 2))
        distances = cKDTree(np.vstack([inputs, points])).query(grid)[0].reshape(first.shape)
        heatmap = axis.pcolormesh(edges, edges, distances, cmap="viridis", vmin=0,
                                 vmax=original_distances.max(), shading="flat", rasterized=True)
        axis.scatter(*inputs.T, s=145, facecolors="white", edgecolors="#263238", linewidths=.8, zorder=3)
        for (x1, x2), rank in zip(inputs, data["rank"]):
            axis.text(x1, x2, str(rank), ha="center", va="center", fontsize=8, color="#172126", zorder=4)
        for index, point in enumerate(points):
            axis.scatter(*point, s=200, marker="*", color="#ffbf3f", edgecolors="#172126", zorder=5)
            axis.annotate(chr(65 + index), point, xytext=(9, 9), textcoords="offset points",
                          fontsize=10, fontweight="bold", color="#172126",
                          bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none", "pad": 1}, zorder=6)
        reduction = study["joint"][count]["reduction"] if count else 0.0
        title = "Original ten locations" if not count else f"N = {count}: {reduction:.2%} mean-distance reduction"
        axis.set(title=title, xlabel="x1", ylabel="x2", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_yticks(np.linspace(0, 1, 6))
    figure.suptitle("Remaining sampling gaps: circles are fixed; stars are planned points", fontsize=13)
    figure.colorbar(heatmap, ax=axes.ravel().tolist(), shrink=.75,
                   label="Distance to nearest original or planned sample")
    return _finish(figure, save_to)

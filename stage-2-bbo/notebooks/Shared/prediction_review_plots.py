"""Comparable GP/neural response views with explicit target and uncertainty units."""

from itertools import combinations
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, QhullError

from model_helpers import finish_figure


def plot_response_profiles(gp, neural_models, inputs, target, anchor, *, input_columns,
                           target_label, title, side=151, save_to=None):
    """Hold every omitted input fixed; distinguish latent GP SD from seed variation."""
    inputs, target, anchor = np.asarray(inputs), np.asarray(target), np.asarray(anchor)
    dimensions = len(input_columns)
    columns = min(dimensions, 3)
    rows = int(np.ceil(dimensions / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5*columns, 3.9*rows), squeeze=False,
                                sharex=True, sharey=True, layout="constrained")
    coordinates = np.linspace(0, .999999, side)
    for axis, label in enumerate(input_columns):
        ax = axes.flat[axis]
        points = np.tile(anchor, (side, 1))
        points[:, axis] = coordinates
        mean, sd = gp.predict(points, return_std=True)
        values = np.array([model.predict(points) for model in neural_models])
        neural_mean, neural_sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        ax.fill_between(coordinates, mean-1.96*sd, mean+1.96*sd, color="#007f86", alpha=.14)
        ax.plot(coordinates, mean, color="#007f86", label="GP mean; ±1.96 latent SD")
        ax.fill_between(coordinates, neural_mean-neural_sd, neural_mean+neural_sd,
                        color="#8055a0", alpha=.14)
        ax.plot(coordinates, neural_mean, color="#8055a0", label="Neural mean; ±1 seed SD")
        other = [i for i in range(dimensions) if i != axis]
        on_slice = np.isclose(inputs[:, other], anchor[other], rtol=0, atol=5e-7).all(axis=1)
        ax.scatter(inputs[on_slice, axis], target[on_slice], marker="x", color="#b76b00", s=65,
                   label="Measurements on this slice", zorder=4)
        ax.axvline(anchor[axis], color="#b76b00", linestyle=":", linewidth=1)
        held = ", ".join(f"{input_columns[j]}={anchor[j]:.3f}" for j in other)
        ax.set(xlim=(0, 1), xlabel=label, ylabel=target_label,
               title=fill("Hold " + held, 52), axisbelow=True)
        ax.grid(alpha=.2)
    for ax in axes.flat[dimensions:]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    figure.suptitle(title + "\nShared output limits. The two bands describe different quantities.", fontsize=14)
    return finish_figure(figure, save_to)


def plot_score_planes(model, inputs, anchor, *, input_columns, target_label, title,
                      weight=1.96, side=61, save_to=None):
    """Show full-vector hull boundaries in conditional planes for two or three inputs."""
    inputs, anchor = np.asarray(inputs), np.asarray(anchor)
    dimensions = len(input_columns)
    if dimensions not in (2, 3):
        raise ValueError("Use conditional coordinate curves for more than three inputs")
    pairs = list(combinations(range(dimensions), 2))
    grid = np.linspace(0, .999999, side)
    xx, yy = np.meshgrid(grid, grid)
    try:
        hull = Delaunay(inputs)
    except QhullError:
        hull = None
    sections = []
    for a, b in pairs:
        points = np.tile(anchor, (side*side, 1));points[:, a] = xx.ravel();points[:, b] = yy.ravel()
        mean, sd = model.predict(points, return_std=True)
        sections.append((points, [mean.reshape(xx.shape), sd.reshape(xx.shape),
                                  (mean+weight*sd).reshape(xx.shape)]))
    figure, axes = plt.subplots(len(pairs), 3, figsize=(14, 4.1*len(pairs)), squeeze=False,
                                layout="constrained")
    labels = [f"GP mean ({target_label})", f"Latent SD ({target_label})", f"UCB: mean + {weight:g} × SD"]
    for row, ((a, b), (points, fields)) in enumerate(zip(pairs, sections)):
        for column, field in enumerate(fields):
            ax = axes[row, column]
            low = min(s[1][column].min() for s in sections)
            high = max(s[1][column].max() for s in sections)
            if high <= low:
                high = low + max(abs(low)*1e-6, 1e-10)
            image = ax.contourf(xx, yy, field, levels=np.linspace(low, high, 17), cmap="viridis")
            figure.colorbar(image, ax=ax, shrink=.8)
            if hull is not None:
                inside = (hull.find_simplex(points) >= 0).reshape(xx.shape)
                if inside.any() and not inside.all():
                    ax.contour(xx, yy, inside.astype(float), levels=[.5], colors="white", linestyles=":")
            other = [i for i in range(dimensions) if i not in (a, b)]
            keep = np.isclose(inputs[:, other], anchor[other], rtol=0, atol=5e-7).all(axis=1)
            ax.scatter(inputs[keep, a], inputs[keep, b], color="black", s=18)
            ax.scatter(anchor[a], anchor[b], color="magenta", marker="X", s=90, edgecolor="black", clip_on=False)
            held = ", ".join(f"{input_columns[j]}={anchor[j]:.3f}" for j in other)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=input_columns[a], ylabel=input_columns[b],
                   title=labels[column] + ("\nHold " + held if held else ""))
            ax.set_aspect("equal")
    figure.suptitle(title + "\nMagenta: proposed query. Dotted white: sampled full-space hull; colours beyond it extrapolate.")
    return finish_figure(figure, save_to)

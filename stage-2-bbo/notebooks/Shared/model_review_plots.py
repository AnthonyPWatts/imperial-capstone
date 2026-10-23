"""Validation plots that depend on prediction results, not input dimensionality."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error

from model_helpers import finish_figure, gp_slice, plot_posterior_curve


def plot_held_out_comparison(predictions, target, *, model_names, latest_mask,
                             title, figsize=(15, 5), padding_fraction=.05, save_to=None):
    """Compare declared models on identical output axes and highlight the latest row."""
    target, latest_mask = np.asarray(target), np.asarray(latest_mask, dtype=bool)
    values = np.concatenate([target, *(predictions[name] for name in model_names)])
    padding = padding_fraction * max(float(np.ptp(values)), np.finfo(float).eps)
    limits = (float(values.min() - padding), float(values.max() + padding))
    fig, axes = plt.subplots(1, len(model_names), figsize=figsize, squeeze=False)
    for ax, name in zip(axes[0], model_names):
        predicted = np.asarray(predictions[name])
        ax.scatter(target[~latest_mask], predicted[~latest_mask], alpha=.75)
        ax.scatter(target[latest_mask], predicted[latest_mask], marker="*", s=180,
                   color="darkorange", edgecolor="black", label="Latest observation")
        ax.plot(limits, limits, "k--", linewidth=1, label="Perfect prediction")
        ax.set(xlim=limits, ylim=limits, xlabel="Observed output",
               title=f"{name}\nLOO RMSE: {root_mean_squared_error(target, predicted):.3f}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="upper left")
    axes[0, 0].set_ylabel("Held-out predicted output")
    fig.suptitle(title, fontsize=15)
    fig.text(.5, .02, "Every point is predicted with its own result held out. Higher outputs are better; all observations are retained.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .95))
    return finish_figure(fig, save_to)


def plot_latest_predictions(table, latest_output, *, n_training_rows, title,
                             figsize=(11, 5.5), save_to=None):
    """Compare predictions for one held-out return against its measured value."""
    table = table.sort_values("Latest held-out prediction", ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(table))
    values = table["Latest held-out prediction"].to_numpy()
    ax.scatter(values, positions, color="#3279a8", s=70, zorder=3)
    ax.set_yticks(positions, table["Model"])
    ax.invert_yaxis()
    ax.axvline(latest_output, color="darkorange", linewidth=2, linestyle="--",
               label=f"Observed: {latest_output:.3f}")
    lower, upper = min(latest_output, values.min()), max(latest_output, values.max())
    padding = .04 * max(float(upper - lower), np.finfo(float).eps)
    for position, value in zip(positions, values):
        ax.text(value + padding, position, f"{value:.3f}", va="center", fontsize=10)
    ax.set_xlim(lower - padding, upper + 4 * padding)
    ax.set_xlabel("Held-out prediction of the latest output (higher is better)")
    ax.set_title(f"{title}\nTrain on {n_training_rows} earlier observations; actual output {latest_output:.3f}")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return finish_figure(fig, save_to)


def plot_nearby_differences(review, *, assumed_noise_sd, title, maximum_distance=.1,
                            figsize=(12, 5), save_to=None):
    """Show local output differences and cutoff sensitivity in original output units."""
    pairs, cutoffs = review["pairs"], review["cutoffs"]
    shown = pairs.loc[pairs.Distance <= maximum_distance]
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    axes[0].scatter(shown.Distance, shown["Absolute output difference"], color="#4678a0")
    if not shown.empty:
        closest = shown.iloc[0]
        axes[0].scatter(closest.Distance, closest["Absolute output difference"],
                        marker="*", s=130, color="darkorange", label="Closest pair", zorder=3)
        axes[0].legend()
    axes[0].set(xlim=(0, maximum_distance), ylim=(0, None),
                xlabel="Full-input Euclidean distance", ylabel="Absolute output difference",
                title=f"Nearby pairs ({len(shown)} shown)")
    axes[1].plot(cutoffs["Maximum distance"], cutoffs["Noise SD proxy"], "o-", color="#4678a0")
    for row in cutoffs.to_dict("records"):
        count = row["Pairs"]
        axes[1].annotate(f"{count} {'pair' if count == 1 else 'pairs'}",
                         (row["Maximum distance"], row["Noise SD proxy"]),
                         xytext=(12, 0) if count == 1 else (0, 8), textcoords="offset points",
                         ha="left" if count == 1 else "center",
                         va="center" if count == 1 else "bottom", fontsize=9)
    axes[1].margins(x=.12, y=.15)
    axes[1].axhline(assumed_noise_sd, color="darkorange", linestyle="--", label="Current assumed noise SD")
    axes[1].set(xlabel="Maximum pair distance", ylabel="RMS output difference / √2",
                title="Noise proxy depends on the cutoff", ylim=(0, None))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.suptitle(title)
    fig.supxlabel("Distinct inputs mix function changes with noise. Pairs can share observations; counts are not independent repeats.", fontsize=10)
    return finish_figure(fig, save_to)


def plot_coordinate_model_review(models, inputs, target, anchor, *, input_columns,
                                  title, side=250, padding_fraction=.05,
                                  figsize=None, save_to=None):
    """Conditional curves for any input count, holding every other coordinate fixed."""
    inputs, target, anchor = np.asarray(inputs), np.asarray(target), np.asarray(anchor)
    dimensions = len(input_columns)
    if inputs.ndim != 2 or inputs.shape[1] != dimensions or anchor.shape != (dimensions,):
        raise ValueError("Input labels and anchor must match all input dimensions")
    sections = [[gp_slice(model, inputs, target, anchor, axis, side=side) for axis in range(dimensions)]
                for model in models.values()]
    lower = min(np.min(section["mean"]-1.96*section["std"]) for row in sections for section in row)
    upper = max(np.max(section["mean"]+1.96*section["std"]) for row in sections for section in row)
    padding = padding_fraction * max(float(upper-lower), np.finfo(float).eps)
    fig, axes = plt.subplots(len(models), dimensions, figsize=figsize or (4*dimensions, 4*len(models)),
                              sharex=True, sharey=True, squeeze=False)
    for row, name in enumerate(models):
        for axis, section in enumerate(sections[row]):
            ax = axes[row, axis]
            plot_posterior_curve(ax, section["coordinates"], section["mean"], section["std"],
                                 section["observed_x"], section["observed_y"])
            held = ", ".join(f"{input_columns[i]}={anchor[i]:.3f}" for i in range(dimensions) if i != axis)
            ax.set(xlim=(0, 1), ylim=(lower-padding, upper+padding), xlabel=f"Vary {input_columns[axis]}",
                   title=f"{name}\nHold {held}" if held else name)
            if axis == 0:
                ax.set_ylabel("Output y")
            ax.legend(fontsize=7)
    fig.suptitle(title)
    fig.text(.5, .01, "All panels share vertical limits. Green crosses match every held coordinate; other measurements are not projected onto the curves.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    return finish_figure(fig, save_to)


def plot_paired_cohort_profiles(paired, proposals, *, input_columns, target_count,
                                title, figsize=(14, 5.5), save_to=None):
    """Paired errors and exact proposed input profiles without dropping dimensions."""
    points = np.asarray(list(proposals.values()))
    if points.ndim != 2 or points.shape[1] != len(input_columns):
        raise ValueError("Proposals must retain every labelled input coordinate")
    fig, axes = plt.subplots(1, 2, figsize=figsize, layout="constrained")
    positions = np.arange(len(paired))
    axes[0].barh(positions-.18, paired.iloc[:, 1], height=.34, label="Official training")
    axes[0].barh(positions+.18, paired.iloc[:, 2], height=.34, label="Official + peer training")
    axes[0].set(yticks=positions, yticklabels=paired.Model,
                xlabel=f"RMSE on the same {target_count} official targets", title="Paired held-out comparison")
    axes[0].legend(fontsize=8)
    image = axes[1].imshow(points, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (row, column), value in np.ndenumerate(points):
        axes[1].text(column, row, f"{value:.6f}", ha="center", va="center", fontsize=10,
                     color="white" if value > .58 else "#172d40")
    axes[1].set(xticks=np.arange(len(input_columns)), xticklabels=input_columns,
                yticks=np.arange(len(proposals)), yticklabels=list(proposals),
                title="Unevaluated input profiles (all coordinates)")
    fig.colorbar(image, ax=axes[1], orientation="horizontal", pad=.10, shrink=.85,
                 label="Input setting (colour does not encode output)")
    fig.suptitle(title)
    return finish_figure(fig, save_to)

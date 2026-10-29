"""Plots for measured surrogate validation, input gradients and conditional boundaries."""

from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from model_helpers import finish_figure


def validation_overview(regression, classification, *, title, save_to=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), layout="constrained")
    reg = regression["table"].sort_values("LOO RMSE", ascending=False)
    cls = classification["table"].sort_values("LOO balanced accuracy")
    for ax, table, column in ((axes[0], reg, "RMSE / baseline"), (axes[1], cls, "LOO balanced accuracy")):
        colours = ["#087f8c" if name.startswith("NN ") else "#66788a" for name in table.Model]
        ax.barh(table.Model, table[column], color=colours)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].axvline(1, color="#b84a36", linestyle="--", label="Mean baseline")
    axes[0].set(xlabel="Held-out RMSE / mean-baseline RMSE (lower is better)", title="Numerical score prediction")
    axes[0].legend(fontsize=8)
    axes[1].axvline(.5, color="#b84a36", linestyle="--", label="Balanced-accuracy reference")
    axes[1].set(xlim=(0, 1.03), xlabel="Held-out balanced accuracy (higher is better)", title="Good versus bad: training-fold upper quartile")
    axes[1].legend(fontsize=8)
    fig.suptitle(title + "\nAll fixed configurations and seeds are retained; teal marks neural networks")
    return finish_figure(fig, save_to)


def seed_stability(regression, classification, *, title, save_to=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for ax, study, label in ((axes[0], regression, "Held-out RMSE"),
                             (axes[1], classification, "Held-out balanced accuracy")):
        table = study["table"]
        ax.errorbar(np.arange(len(table)), table["mean"], yerr=table["std"], fmt="o", capsize=6, color="#087f8c")
        ax.set(xticks=np.arange(len(table)), xticklabels=table.architecture, ylabel=label)
        ax.grid(alpha=.2)
    axes[0].set_title("Regression: architecture means and seed variation")
    axes[1].set(title="Classification: architecture means and seed variation", ylim=(0, 1.03))
    fig.suptitle(title + "\nBars show one standard deviation across three initialisations, not confidence intervals")
    return finish_figure(fig, save_to)


def regression_validation(run, target, names, *, title, target_label, save_to=None):
    values = np.concatenate([target, *(run["predictions"][name] for name in names)])
    pad = .05 * max(float(np.ptp(values)), 1e-10)
    limits = (values.min()-pad, values.max()+pad)
    fig, axes = plt.subplots(1, len(names), figsize=(14, 4.7), squeeze=False, layout="constrained")
    table = run["table"].set_index("Model")
    for ax, name in zip(axes[0], names):
        ax.scatter(target, run["predictions"][name], color="#087f8c", edgecolor="white", s=45)
        ax.plot(limits, limits, "k--", linewidth=1)
        ax.set(xlim=limits, ylim=limits, xlabel="Observed " + target_label,
               title=fill(name, 28) + f"\nLOO RMSE {table.loc[name, 'LOO RMSE']:.4g}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Held-out predicted " + target_label)
    fig.suptitle(title + "\nIdentical held-out observations and shared axes")
    return finish_figure(fig, save_to)


def classifier_confusions(run, names, *, title, save_to=None):
    matrices = [confusion_matrix(run["truth"], run["predictions"][name], labels=[0, 1]) for name in names]
    fig, axes = plt.subplots(1, len(names), figsize=(13, 4.7), layout="constrained")
    maximum = max(matrix.max() for matrix in matrices)
    for ax, name, matrix in zip(axes, names, matrices):
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for row in range(2):
            for column in range(2):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=16,
                        color="white" if matrix[row, column] > .6*maximum else "#142d42")
        ax.set(xticks=[0, 1], xticklabels=["Bad", "Good"], yticks=[0, 1], yticklabels=["Bad", "Good"],
               xlabel="Held-out predicted class", ylabel="Actual class against training-fold threshold",
               title=fill(name, 28))
    fig.suptitle(title + "\nBottom left: missed good points; top right: queries that could disappoint")
    return finish_figure(fig, save_to)


def gradient_comparison(columns, local_gradients, seed_mean, seed_sd, probability_gradient, *,
                        title, target_label, save_to=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), layout="constrained")
    positions = np.arange(len(columns))
    width = .8/len(local_gradients)
    for index, (name, values) in enumerate(local_gradients.items()):
        axes[0].bar(positions + (index-(len(local_gradients)-1)/2)*width, values, width=width, label=name)
    axes[0].set(title="Signed gradients at the measured incumbent", ylabel=target_label + " per unit input")
    axes[0].legend(fontsize=8)
    axes[1].bar(positions, seed_mean, yerr=seed_sd, capsize=3, color="#087f8c")
    axes[1].set(title="NN mean absolute gradient at observed inputs", ylabel=target_label + " per unit input")
    axes[2].bar(positions, probability_gradient, color="#8a5cb5")
    axes[2].set(title="NN probability gradient near its fitted boundary", ylabel="Change in P(good) per unit input")
    for ax in axes:
        ax.set(xticks=positions, xticklabels=columns)
        ax.axhline(0, color="#333333", linewidth=.6)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(title + "\nOriginal input units; middle bars show seed variation. Sampled sensitivity is not global or causal importance.")
    return finish_figure(fig, save_to)


def conditional_boundaries(models, inputs, labels, anchor, axes_to_vary, *, title,
                           side=100, save_to=None):
    """Full squares in 2D; higher-dimensional views retain fixed other coordinates."""
    a, b = axes_to_vary
    coordinates = np.linspace(0, .999999, side)
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.tile(anchor, (side*side, 1))
    points[:, a], points[:, b] = xx.ravel(), yy.ravel()
    held = [axis for axis in range(inputs.shape[1]) if axis not in axes_to_vary]
    on_plane = np.all(np.isclose(inputs[:, held], anchor[held], atol=1e-10, rtol=0), axis=1) if held else np.ones(len(inputs), bool)
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5.5), layout="constrained")
    for ax, (name, model) in zip(axes, models.items()):
        if hasattr(model, "predict_proba"):
            values = model.predict_proba(points)[:, 1].reshape(xx.shape)
            image = ax.pcolormesh(xx, yy, values, shading="auto", cmap="viridis", vmin=0, vmax=1)
            level, colour_label = .5, "Fitted P(good)"
        else:
            values = model.decision_function(points).reshape(xx.shape)
            extent = max(float(np.abs(values).max()), .01)
            image = ax.pcolormesh(xx, yy, values, shading="auto", cmap="coolwarm", vmin=-extent, vmax=extent)
            level, colour_label = 0., "SVM decision score (not probability)"
        if values.min() < level < values.max():
            ax.contour(xx, yy, values, levels=[level], colors="white", linewidths=2)
        else:
            ax.text(.5, .02, "No boundary crossing in this slice", transform=ax.transAxes, ha="center", fontsize=8,
                    bbox=dict(facecolor="white", alpha=.8, edgecolor="none"))
        for label, marker, colour in ((0, "o", "#e5a142"), (1, "^", "#e84455")):
            mask = on_plane & (labels == label)
            ax.scatter(inputs[mask, a], inputs[mask, b], marker=marker, c=colour, edgecolor="black", s=55,
                       label=f"Observed {'good' if label else 'bad'} on this slice")
        ax.set(xlabel=f"x{a+1}", ylabel=f"x{b+1}", title=fill(name, 30), xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
        ax.legend(fontsize=6, loc="upper right")
        fig.colorbar(image, ax=ax, orientation="horizontal", pad=.12, label=colour_label)
    settings = ", ".join(f"x{i+1}={anchor[i]:.3f}" for i in held)
    caption = fill("Held fixed: " + settings, 120) if held else "Both inputs vary: this is the full input square."
    fig.suptitle(title + "\n" + caption + f"\nOnly {on_plane.sum()} on-slice observations are shown; boundaries are fitted, not known truth.")
    return finish_figure(fig, save_to)


def support_profiles(support, columns, *, title, save_to=None):
    fig, ax = plt.subplots(figsize=(max(8, len(columns)*1.25), 5.5), layout="constrained")
    values = support[columns].to_numpy()
    image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    labels = [f"{row.observation_id} | class {int(row['Good class'])} | score {row['Decision score']:.3f}"
              for _, row in support.iterrows()]
    ax.set(xticks=np.arange(len(columns)), xticklabels=columns, yticks=np.arange(len(support)), yticklabels=labels)
    ax.tick_params(axis="y", labelsize=8)
    for i in range(len(support)):
        for j in range(len(columns)):
            ax.text(j, i, f"{values[i,j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if values[i,j] > .65 else "#142d42")
    fig.colorbar(image, ax=ax, label="Input setting; colour does not encode output")
    fig.suptitle(title + "\nActual fitted RBF-SVM support vectors, ordered by absolute decision score; every input is retained")
    return finish_figure(fig, save_to)


def gradient_paths(paths, *, title, target_label, save_to=None):
    fig, ax = plt.subplots(figsize=(9, 4.7), layout="constrained")
    for name, path in paths.items():
        ax.plot(path.step, path.prediction, "o-", label=name)
    ax.set(xlabel="Bounded surrogate-only step", ylabel="NN predicted " + target_label)
    ax.grid(alpha=.2)
    ax.legend()
    fig.suptitle(title + "\nPositive direction for maximisation; negative direction demonstrates minimisation. No function evaluations.")
    return finish_figure(fig, save_to)

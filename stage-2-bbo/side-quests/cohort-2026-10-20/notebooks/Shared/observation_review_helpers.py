"""Measured-output, round-progress and input-profile plots for weekly reviews."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finish(figure, save_to):
    if save_to is not None:
        figure.savefig(Path(save_to), dpi=160, bbox_inches="tight")
    plt.close(figure)
    return figure


def plot_output_review(review, *, save_to=None):
    """Show every signed output, then an explicitly labelled highest-five subset."""
    data = review["data"]
    values = data[review["output_column"]].to_numpy(dtype=float)
    rounds = data[review["round_column"]].to_numpy(dtype=int)
    latest = np.asarray(review["latest_mask"], dtype=bool)
    incumbent = values == values.max()
    highest = np.argsort(values, kind="stable")[-5:]
    figure, axes = plt.subplots(2, 1, figsize=(12, 6.2))

    for axis, positions, title in (
        (axes[0], np.arange(len(values)), f"All {len(values)} measured outputs"),
        (axes[1], highest, f"Close-up: highest {len(highest)} of {len(values)} outputs (subset)"),
    ):
        for mask, colour, marker, size, label in (
            (rounds == 0, "#4678a0", "o", 42, "Initial supplied observations"),
            (rounds > 0, "#bd7720", "D", 58, "Returned queries"),
            (incumbent, "#23734f", "o", 170, "Incumbent (highest observed)"),
            (latest, "#c6403b", "*", 190, "Latest return"),
        ):
            selected = positions[mask[positions]]
            if not len(selected):
                continue
            options = {"facecolors": "none", "linewidths": 2} if label.startswith("Incumbent") else {}
            axis.scatter(values[selected], np.zeros(len(selected)), color=colour,
                         marker=marker, s=size, label=label, zorder=3, **options)
        shown = values[positions]
        padding = max(float(np.ptp(shown)) * 0.07, 0.1)
        axis.set(xlim=(shown.min() - padding, shown.max() + padding),
                 ylim=(-0.30, 1.0), yticks=[], xlabel="Signed output y (linear scale)")
        axis.set_title(title, loc="left", fontsize=11, pad=12)
        axis.spines[["left", "right", "top"]].set_visible(False)
        axis.spines["bottom"].set_position(("data", 0))
        axis.spines["bottom"].set_color("#64748b")
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.xaxis.set_label_coords(0.5, -0.10)
        if axis.get_xlim()[0] <= 0 <= axis.get_xlim()[1]:
            axis.axvline(0, color="grey", linestyle=":", linewidth=0.8, zorder=1)

    for position, text_x in zip(highest, np.linspace(0.08, 0.92, len(highest))):
        label = f"round-{rounds[position]:02d}" if rounds[position] > 0 else "Initial"
        if incumbent[position]:
            label += " (incumbent)"
        if latest[position]:
            label += " (latest)"
        axes[1].annotate(f"{label}\ny = {values[position]:.6f}",
                         xy=(values[position], 0), xycoords="data",
                         xytext=(text_x, 0.73), textcoords="axes fraction",
                         ha="center", va="top", fontsize=9,
                         arrowprops={"arrowstyle": "-", "color": "#94a3b8", "linewidth": 0.8})
    rows = review["latest_submissions"]
    caption = "; ".join(
        f"Latest y = {row[review['output_column']]:.6f}; rank {int(row['output_rank'])}/{len(values)}"
        for _, row in rows.iterrows()
    )
    if caption:
        caption += " (1 = highest output)."
    figure.suptitle("Where the returned results sit", fontsize=15, y=0.99)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(0.5, 0.94), ncol=4, frameon=False, fontsize=9)
    figure.text(0.5, 0.02, caption + "  Better outputs run left to right.",
                 ha="center", fontsize=10)
    figure.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.14, hspace=0.80)
    return _finish(figure, save_to)


def plot_round_progress(progress, *, save_to=None):
    """Trace returned outputs and the incumbent from one initial-batch baseline."""
    required = {"source_round", "stage", "returned_y", "best_y"}
    if progress.empty or not required.issubset(progress.columns):
        raise ValueError("Progress needs source_round, stage, returned_y and best_y.")
    progress = progress.sort_values("source_round")
    rounds = progress["source_round"].to_numpy(dtype=int)
    best = progress["best_y"].to_numpy(dtype=float)
    returned = progress["returned_y"].to_numpy(dtype=float)
    if not np.isfinite(best).all() or len(set(rounds)) != len(rounds):
        raise ValueError("Progress needs finite best outputs and one row per round.")
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.step(np.r_[rounds, rounds[-1] + 0.25], np.r_[best, best[-1]], where="post",
              color="#23734f", linewidth=2, label="Best observed after each round")
    axis.scatter(rounds, best, s=65, color="#23734f", zorder=3)
    observed = np.isfinite(returned)
    axis.scatter(rounds[observed], returned[observed], s=85, marker="D",
                 color="#bd7720", zorder=4, label="Best published return in round")
    if observed.any():
        last = np.flatnonzero(observed)[-1]
        axis.scatter(rounds[last], returned[last], s=190, marker="*", color="#c6403b",
                     edgecolors="#263238", linewidths=0.5, zorder=5, label="Latest return")
    for position, value in enumerate(returned):
        if np.isfinite(value):
            offset = -24 if value < best[position] else 13
            axis.annotate(f"{value:.6f}", (rounds[position], value),
                          xytext=(0, offset), textcoords="offset points", ha="center", fontsize=9)
    axis.annotate(f"Initial best: {best[0]:.6f}", (rounds[0], best[0]),
                  xytext=(8, 12), textcoords="offset points", fontsize=9)
    axis.set(xticks=rounds, xticklabels=progress["stage"].astype(str),
             ylabel="Signed output y (higher is better)")
    axis.set_xlim(rounds[0] - 0.2, rounds[-1] + 0.4)
    axis.margins(y=0.22)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["right", "top"]].set_visible(False)
    figure.suptitle("Progress from the initial batch through returned queries", fontsize=13, y=0.99)
    figure.legend(*axis.get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(0.5, 0.93), ncol=3, fontsize=9, frameon=False)
    figure.text(0.5, 0.02, "The initial observations form one starting batch; their row order is not a query history.",
                 ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.06, 1, 0.86))
    return _finish(figure, save_to)


def plot_input_profiles(strongest, *, input_columns, latest_id, incumbent_id, save_to=None):
    """Compare measured input settings; colours encode inputs, never performance."""
    input_columns = list(input_columns)
    if strongest.empty or not {"observation_id", "y", *input_columns}.issubset(strongest.columns):
        raise ValueError("Strongest observations need IDs, outputs and the requested inputs.")
    inputs = strongest[input_columns].to_numpy(dtype=float)
    if not input_columns or not np.isfinite(inputs).all() or np.any((inputs < 0) | (inputs > 1)):
        raise ValueError("Input settings must be finite values in [0, 1].")
    labels = []
    for rank, (_, row) in enumerate(strongest.iterrows(), start=1):
        flags = []
        if row["observation_id"] == incumbent_id:
            flags.append("incumbent")
        if row["observation_id"] == latest_id:
            flags.append("latest")
        suffix = f" ({', '.join(flags)})" if flags else ""
        labels.append(f"{rank}. {row['observation_id']}{suffix}\ny = {row['y']:.6f}")
    figure, axis = plt.subplots(figsize=(max(8, len(input_columns) * 1.05 + 4), 5.3))
    image = axis.imshow(inputs, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (row, column), value in np.ndenumerate(inputs):
        axis.text(column, row, f"{value:.3f}", ha="center", va="center",
                   color="white" if value > 0.58 else "#172d40", fontsize=11)
    axis.set(xticks=np.arange(len(input_columns)), xticklabels=input_columns,
             yticks=np.arange(len(labels)), yticklabels=labels,
             title=f"Input profiles of the {len(strongest)} highest observed outputs")
    axis.xaxis.tick_top()
    axis.tick_params(length=0, pad=9)
    axis.set_xticks(np.arange(-0.5, len(input_columns)), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels)), minor=True)
    axis.grid(which="minor", color="white", linewidth=2)
    axis.tick_params(which="minor", length=0)
    colourbar = figure.colorbar(image, ax=axis, orientation="horizontal", pad=0.07, shrink=0.8, aspect=35)
    colourbar.set_label("Input setting in [0, 1] (colour does not encode output)")
    figure.tight_layout()
    return _finish(figure, save_to)

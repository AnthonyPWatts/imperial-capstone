"""Build introductory diagrams and a progress chart from published result records."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator


STAGE = Path(__file__).resolve().parents[1]
INK = "#183749"
TEAL = "#007f86"
GOLD = "#b76b00"
PURPLE = "#7444ac"
MUTED = "#486271"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": "#b9c9d0",
    "savefig.facecolor": "white",
})


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    fig.savefig(target, dpi=150, facecolor="white")
    plt.close(fig)
    print(target.relative_to(STAGE))


def experiment_cycle():
    fig, ax = plt.subplots(figsize=(13.2, 3.5))
    fig.subplots_adjust(left=0.015, right=0.985, top=0.96, bottom=0.04)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    cards = [
        (0.02, "Review", "Earlier observations"),
        (0.27, "Choose", "One new input"),
        (0.52, "Submit", "To the course portal"),
        (0.77, "Receive", "A returned score"),
    ]
    for x, title, subtitle in cards:
        ax.add_patch(FancyBboxPatch(
            (x, 0.45), 0.21, 0.40,
            boxstyle="round,pad=0.009,rounding_size=0.02",
            linewidth=1.5, edgecolor=TEAL, facecolor="#edf7f7",
        ))
        ax.text(x + 0.105, 0.71, title, ha="center", va="center",
                fontsize=20, weight="bold")
        ax.text(x + 0.105, 0.56, subtitle, ha="center", va="center",
                fontsize=14)
        if x < 0.77:
            ax.add_patch(FancyArrowPatch(
                (x + 0.223, 0.65), (x + 0.237, 0.65),
                arrowstyle="-|>", mutation_scale=19, linewidth=2,
                color=TEAL,
            ))
    ax.plot([0.875, 0.875, 0.125], [0.42, 0.22, 0.22], color=TEAL, lw=1.8)
    ax.add_patch(FancyArrowPatch(
        (0.125, 0.22), (0.125, 0.42), arrowstyle="-|>",
        mutation_scale=19, linewidth=1.8, color=TEAL,
    ))
    ax.text(0.5, 0.08, "Use the new evidence in the next weekly review",
            ha="center", va="center", fontsize=16)
    save(fig, STAGE / "docs/figures", "experiment-cycle.png")


def explore_and_refine():
    # Invented locations: this diagram makes no claim about a course function.
    points = [
        (0.12, 0.18), (0.17, 0.46), (0.28, 0.68), (0.20, 0.76),
        (0.38, 0.62), (0.43, 0.86), (0.53, 0.49), (0.61, 0.78),
        (0.72, 0.91), (0.91, 0.70), (0.47, 0.12),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.8))
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.16, top=0.76, wspace=0.30)
    handles = []
    for ax, title, candidate in zip(
        axes,
        ["Explore a gap", "Refine a promising region"],
        [(0.82, 0.30), (0.34, 0.73)],
    ):
        old = ax.scatter(*zip(*points), s=80, color=INK, zorder=3)
        best = ax.scatter(0.28, 0.68, s=160, color=GOLD,
                          edgecolors="white", linewidths=1.5, zorder=4)
        next_test = ax.scatter(*candidate, s=190, marker="D",
                               facecolors="white", edgecolors=PURPLE,
                               linewidths=2.5, zorder=5)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Setting 1", ylabel="Setting 2")
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(color="#dce5e9", linewidth=0.8)
        ax.set_title(title, fontsize=18, pad=15, weight="bold")
        handles = [old, best, next_test]
    fig.legend(handles, ["Earlier tests", "Best score seen", "Possible next test"],
               loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3,
               frameon=False, fontsize=14, handletextpad=0.5, columnspacing=1.3)
    fig.text(0.5, 0.055, "Illustrative locations, not course observations or a known optimum",
             ha="center", fontsize=13, color=MUTED)
    save(fig, STAGE / "docs/figures", "explore-and-refine.png")


def value_label(value):
    if 0 < abs(value) < 0.001:
        return f"{value:.3e}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.4g}"


def best_observed_progress():
    records = [json.loads((STAGE / f"submissions/round-{i:02d}-results.json").read_text())
               for i in (1, 2)]
    by_round = [{row["function"]: row for row in record["results"]} for record in records]
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.89, bottom=0.07,
                        hspace=0.73, wspace=0.35)
    fig.suptitle("Best score observed so far", x=0.54, y=0.975,
                 fontsize=24, weight="bold")
    fig.text(0.54, 0.935, "Two confirmed rounds · each function has its own output scale",
             ha="center", fontsize=15, color=MUTED)
    for function, ax in enumerate(axes.flat, 1):
        first, second = [rows[function] for rows in by_round]
        if second["best_before"] != first["best_after"]:
            raise ValueError(f"Inconsistent round records for Function {function}")
        values = [first["best_before"], first["best_after"], second["best_after"]]
        unchanged = values[0] == values[-1]
        colour = MUTED if unchanged else TEAL
        ax.plot([0, 1, 2], values, "o-", color=colour, lw=2.5, markersize=8)
        ax.set_title(f"Function {function}", loc="left", fontsize=17, weight="bold", pad=13)
        ax.text(1, 1.08, "Unchanged" if unchanged else "Improved", transform=ax.transAxes,
                ha="right", fontsize=12, color=colour)
        ax.set_xticks([0, 1, 2], ["Start", "Round 1", "Round 2"], fontsize=12)
        ax.set_xlim(-0.13, 2.13)
        if unchanged:
            padding = abs(values[0]) * 0.025 or 0.025
            ax.set_ylim(values[0] - padding, values[0] + padding)
            ax.set_yticks([values[0]])
        else:
            gap = max(values) - min(values)
            ax.set_ylim(min(values) - gap * 0.14, max(values) + gap * 0.14)
            ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, position: value_label(value)))
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", color="#dce5e9")
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.54, 0.025, "Source: published round 1 and 2 records · Week 3 results not yet recorded",
             ha="center", fontsize=13, color=MUTED)
    save(fig, STAGE / "results/figures", "best-observed-progress.png")


if __name__ == "__main__":
    experiment_cycle()
    explore_and_refine()
    best_observed_progress()

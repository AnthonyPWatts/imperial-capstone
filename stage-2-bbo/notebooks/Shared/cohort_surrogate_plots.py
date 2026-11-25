"""Validation and data diagnostics for the Week 9 cohort surrogate study."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from cohort_surrogate_study import GP_SELECTION, nested_summary
from model_helpers import finish_figure


_COLOURS = ("#255e85", "#b76227")


def cohort_data_diagnostics(review, official_target, peer_target, *, title, target_label, save_to=None):
    """Show group shifts on the same input and score axes and count portal repeats."""
    columns = review["input_columns"]
    x = review["own_data"][columns].to_numpy(float)
    peer_x = review["data"].loc[review["peer_mask"], columns].to_numpy(float)
    all_x = np.vstack([x, peer_x]).round(6)
    _, counts = np.unique(all_x, axis=0, return_counts=True)
    distance = cdist(peer_x, x).min(axis=1)
    same_input = np.all(peer_x.round(6)[:, None, :] == x.round(6)[None, :, :], axis=2).any(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    positions = np.arange(len(columns))
    rows = []
    for name, points, scores, colour, offset in (
        ("Official", x, np.asarray(official_target), _COLOURS[0], -.12),
        ("Published peers", peer_x, np.asarray(peer_target), _COLOURS[1], .12),
    ):
        q1, median, q3 = np.quantile(points, [.25, .5, .75], axis=0)
        axes[0].errorbar(median, positions + offset, xerr=[median - q1, q3 - median],
                         fmt="o", capsize=4, label=name, color=colour)
        sorted_scores = np.sort(scores)
        axes[1].step(sorted_scores, np.arange(1, len(scores) + 1) / len(scores), where="post",
                     label=name, color=colour)
        endpoint = np.any(np.isin(points.round(6), [0., .999999]), axis=1)
        rows.append(dict(scope=name, rows=len(points), rows_on_portal_boundary=int(endpoint.sum()),
                         score_median=float(np.median(scores))))
    axes[0].set(xlim=(-.025, 1.025), yticks=positions, yticklabels=columns,
                xlabel="Coordinate value", title="Input medians and middle 50%")
    axes[0].invert_yaxis()
    axes[1].set(xlabel=target_label, ylabel="Cumulative fraction",
                ylim=(-.02, 1.04), title="Empirical score distributions")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.legend(frameon=False, loc="best")
    fig.suptitle(title)
    audit = dict(portal_input_groups=len(counts), repeated_input_groups=int((counts > 1).sum()),
                 peer_rows_matching_official=int(same_input.sum()),
                 median_peer_to_official_distance=float(np.median(distance)), by_scope=rows)
    return audit, finish_figure(fig, save_to)


def plot_surrogate_screen(study, *, title, save_to=None):
    """Highlight the best fixed LOO screens and retain the matching GP reference."""
    records = pd.DataFrame(study["fixed_screen"]).set_index("method")
    baseline = study["reference_model"]
    show = records.iloc[:9]
    if baseline not in show.index:
        show = pd.concat([show, records.loc[[baseline]]])
    ratios = (show.error / records.loc[baseline, "error"]).to_numpy()
    fig, axis = plt.subplots(figsize=(9.4, 5.2), layout="constrained")
    indices = np.arange(len(show))
    axis.barh(indices, ratios, color=np.where(ratios < 1., "#35806b", "#8196a8"))
    axis.axvline(1., color="#222f3d", linestyle="--", linewidth=1.1)
    for i, ratio in enumerate(ratios):
        axis.text(ratio + .015, i, f"{ratio:.2f}×", va="center", fontsize=9)
    axis.set(yticks=indices, yticklabels=show.index, xlabel=f"Official LOO {study['metric']} relative to the saved GP",
             xlim=(0, max(1.17, float(ratios.max()) + .14)))
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    axis.set_axisbelow(True)
    axis.set_title(title)
    return finish_figure(fig, save_to)


def plot_nested_comparison(study, *, title, save_to=None):
    """Compare repeat-specific errors and top-choice gaps on untouched outer folds."""
    summary = nested_summary(study)
    categories = summary.policy.tolist()
    y = np.arange(len(categories))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    repeats = study["nested"]
    width = .72 / len(repeats)
    ratios = []
    for index, repeat in enumerate(repeats):
        seed = repeat["seed"]
        error = summary[f"seed_{seed}_error"]
        baseline = repeat["policies"][GP_SELECTION]["error"]
        ratio = error / baseline
        ratios.extend(ratio.tolist())
        axes[0].barh(y + (index - (len(repeats) - 1) / 2) * width, ratio, width * .90,
                     label=f"Split seed {seed}", color=_COLOURS[index % len(_COLOURS)])
    axes[0].axvline(1., color="#222f3d", linestyle="--", linewidth=1)
    axes[0].set(xlabel=f"Outer {study['metric']} relative to nested GP selection",
                xlim=(0, max(1.10, max(ratios) * 1.08)), title="Prediction error across two splits")
    axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.16), ncol=2)
    values = summary.mean_selection_gap.to_numpy()
    axes[1].barh(y, values, color="#35806b")
    max_gap = max(float(values.max()), np.finfo(float).eps)
    total_folds = sum(len(rep["folds"]) for rep in repeats)
    for i, (gap, best) in enumerate(zip(values, summary.best_picks, strict=True)):
        axes[1].text(gap + max_gap * .018, i, f"{best}/{total_folds} best", va="center", fontsize=9)
    axes[1].set(xlim=(0, max_gap * 1.35), xlabel="Mean gap from best observed score in held-out shortlist",
                title="Retrospective ranking on the same folds")
    for axis in axes:
        axis.set(yticks=y, yticklabels=categories)
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
    fig.suptitle(title)
    return finish_figure(fig, save_to)

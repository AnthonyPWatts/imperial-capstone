
"""Conditional pairwise views of the owner model and its cohort advisers."""

import matplotlib.patheffects as effects
import matplotlib.pyplot as plt
import numpy as np

from cohort_advisory_bo import SETTINGS, owner_posterior, residuals
from weekly_peak_review import PORTAL_MAX


def pairwise_slice(function, result, pair):
    """Hold other coordinates at the selected candidate and plot one pair."""
    radius = SETTINGS[function][2]
    anchor = np.array(result["chosen"].split("-"), dtype=float)
    incumbent = result["incumbent"]
    archived = np.array(result["comparison"].loc[1, "Best observed input"].split("-"), dtype=float)
    reference = np.vstack([anchor, incumbent, archived])
    low = np.maximum(reference.min(axis=0) - radius, 0.)
    high = np.minimum(reference.max(axis=0) + radius, PORTAL_MAX)
    halo = [effects.withStroke(linewidth=2.6, foreground="white")]
    i, j = pair
    xx, yy = np.meshgrid(np.linspace(low[i], high[i], 41), np.linspace(low[j], high[j], 41))
    coordinates = np.tile(anchor, (xx.size, 1))
    coordinates[:, i], coordinates[:, j] = xx.ravel(), yy.ravel()
    mean, sd = owner_posterior(result["model"], coordinates)
    mean_label = "Official GP – fitted mean"
    if function == 1 and result["model"][1] is not None:
        mean_label = "Official source + GP – fitted mean"
    sd_label = "Official GP – latent standard deviation"
    if result["structural"] is not None:
        structure = result["structural"]
        loss = ((coordinates - structure["point"])**2) @ np.asarray(structure["curvature"])
        mean = structure["predicted_maximum"] - loss
        mean_label = "Official quadratic – fitted mean"
        sd_label = "GP comparator – latent standard deviation"
    change = residuals(result["advisers"], coordinates) * .1
    max_change = max(float(np.max(np.abs(change))), 1e-12)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.8), layout="constrained", sharex=True, sharey=True)
    panels = [
        (mean, mean_label, "viridis", {}),
        (sd, sd_label, "magma", {}),
        (change[:, 0], "Current cohort – 10% mean adjustment", "RdBu_r", {"vmin": -max_change, "vmax": max_change}),
        (change[:, 1], "2024 archive – 10% mean adjustment", "RdBu_r", {"vmin": -max_change, "vmax": max_change}),
    ]
    for ax, (values, title, cmap, limits) in zip(axes.flat, panels, strict=True):
        surface = ax.pcolormesh(xx, yy, values.reshape(xx.shape), cmap=cmap, shading="auto", **limits)
        fig.colorbar(surface, ax=ax, format="%.3g")
        ax.scatter(anchor[i], anchor[j], marker="*", s=106, edgecolors="#141414", linewidths=.7,
                   color="#fafafa", clip_on=False, label="Selected point")
        ax.scatter(incumbent[i], incumbent[j], marker="o", s=49, facecolors="none", linewidths=1.2,
                   edgecolors="#161616", clip_on=False, path_effects=halo, label="Incumbent projection")
        if np.any(archived[[i, j]] != incumbent[[i, j]]) and np.any(archived[[i, j]] != anchor[[i, j]]):
            ax.scatter(archived[i], archived[j], marker="+", s=74, color="#161616", linewidths=1.2,
                       clip_on=False, path_effects=halo, label="2024 best projection")
        ax.set(xlim=(low[i], high[i]), ylim=(low[j], high[j]))
        ax.set_title(title, fontsize=10)
        ax.ticklabel_format(useOffset=False)
    for ax in axes[-1]:
        ax.set_xlabel(f"x{i + 1}")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"x{j + 1}")
    fig.suptitle(f"Function {function} · x{i + 1} and x{j + 1}", fontsize=14)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center",
               ncols=3, frameon=False)
    return fig

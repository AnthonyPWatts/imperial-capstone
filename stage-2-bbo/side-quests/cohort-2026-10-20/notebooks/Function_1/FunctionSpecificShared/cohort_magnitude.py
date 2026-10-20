"""Absolute-magnitude views for Function 1's pooled-data side quest only."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import numpy as np
import plotly.graph_objects as go

from function_1_model_helpers import signed_log
from plot_helpers import PLOT_CONFIG, plot_observed_terrain


SIGNS = ((1, "Positive", "o", "circle", "#2475b0"),
         (-1, "Negative", "s", "square", "#bf681d"),
         (0, "Exact zero", "x", "x", "#64748b"))


def _values(data):
    inputs = data[["x1", "x2"]].to_numpy(dtype=float)
    outputs = data["y"].to_numpy(dtype=float)
    if not len(outputs) or not np.isfinite(np.column_stack([inputs, outputs])).all():
        raise ValueError("Supply finite x1, x2 and y observations.")
    return inputs, outputs, np.abs(outputs)


def plot_magnitude_map(data, *, save_to=None):
    """Colour both signs by one common logarithmic absolute-magnitude scale."""
    inputs, outputs, magnitudes = _values(data)
    nonzero = magnitudes > 0
    figure, axis = plt.subplots(figsize=(8.5, 6.5), layout="constrained")
    norm = None
    if nonzero.any():
        lower, upper = np.floor(np.log10(magnitudes[nonzero].min())), np.ceil(np.log10(magnitudes.max()))
        if lower == upper:
            lower, upper = lower - 1, upper + 1
        norm = LogNorm(vmin=10.0 ** lower, vmax=10.0 ** upper)
    for sign, label, marker, _, colour in SIGNS:
        mask = np.sign(outputs) == sign
        if not mask.any():
            continue
        if sign:
            axis.scatter(*inputs[mask].T, c=magnitudes[mask], norm=norm, cmap="viridis",
                         marker=marker, s=100, edgecolors="#172d40", linewidths=0.7,
                         label=f"{label} ({mask.sum()})", zorder=3, clip_on=False)
        else:
            axis.scatter(*inputs[mask].T, color=colour, marker=marker, s=100,
                         label=f"{label} ({mask.sum()}; outside log scale)", zorder=3)
    for sign, text, offset in ((1, "Strongest positive", (20, 0)),
                               (-1, "Strongest negative", (-12, -35))):
        positions = np.flatnonzero(np.sign(outputs) == sign)
        if len(positions):
            position = positions[np.argmax(magnitudes[positions])]
            axis.annotate(text, inputs[position], xytext=offset, textcoords="offset points",
                          ha="left" if sign > 0 else "right", fontsize=10,
                          arrowprops=dict(arrowstyle="-", color="#475569"))
    if norm is not None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        step = max(1, int(np.ceil((upper - lower) / 5)))
        ticks = 10.0 ** np.unique(np.r_[lower, np.arange(np.ceil(lower / step) * step, upper, step), upper])
        colourbar = figure.colorbar(mappable, ax=axis, ticks=ticks, shrink=0.83, pad=0.04)
        colourbar.ax.yaxis.set_major_formatter(LogFormatterMathtext(labelOnlyBase=False))
        colourbar.set_label("Absolute output |y| (logarithmic colour scale)")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="x1", ylabel="x2", aspect="equal",
             title="Function 1: where are the strongest departures from zero?")
    axis.grid(alpha=0.18)
    legend = axis.legend(loc="lower left", framealpha=0.95)
    for handle in legend.legend_handles:
        handle.set_array(None)
        handle.set_facecolor("#e2e8f0")
        handle.set_edgecolor("#172d40")
    figure.supxlabel("Shape preserves the sign; colour shows |y| on the same scale for both signs.\n"
                     "Observed points only: no fitted or interpolated background.", fontsize=10)
    if save_to is not None:
        figure.savefig(Path(save_to), dpi=160, bbox_inches="tight")
    plt.close(figure)
    return figure


def plot_magnitude_terrain(data, *, scale, save_to=None):
    """Fold signed heights upwards while retaining original signed hover values."""
    inputs, outputs, magnitudes = _values(data)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The magnitude display scale must be positive and finite.")
    ticks = np.array([0.0, 1e-100, 1e-3])
    figure = plot_observed_terrain(
        inputs, magnitudes, title="Function 1: absolute-magnitude terrain",
        heights=signed_log(magnitudes, scale=scale),
        height_axis=dict(title="|y| (log height)", tickmode="array",
                         tickvals=signed_log(ticks, scale=scale), ticktext=["0", "1e-100", "1e-3"]),
        relief=0.16, width=850, height=560,
        camera=dict(eye=dict(x=-1.05, y=-1.35, z=0.9)), show_reset=True,
    )
    mesh, strongest = figure.data[0], figure.data[2]
    mesh.update(name="Joined absolute magnitudes", showlegend=False)
    best = int(np.argmax(magnitudes))
    strongest.update(name="Largest |y|", customdata=[[float(outputs[best]), float(magnitudes[best])]],
                     hovertemplate="Largest |y|<br>Signed y=%{customdata[0]:.3e}<br>|y|=%{customdata[1]:.3e}<extra></extra>")
    figure.data = (mesh,)
    ids = data["observation_id"].astype(str).to_numpy()
    for sign, label, _, marker, colour in SIGNS:
        mask = np.sign(outputs) == sign
        if not mask.any():
            continue
        customdata = [[str(ids[i]), float(outputs[i]), float(magnitudes[i])]
                      for i in np.flatnonzero(mask)]
        figure.add_trace(go.Scatter3d(
            x=inputs[mask, 0], y=inputs[mask, 1], z=signed_log(magnitudes[mask], scale=scale),
            mode="markers", marker=dict(size=4, color=colour, symbol=marker),
            name=f"{label} ({mask.sum()})", customdata=customdata,
            hovertemplate="%{customdata[0]}<br>x1=%{x:.6f}<br>x2=%{y:.6f}"
                          "<br>Signed y=%{customdata[1]:.3e}<br>|y|=%{customdata[2]:.3e}<extra></extra>",
        ))
    figure.add_trace(strongest)
    figure.update_layout(legend=dict(x=0, y=1.02, orientation="h", yanchor="bottom"))
    figure.layout.annotations[0].text = (
        "Faces join observed log-magnitude heights; the gaps between readings are unmeasured."
        "<br>Higher means larger |y|, not a better signed optimisation score."
    )
    if save_to is not None:
        figure.write_html(Path(save_to), include_plotlyjs=True, config={**PLOT_CONFIG, "responsive": False})
    return figure

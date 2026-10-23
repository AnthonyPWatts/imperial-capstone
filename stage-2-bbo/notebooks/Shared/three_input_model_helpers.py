"""Coordinate curves, plane sections and input clouds for the Week 4 3D study."""

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import Delaunay

from model_helpers import finish_figure, gp_slice, plot_posterior_curve
from plot_helpers import PLOT_CONFIG, TERRAIN_COLOURS


INPUT_NAMES = ("x1", "x2", "x3")
PLANE_AXES = ((0, 1, 2), (0, 2, 1), (1, 2, 0))


def input_cloud(inputs, outputs, *, title, anchor=None, planned=None, suggestion=None,
                save_to=None):
    """All axes are inputs; output is colour. Unevaluated points have fixed colours."""
    inputs, outputs = np.asarray(inputs), np.asarray(outputs)
    fig = go.Figure(go.Scatter3d(
        x=inputs[:, 0], y=inputs[:, 1], z=inputs[:, 2], mode="markers",
        marker=dict(size=5, color=outputs, colorscale=TERRAIN_COLOURS,
                    colorbar=dict(title="Observed y"), line=dict(color="black", width=1)),
        customdata=outputs, name="Observed outputs",
        hovertemplate="x1=%{x:.6f}<br>x2=%{y:.6f}<br>x3=%{z:.6f}<br>Observed y=%{customdata:.4f}<extra></extra>",
    ))
    if anchor is not None:
        for axis, colour in enumerate(("tab:blue", "tab:orange", "tab:purple")):
            # Plotly uses CSS colours rather than Matplotlib's tab: names.
            colour = {"tab:blue": "#1f77b4", "tab:orange": "#ff7f0e", "tab:purple": "#9467bd"}[colour]
            points = np.tile(anchor, (2, 1))
            points[:, axis] = [0, 0.999999]
            fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode="lines", line=dict(color=colour, width=5), name=f"Vary {INPUT_NAMES[axis]}; hold two inputs"))
        fig.add_trace(go.Scatter3d(x=[anchor[0]], y=[anchor[1]], z=[anchor[2]], mode="markers",
            marker=dict(size=8, color="gold", symbol="diamond"), name="Best observed input"))
    for point, colour, name, symbol in (
        (planned, "gold", "Actual saved coverage point (unevaluated)", "square"),
        (suggestion, "magenta", "Hypothetical GP-UCB point (unevaluated)", "x"),
    ):
        if point is not None:
            fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
                marker=dict(size=8, color=colour, symbol=symbol, line=dict(color="black", width=1)), name=name,
                hovertemplate="x1=%{x:.6f}<br>x2=%{y:.6f}<br>x3=%{z:.6f}<br>Output not measured<extra></extra>"))
    fig.update_layout(title=title, height=650, margin=dict(l=10, r=60, t=80, b=20),
        scene=dict(xaxis=dict(title="x1", range=[0, 1]), yaxis=dict(title="x2", range=[0, 1]),
                   zaxis=dict(title="x3 (input)", range=[0, 1]), aspectmode="cube", dragmode="orbit"),
        legend=dict(x=0, y=1))
    if save_to is not None:
        fig.write_html(save_to, include_plotlyjs=True, config=PLOT_CONFIG)
    return fig


def plot_coordinate_curves(models, inputs, outputs, anchor, *, save_to=None):
    """One input varies per column; all other input coordinates remain fixed."""
    sections = [[gp_slice(model, inputs, outputs, anchor, axis) for axis in range(3)]
                for model in models.values()]
    lower = min(np.min(s["mean"] - 1.96 * s["std"]) for row in sections for s in row)
    upper = max(np.max(s["mean"] + 1.96 * s["std"]) for row in sections for s in row)
    padding = (upper - lower) * 0.05
    fig, axes = plt.subplots(len(models), 3, figsize=(15, 4 * len(models)),
                             sharex=True, sharey=True, squeeze=False)
    for row, (name, _) in enumerate(models.items()):
        for axis in range(3):
            ax, section = axes[row, axis], sections[row][axis]
            plot_posterior_curve(ax, section["coordinates"], section["mean"], section["std"],
                                 section["observed_x"], section["observed_y"])
            held = ", ".join(f"{INPUT_NAMES[i]}={anchor[i]:.4f}" for i in range(3) if i != axis)
            ax.set(xlim=(0, 1), ylim=(lower - padding, upper + padding), xlabel=f"Vary {INPUT_NAMES[axis]}",
                   title=f"{name}\nHold {held}")
            if axis == 0:
                ax.set_ylabel("Output y")
    fig.suptitle("Function 3: repeat the GP picture along three paths through the input cube")
    fig.text(0.5, 0.01, "All panels share vertical limits. Green crosses match both held coordinates; other measurements are not projected onto these curves.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return finish_figure(fig, save_to)


def plane_grid(model, inputs, varying_axes, fixed_value, *, side=61, uncertainty=False):
    """A conditional input plane with support measured in the full 3D hull.

    Predictions remain available outside the hull for explicit uncertainty and
    acquisition displays. Use inside to mask response-contour illustrations.
    """
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim != 2 or inputs.shape[1] != 3:
        raise ValueError("Expected three-input observations")
    if len(varying_axes) != 2 or len(set(varying_axes)) != 2 or not set(varying_axes) <= {0, 1, 2}:
        raise ValueError("Expected two distinct input axes")
    if not np.isfinite(fixed_value) or not 0 <= fixed_value <= 0.999999:
        raise ValueError("fixed_value must be in [0, 0.999999]")
    first, second = varying_axes
    fixed = next(axis for axis in range(3) if axis not in varying_axes)
    axis = np.linspace(0, 0.999999, side)
    u, v = np.meshgrid(axis, axis)
    points = np.empty((u.size, 3))
    points[:, first], points[:, second], points[:, fixed] = u.ravel(), v.ravel(), fixed_value
    if uncertainty:
        mean, std = model.predict(points, return_std=True)
    else:
        mean, std = model.predict(points), None
    inside = Delaunay(inputs).find_simplex(points) >= 0
    return dict(axis=axis, u=u, v=v, points=points, first=first, second=second, fixed=fixed,
                fixed_value=fixed_value, mean=mean.reshape(u.shape), inside=inside.reshape(u.shape),
                std=None if std is None else std.reshape(u.shape))


def build_plane_slices(model, inputs, values, *, side=61):
    return [[plane_grid(model, inputs, (a, b), value, side=side) for a, b, _ in PLANE_AXES] for value in values]


def plot_contour_sections(rows, outputs, *, title, save_to=None):
    """Rows are (label, three grids). All mean contours share output limits."""
    values = np.concatenate([np.asarray(outputs), *[grid["mean"][grid["inside"]]
                              for _, grids in rows for grid in grids]])
    limits = (values.min(), values.max())
    levels = np.linspace(*limits, 13)
    fig, axes = plt.subplots(len(rows), 3, figsize=(13, 3.5 * len(rows) + 0.7),
                             squeeze=False, layout="constrained")
    for row, (name, grids) in enumerate(rows):
        for ax, grid in zip(axes[row], grids):
            heights = np.where(grid["inside"], grid["mean"], np.nan)
            if grid["inside"].any():
                ax.contourf(grid["u"], grid["v"], heights, levels=levels, cmap="terrain")
                lines = ax.contour(grid["u"], grid["v"], heights, levels=levels,
                                   colors="black", linewidths=0.5, alpha=0.6)
                ax.clabel(lines, inline=True, fontsize=7, fmt="%.2f")
            else:
                ax.text(0.5, 0.5, "No sampled hull intersection", ha="center", va="center", transform=ax.transAxes)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=INPUT_NAMES[grid["first"]], ylabel=INPUT_NAMES[grid["second"]],
                   title=f"{name}\nHold {INPUT_NAMES[grid['fixed']]} = {grid['fixed_value']:.4f}")
            ax.set_aspect("equal")
    colour_map = ScalarMappable(norm=Normalize(*limits), cmap="terrain")
    fig.colorbar(colour_map, ax=axes, shrink=0.85, label="Fitted output y (shared scale)")
    fig.suptitle(title)
    fig.supxlabel("Each panel is a model section at its stated held coordinate, with shared colours and contour levels.\n"
                  "Blank areas lie outside the full 3D sampled hull; no off-plane measurements are projected here.", fontsize=9)
    return finish_figure(fig, save_to)


def plot_gp_score_sections(grids, inputs, *, weight=1.96, suggestion=None, title, save_to=None):
    """Compare conditional mean, SD and UCB; mark extrapolation in each input plane."""
    if any(grid["std"] is None for grid in grids):
        raise ValueError("Uncertainty grids are required")
    fields = [[grid["mean"], grid["std"], grid["mean"] + weight * grid["std"]] for grid in grids]
    limits = [(min(np.min(row[column]) for row in fields), max(np.max(row[column]) for row in fields))
              for column in range(3)]
    names = ("GP mean (y)", "GP standard deviation", f"UCB = mean + {weight:g} x SD")
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.11, top=0.91, wspace=0.50, hspace=0.55)
    for row, grid in enumerate(grids):
        for column, name in enumerate(names):
            ax = axes[row, column]
            ax.contourf(grid["u"], grid["v"], fields[row][column], levels=np.linspace(*limits[column], 16), cmap="viridis")
            if grid["inside"].any() and not grid["inside"].all():
                ax.contour(grid["u"], grid["v"], grid["inside"].astype(float), levels=[0.5],
                           colors="grey", linestyles=":", linewidths=1.2)
            on_plane = np.isclose(inputs[:, grid["fixed"]], grid["fixed_value"], rtol=0, atol=1e-10)
            ax.scatter(inputs[on_plane, grid["first"]], inputs[on_plane, grid["second"]], color="black", s=25)
            if suggestion is not None and np.isclose(suggestion[grid["fixed"]], grid["fixed_value"], rtol=0, atol=1e-10):
                marker = ax.scatter(suggestion[grid["first"]], suggestion[grid["second"]], marker="X",
                                    color="magenta", edgecolor="black", s=90, clip_on=False, zorder=5)
                marker.set_in_layout(False)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=INPUT_NAMES[grid["first"]], ylabel=INPUT_NAMES[grid["second"]],
                   title=f"{name}\nHold {INPUT_NAMES[grid['fixed']]} = {grid['fixed_value']:.4f}")
            ax.set_aspect("equal")
    for column in range(3):
        fig.colorbar(ScalarMappable(norm=Normalize(*limits[column]), cmap="viridis"),
                     ax=axes[:, column], shrink=0.85, pad=0.04, fraction=0.045, label=names[column])
    fig.suptitle(title)
    fig.supxlabel("Dotted boundary: intersection with the full 3D sampled hull; coloured regions outside it are extrapolations.\n"
                  "Colours are shared down each column. Black dots match the held input; magenta X is the hypothetical query when on-plane.",
                  fontsize=9, y=0.02)
    return finish_figure(fig, save_to)


def interactive_terrain_slices(slices, values, outputs, *, title, save_to=None):
    """Three rotatable sections, using 31x31 display meshes and minimal frames."""
    limits = (min(np.min(outputs), min(np.nanmin(np.where(g["inside"], g["mean"], np.nan))
                                      for row in slices for g in row if g["inside"].any())),
              max(np.max(outputs), max(np.nanmax(np.where(g["inside"], g["mean"], np.nan))
                                      for row in slices for g in row if g["inside"].any())))
    initial = int(np.argmin(np.abs(np.asarray(values) - 0.5)))
    def headings(index):
        return [f"Vary {INPUT_NAMES[a]}, {INPUT_NAMES[b]}; hold {INPUT_NAMES[c]}={values[index]:.2f}" for a, b, c in PLANE_AXES]
    def trace(index, pair, *, initial_trace=False):
        grid = slices[index][pair]
        z = np.where(grid["inside"], grid["mean"], np.nan)[::2, ::2]
        hover = (f"{INPUT_NAMES[grid['first']]}=%{{x:.3f}}<br>{INPUT_NAMES[grid['second']]}=%{{y:.3f}}<br>"
                 f"{INPUT_NAMES[grid['fixed']]}={grid['fixed_value']:.2f}<br>Fitted y=%{{z:.4f}}<extra></extra>")
        if not initial_trace:
            return go.Surface(z=z, hovertemplate=hover)
        return go.Surface(x=grid["axis"][::2], y=grid["axis"][::2], z=z, coloraxis="coloraxis",
                          connectgaps=False, hovertemplate=hover, contours=dict(z=dict(show=True, usecolormap=True)))
    fig = make_subplots(rows=1, cols=3, specs=[[{"type": "surface"}] * 3],
                        horizontal_spacing=0.02, subplot_titles=headings(initial))
    for pair, (first, second, _) in enumerate(PLANE_AXES):
        fig.add_trace(trace(initial, pair, initial_trace=True), row=1, col=pair + 1)
        fig.update_scenes(xaxis=dict(title=INPUT_NAMES[first], range=[0, 1]),
            yaxis=dict(title=INPUT_NAMES[second], range=[0, 1]), zaxis=dict(title="Fitted y", range=list(limits), nticks=3),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.16), dragmode="orbit",
            uirevision="slice-camera", row=1, col=pair + 1)
    frames = []
    for index, value in enumerate(values):
        annotations = [a.to_plotly_json() for a in fig.layout.annotations]
        for annotation, text in zip(annotations, headings(index)):
            annotation["text"] = text
        frames.append(go.Frame(name=f"{value:.2f}", traces=[0, 1, 2],
            data=[trace(index, pair) for pair in range(3)], layout=dict(annotations=annotations)))
    fig.frames = frames
    fig.update_layout(title=title, height=650, margin=dict(l=10, r=90, t=120, b=100),
        coloraxis=dict(colorscale=TERRAIN_COLOURS, cmin=limits[0], cmax=limits[1], colorbar=dict(title="Fitted y", len=0.75)),
        sliders=[dict(active=initial, y=-0.05, currentvalue=dict(prefix="Each held input = "),
            steps=[dict(label=f"{value:.2f}", method="animate", args=[[f"{value:.2f}"],
                dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))]) for value in values])])
    if save_to is not None:
        fig.write_html(save_to, include_plotlyjs=True, config=PLOT_CONFIG)
    return fig


def plot_observed_input_cloud(inputs, outputs, *, title, output_limits=None,
                               colourscale="Plasma", width=850, height=560,
                               best_label="Best observed output", save_to=None):
    """Three input axes and measured output colours; optional limits filter display only."""
    inputs, outputs = np.asarray(inputs), np.asarray(outputs)
    if inputs.ndim != 2 or inputs.shape[1] != 3 or outputs.shape != (len(inputs),):
        raise ValueError("Supply three-input observations and their output vector")
    if output_limits is not None:
        lower, upper = output_limits
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("output_limits must contain two increasing finite values")
        keep = (outputs >= lower) & (outputs <= upper)
        inputs, outputs = inputs[keep], outputs[keep]
    if not len(outputs) or not np.isfinite(inputs).all() or not np.isfinite(outputs).all():
        raise ValueError("The cloud needs at least one finite displayed observation")
    best = int(np.argmax(outputs))
    fig = go.Figure(go.Scatter3d(
        x=inputs[:, 0], y=inputs[:, 1], z=inputs[:, 2], mode="markers",
        marker=dict(size=6, color=outputs, colorscale=colourscale,
                    cmin=outputs.min(), cmax=outputs.max(),
                    colorbar=dict(title="Observed y" if output_limits is None else "Observed y<br>(displayed range)"),
                    line=dict(color="black", width=1)),
        customdata=outputs, name="Observed inputs and outputs",
        hovertemplate="x1=%{x:.4f}<br>x2=%{y:.4f}<br>x3=%{z:.4f}<br>Observed y=%{customdata:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=[inputs[best, 0]], y=[inputs[best, 1]], z=[inputs[best, 2]], mode="markers",
        marker=dict(size=9, color="gold", symbol="diamond", line=dict(color="black", width=2)),
        customdata=[outputs[best]], name=best_label,
        hovertemplate="Best observed y=%{customdata:.4f}<extra></extra>",
    ))
    fig.update_layout(title=title, width=width, height=height, autosize=width is None,
        margin=dict(l=10, r=80, t=70, b=30), legend=dict(x=0, y=1),
        scene=dict(xaxis=dict(title="x1", range=[0, 1]), yaxis=dict(title="x2", range=[0, 1]),
                   zaxis=dict(title="x3 (input)", range=[0, 1]), aspectmode="cube", dragmode="orbit"))
    if save_to is not None:
        fig.write_html(save_to, include_plotlyjs=True, config={**PLOT_CONFIG, "responsive": width is None})
    return fig


def save_three_input_model_views(model_run, inputs, target, *, gp_names, title, figures_dir,
                                 reference_label="Kernel ridge", grid_side=61,
                                 slice_values=(.10, .25, .40, .50, .60, .75, .90)):
    """Export the two fitted GP explanations and full-hull conditional response planes."""
    from pathlib import Path

    figures_dir = Path(figures_dir)
    anchor = np.asarray(inputs)[int(np.argmax(target))].copy()
    fitted = model_run["fitted"]
    curves = plot_coordinate_curves({name: fitted[name] for name in gp_names}, inputs, target, anchor,
                                    save_to=figures_dir / "gp-coordinate-slices.png")
    rows = [(name, [plane_grid(fitted[name], inputs, (a, b), anchor[c], side=grid_side)
                    for a, b, c in PLANE_AXES])
            for name in (reference_label, model_run["selected_name"])]
    contours = plot_contour_sections(rows, target,
        title=f"{title}: fitted output planes through the incumbent",
        save_to=figures_dir / "fitted-output-contour-slices.png")
    slices = build_plane_slices(model_run["selected_model"], inputs, slice_values, side=grid_side)
    terrain = interactive_terrain_slices(slices, slice_values, target,
        title=f"{title}: {model_run['selected_name']} conditional terrains",
        save_to=figures_dir / "fitted-terrain-slices.html")
    return dict(figures=[curves, contours], terrain=terrain, anchor=anchor)


def save_three_input_query_views(model, inputs, target, anchor, proposal, *, title,
                                 figures_dir, weight=1.96, grid_side=61):
    """Export the unevaluated proposal in a 3D cloud and three conditional score planes."""
    from pathlib import Path

    figures_dir = Path(figures_dir)
    cloud = input_cloud(inputs, target, anchor=anchor, suggestion=proposal,
                         title=f"{title}: observed inputs and proposed GP-UCB query")
    for trace in cloud.data:
        if trace.name and "Hypothetical" in trace.name:
            trace.name = trace.name.replace("Hypothetical", "Proposed")
    cloud.write_html(figures_dir / "model-query-input-cloud.html", include_plotlyjs=True, config=PLOT_CONFIG)
    planes = [plane_grid(model, inputs, (a, b), proposal[c], side=grid_side, uncertainty=True)
              for a, b, c in PLANE_AXES]
    scores = plot_gp_score_sections(planes, inputs, weight=weight, suggestion=proposal,
                                    title=f"{title}: conditional query mean, uncertainty and UCB")
    scores.supxlabel("Dotted boundary: full 3D sampled hull. Coloured regions outside it are extrapolations.\n"
                     "Colours are shared down each column. Black dots match the held input; magenta X is the proposed query.",
                     fontsize=9, y=.02)
    scores.savefig(figures_dir / "model-query-proposal-sections.png", dpi=160, bbox_inches="tight")
    return dict(cloud=cloud, scores=scores)


def plot_cohort_comparison(paired, own_inputs, peer_inputs, own_proposal, pooled_proposal,
                            *, title, figsize=(14, 6), save_to=None):
    """Paired target errors beside the explicit three-input observation/proposal map."""
    own_inputs, peer_inputs = np.asarray(own_inputs), np.asarray(peer_inputs)
    if own_inputs.ndim != 2 or peer_inputs.ndim != 2 or own_inputs.shape[1] != 3 or peer_inputs.shape[1] != 3:
        raise ValueError("The cohort input map requires exactly three coordinates")
    fig = plt.figure(figsize=figsize, layout="constrained")
    axes = [fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2, projection="3d")]
    positions = np.arange(len(paired))
    axes[0].barh(positions-.18, paired.iloc[:, 1], height=.34, label="Own training")
    axes[0].barh(positions+.18, paired.iloc[:, 2], height=.34, label="Own + peer training")
    axes[0].set(yticks=positions, yticklabels=paired.Model,
                xlabel=f"RMSE on the same {len(own_inputs)} official targets", title="Paired held-out comparison")
    axes[0].legend(fontsize=8)
    axes[1].scatter(*own_inputs.T, color="#4678a0", label="Official observations", s=30)
    axes[1].scatter(*peer_inputs.T, color="#bd7720", marker="D", label="Added peer observations", s=45)
    axes[1].scatter(*own_proposal, color="purple", marker="X", s=110, label="Official provisional query")
    axes[1].scatter(*pooled_proposal, color="crimson", marker="*", s=150, label="Cohort exploratory query")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xlabel="x1", ylabel="x2", zlabel="x3",
                title="Observed inputs and the two proposals")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle(title)
    return finish_figure(fig, save_to)

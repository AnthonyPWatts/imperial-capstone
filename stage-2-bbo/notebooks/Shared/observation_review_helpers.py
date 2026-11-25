"""Measured-output, round-progress and input-profile plots for weekly reviews."""

import hashlib
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd


def prepare_weekly_observation_review(review, *, history=None, strongest_count=5):
    """Prepare measured rankings and official progress using every input coordinate.

    A pooled review supplies official rows as history. Peer round labels never
    become owner history, while ranks and the incumbent use the complete snapshot.
    """
    data = review["data"]
    history = data if history is None else history
    inputs, output, round_column = review["input_columns"], review["output_column"], review["round_column"]
    if review["latest_mask"].sum() != 1 or strongest_count < 1:
        raise ValueError("Require one latest return and a positive strongest_count")
    latest = data.loc[review["latest_mask"]].iloc[0]
    initial = history.loc[history[round_column].eq(0)]
    returned = history.loc[history[round_column].gt(0)].sort_values(round_column)
    if (initial.empty or returned[round_column].duplicated().any()
            or not history.observation_id.eq(latest.observation_id).any()):
        raise ValueError("History needs an initial batch, one return per round and the latest observation")
    running_best = float(initial[output].max())
    rows = [dict(source_round=0, stage="Initial batch", returned_y=np.nan,
                 best_y=running_best, change_vs_prior_best=np.nan)]
    for _, observation in returned.iterrows():
        value = float(observation[output])
        change = value - running_best
        running_best = max(running_best, value)
        round_number = int(observation[round_column])
        rows.append(dict(source_round=round_number, stage=f"Round {round_number}",
                         returned_y=value, best_y=running_best, change_vs_prior_best=change))
    ranked = data.assign(rank=data[output].rank(ascending=False, method="min").astype(int))
    strongest = ranked.sort_values(["rank", "observation_id"]).head(strongest_count)
    earlier = history.loc[history[round_column].lt(latest[round_column])]
    previous_best = earlier.loc[earlier[output].idxmax()]
    changes = latest[inputs].astype(float) - previous_best[inputs].astype(float)
    distances = np.linalg.norm(earlier[inputs].to_numpy(dtype=float) - latest[inputs].to_numpy(dtype=float), axis=1)
    nearest = earlier.iloc[int(distances.argmin())]
    return dict(latest=latest, incumbent=data.loc[data[output].idxmax()],
        strongest=strongest, progress=pd.DataFrame(rows),
        coordinate_changes=pd.DataFrame({"Previous best": previous_best[inputs].astype(float),
            "Latest query": latest[inputs].astype(float), "Latest minus previous best": changes}),
        neighbour_summary=dict(previous_best_id=previous_best.observation_id,
            previous_best_output=float(previous_best[output]),
            distance_to_previous_best=float(np.linalg.norm(changes.to_numpy())),
            latest_minus_previous_best=float(latest[output]-previous_best[output]),
            nearest_previous_id=nearest.observation_id, nearest_previous_distance=float(distances.min())))


def read_saved_forecast(notebook_path, *, point, precision=6):
    """Read a historical printed forecast only when its saved coordinates match.

    The legacy model notebook prints both fields in one output cell. Preserve
    that recorded forecast; do not reconstruct it using later observations.
    """
    matches = []
    for cell in nbformat.read(Path(notebook_path), as_version=4).cells:
        text = "".join(output.get("text", "") for output in cell.get("outputs", []))
        coordinates = re.search(r"^Coordinates: (.+)$", text, re.MULTILINE)
        prediction = re.search(r"^Predicted output at the proposed query: ([-+\d.eE]+)$", text, re.MULTILINE)
        if coordinates and prediction:
            saved_point = np.asarray([float(value.strip()) for value in coordinates.group(1).split(",")])
            if saved_point.shape == np.asarray(point).shape and np.array_equal(saved_point.round(precision), np.asarray(point).round(precision)):
                matches.append(dict(point=saved_point.tolist(), predicted_output=float(prediction.group(1)),
                                    printed_decimal_places=precision))
    if len(matches) != 1 or not np.isfinite(matches[0]["predicted_output"]):
        raise ValueError("Require one saved forecast matching the actual submitted input")
    return matches[0]


def replay_weekly_forecast(analysis_dir, *, function, week, input_columns, point):
    """Evaluate archived fits at a returned point without retraining on feedback."""
    from neural_explorer import gp_prediction, network_prediction

    folder = Path(analysis_dir)
    model_path = folder / "model-review.json"
    saved = json.loads(model_path.read_text(encoding="utf-8"))
    snapshot_hash = hashlib.sha256((folder / "observations.csv").read_bytes()).hexdigest()
    if (saved["function"], saved["week"], saved["snapshot_sha256"]) != (function, week, snapshot_hash):
        raise ValueError("Archived forecast does not match the declared official snapshot")
    if saved["cohort_audit_sha256"] is not None:
        audit_path = folder.parent / "cohort/data-audit.json"
        if hashlib.sha256(audit_path.read_bytes()).hexdigest() != saved["cohort_audit_sha256"]:
            raise ValueError("Archived forecast does not match the retained peer audit")

    explorer = folder / "figures/neural-network-explorer.html"
    explorer_bytes = explorer.read_bytes()
    match = re.search(r'<script id="model-data" type="application/json">(.*?)</script>',
                      explorer_bytes.decode("utf-8"), re.S)
    if match is None:
        raise ValueError("Archived explorer is missing its fitted models")
    fitted = json.loads(match[1])
    if (fitted["input_columns"] != list(input_columns) or fitted["proposal"] != saved["point"]
            or fitted["selected_architecture"] != saved["selected_neural_architecture"]
            or fitted["scope"] != saved["training_scope"] or fitted["target_label"] != saved["target_units"]):
        raise ValueError("Archived explorer and review describe different fits")
    reference_mean, reference_sd = gp_prediction(fitted["gp"], saved["point"])
    if not np.allclose([reference_mean[0], reference_sd[0]],
                       [saved["proposal"]["predicted_output"], saved["proposal"]["posterior_std"]],
                       rtol=1e-10, atol=1e-9):
        raise ValueError("Archived GP fails parity at its saved proposal")

    gp_mean, gp_sd = gp_prediction(fitted["gp"], point)
    seeds = [network_prediction(network, point)[0]
             for network in fitted["networks"][fitted["selected_architecture"]]]
    return dict(
        gp_model=saved["query_model"], data_scope=saved["training_scope"],
        prediction_units=saved["target_units"], height_scale=saved["height_scale"],
        gp_mean=float(gp_mean[0]), gp_latent_sd=float(gp_sd[0]),
        neural_architecture=fitted["selected_architecture"], neural_mean=float(np.mean(seeds)),
        neural_seed_sd=float(np.std(seeds, ddof=1)),
        official_snapshot_sha256=snapshot_hash,
        explorer_sha256=hashlib.sha256(explorer_bytes).hexdigest(),
    )


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
    peer = np.asarray(review.get("peer_mask", np.zeros(len(values), dtype=bool)), dtype=bool)
    incumbent = values == values.max()
    highest = np.argsort(values, kind="stable")[-5:]
    figure, axes = plt.subplots(2, 1, figsize=(12, 6.2))

    for axis, positions, title in (
        (axes[0], np.arange(len(values)), f"All {len(values)} measured outputs"),
        (axes[1], highest, f"Close-up: highest {len(highest)} of {len(values)} outputs (subset)"),
    ):
        for mask, colour, marker, size, label in (
            (rounds == 0, "#4678a0", "o", 42, "Initial supplied observations"),
            ((rounds > 0) & ~peer, "#bd7720", "D", 58, "Returned queries"),
            (peer, "#8055a0", "s", 55, "Retained peer returns"),
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
        if peer[position]:
            label = str(data.iloc[position]["observation_id"]).replace("-round-", "\nround-")
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
                 color="#bd7720", zorder=4, label="Returned query output")
    if observed.any():
        last = np.flatnonzero(observed)[-1]
        axis.scatter(rounds[last], returned[last], s=190, marker="*", color="#c6403b",
                     edgecolors="#263238", linewidths=0.5, zorder=5, label="Latest return")
    for position, value in enumerate(returned):
        if np.isfinite(value):
            offset = -24 if value < best[position] else 13
            axis.annotate(f"{value:.6f}", (rounds[position], value),
                          xytext=(0, offset), textcoords="offset points", ha="center", fontsize=9)
    axis.annotate(f"{best[0]:.6f}", (rounds[0], best[0]),
                  xytext=(10, -13), textcoords="offset points", va="top", fontsize=9)
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

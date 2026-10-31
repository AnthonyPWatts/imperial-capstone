"""Test Function 1's activity hypothesis without changing its signed-score decision."""

import numpy as np
import pandas as pd

from function_1_model_helpers import signed_log
from model_helpers import make_models
from model_review_helpers import make_regression_models, score_review_candidates
from prediction_review_helpers import (
    choose_training_scope, make_candidate_review, paired_peer_review,
)
from surrogate_study_helpers import evaluate_regressors


TARGET_LABELS = {
    "signed": "signed-log score h",
    "magnitude": "log-magnitude activity g",
    "raw_magnitude": "absolute raw output |y|",
}


def activity_targets(outputs, *, scale):
    """Keep signed, log-magnitude and literal absolute targets separate."""
    outputs = np.asarray(outputs, float)
    if outputs.ndim != 1 or not np.isfinite(outputs).all():
        raise ValueError("Require a finite output vector")
    heights = signed_log(outputs, scale=scale)
    return dict(signed=heights, magnitude=np.abs(heights), raw_magnitude=np.abs(outputs))


def proposal_geometry(point, reference, *, centre=(.5, .5)):
    """Describe a decision's movement; the geometric centre is not a source estimate."""
    point, reference, centre = (np.asarray(x, float) for x in (point, reference, centre))
    if (point.shape != (2,) or reference.shape != (2,) or centre.shape != (2,)
            or not np.isfinite([point, reference, centre]).all()):
        raise ValueError("Supply finite two-input points and centre")
    distance = float(np.linalg.norm(point-centre))
    reference_distance = float(np.linalg.norm(reference-centre))
    return dict(distance_to_centre=distance, reference_distance_to_centre=reference_distance,
                change_in_centre_distance=distance-reference_distance,
                percent_closer_to_centre=(100*(1-distance/reference_distance) if reference_distance else None),
                movement_from_reference=float(np.linalg.norm(point-reference)))


def fit_activity_targets(official, pooled, *, input_columns, scale, fixed_settings,
                         adaptive_settings, gp_names, selection_metric="LOO MAE", verbose=True):
    """Fit three declared targets and compare training scopes on identical official rows."""
    inputs = official[input_columns].to_numpy()
    peer_mask = pooled.observation_id.str.startswith("peer-")
    peers = pooled.loc[peer_mask]
    aligned = pooled.loc[~peer_mask].reset_index(drop=True)
    pd.testing.assert_frame_equal(aligned, official.reset_index(drop=True), check_exact=True)
    peer_inputs = peers[input_columns].to_numpy()
    targets = activity_targets(official.y.to_numpy(), scale=scale)
    peer_targets = activity_targets(peers.y.to_numpy(), scale=scale)
    models = make_models(**fixed_settings)
    adaptive = make_regression_models(len(input_columns), **adaptive_settings)
    models.update({name: adaptive[name] for name in ("Gaussian process", "GP: separate length scales")})
    models = {name: models[name] for name in ("Mean baseline", *gp_names)}
    studies = {}
    for name, target in targets.items():
        if verbose:
            print(f"Target: {TARGET_LABELS[name]}", flush=True)
        official_run = evaluate_regressors(models, inputs, target, verbose=verbose)
        peer_run = paired_peer_review({key: models[key] for key in gp_names}, inputs, peer_inputs,
            target, peer_targets[name], official_run, verbose=verbose)
        selected = choose_training_scope(official_run, gp_names, peer_run=peer_run, metric=selection_metric)
        studies[name] = dict(target=target, official=official_run, peers=peer_run, selected=selected)
    return dict(studies=studies, models=models, inputs=inputs, peer_inputs=peer_inputs,
                targets=targets, peer_targets=peer_targets)


def common_activity_pools(official_inputs, known_inputs, raw_outputs, *, input_columns,
                          search_settings, prepared_point, centre=(.5, .5), activity_anchor=None):
    """Freeze one baseline pool and one symmetric expansion before comparing targets."""
    official_inputs, known_inputs = np.asarray(official_inputs), np.asarray(known_inputs)
    raw_outputs = np.asarray(raw_outputs)
    signed_anchor = official_inputs[int(np.argmax(raw_outputs))]
    baseline, _ = make_candidate_review(official_inputs, signed_anchor, input_columns=input_columns,
        search_settings=search_settings, excluded_inputs=known_inputs)
    # Give both policies access to a neighbourhood around the strongest activity as well.
    magnitude_anchor = (official_inputs[int(np.argmax(np.abs(raw_outputs)))] if activity_anchor is None
                        else np.asarray(activity_anchor, float))
    activity_pool, _ = make_candidate_review(official_inputs, magnitude_anchor, input_columns=input_columns,
        search_settings=search_settings, excluded_inputs=known_inputs)
    extras = pd.DataFrame([centre, prepared_point], columns=input_columns)
    extras["candidate_source"] = ["Geometric centre", "Prepared signed proposal"]
    expanded = pd.concat([baseline, activity_pool, extras], ignore_index=True)
    expanded[input_columns] = expanded[input_columns].round(6)
    expanded = expanded.drop_duplicates(subset=input_columns, keep="first")
    known = set(map(tuple, known_inputs.round(6)))
    expanded = expanded.loc[[tuple(row) not in known for row in expanded[input_columns].to_numpy()]].copy()
    return dict(baseline=baseline.reset_index(drop=True), symmetric=expanded.reset_index(drop=True))


def compare_activity_policies(study, pools, *, input_columns, reference, gp_names,
                              weights=(0., 1., 1.96, 3.), main_weight=1.96, centre=(.5, .5)):
    """Use identical candidates for each target and record model/weight sensitivity."""
    rows, proposals, scores = [], {}, {}
    for pool_name, pool in pools.items():
        proposals[pool_name], scores[pool_name] = {}, {}
        for target_name, run in study["studies"].items():
            selected = run["selected"]
            scored = score_review_candidates(selected["model"], pool, input_columns=input_columns, weight=main_weight)
            top = scored.iloc[0]
            point = top[input_columns].to_numpy(dtype=float)
            proposals[pool_name][target_name] = dict(
                target=target_name, units=TARGET_LABELS[target_name], pool=pool_name,
                model=selected["name"], training_scope=selected["scope"],
                point=point.tolist(), portal_input="-".join(f"{x:.6f}" for x in point),
                predicted_target=float(top.posterior_mean), latent_sd=float(top.posterior_sd),
                ucb=float(top.ucb), **proposal_geometry(point, reference, centre=centre))
            scores[pool_name][target_name] = scored
            for weight in weights:
                best = scored.loc[(scored.posterior_mean+weight*scored.posterior_sd).idxmax()]
                p = best[input_columns].to_numpy(dtype=float)
                rows.append(dict(pool=pool_name, target=target_name, model=selected["name"],
                                 scope=selected["scope"], weight=weight, x1=p[0], x2=p[1],
                                 **proposal_geometry(p, reference, centre=centre)))
    # Hold model family, scope, pool and weight fixed, changing only the fitted target.
    controlled = []
    pool = pools["symmetric"]
    for scope in ("official", "peers"):
        for model_name in gp_names:
            for target_name, run in study["studies"].items():
                scored = score_review_candidates(run[scope]["fitted"][model_name], pool,
                                                  input_columns=input_columns, weight=main_weight)
                top = scored.iloc[0]
                point = top[input_columns].to_numpy(dtype=float)
                controlled.append(dict(scope=scope, model=model_name, target=target_name,
                    x1=point[0], x2=point[1], **proposal_geometry(point, reference, centre=centre)))
    return dict(proposals=proposals, scores=scores, weight_sensitivity=pd.DataFrame(rows),
                same_family_comparison=pd.DataFrame(controlled))


def opposite_sign_pairs(data, *, input_columns, maximum=6):
    """Show measured proximity of opposite signs without inferring a source or hidden peak."""
    positive, negative = data.loc[data.y > 0], data.loc[data.y < 0]
    rows = []
    for _, pos in positive.iterrows():
        for _, neg in negative.iterrows():
            rows.append(dict(positive_id=pos.observation_id, negative_id=neg.observation_id,
                positive_output=float(pos.y), negative_output=float(neg.y),
                input_distance=float(np.linalg.norm(pos[input_columns].to_numpy(float)-neg[input_columns].to_numpy(float)))))
    return pd.DataFrame(rows).sort_values("input_distance").head(maximum).reset_index(drop=True)


def plot_activity_evidence(data, *, scale, input_columns, save_to=None):
    """Show measured sign and response strength in the same input square."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from model_helpers import finish_figure

    target = activity_targets(data.y.to_numpy(), scale=scale)
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True, layout="constrained")
    peer = data.observation_id.str.startswith("peer-").to_numpy()
    for ax, name in zip(axes, ("signed", "magnitude")):
        values = target[name]
        normal = Normalize(-abs(values).max(), abs(values).max()) if name == "signed" else Normalize(0, values.max())
        for mask, marker in ((~peer, "o"), (peer, "s")):
            image = ax.scatter(data.loc[mask, input_columns[0]], data.loc[mask, input_columns[1]],
                c=values[mask], norm=normal, cmap="coolwarm" if name == "signed" else "viridis",
                s=40+80*target["magnitude"][mask]/target["magnitude"].max(), marker=marker,
                edgecolors="#263238", linewidths=.7, zorder=3)
        figure.colorbar(image, ax=ax, shrink=.8, label=TARGET_LABELS[name])
        ax.scatter(.5, .5, color="#b76b00", marker="+", s=200, linewidths=2, zorder=4)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=input_columns[0], ylabel=input_columns[1],
               title="Measured " + TARGET_LABELS[name])
        ax.set_aspect("equal");ax.grid(alpha=.15)
    legend = [Line2D([], [], marker="o", color="none", markerfacecolor="grey", label="Official observations"),
              Line2D([], [], marker="s", color="none", markerfacecolor="grey", label="Retained peer reports"),
              Line2D([], [], marker="+", color="#b76b00", linestyle="none", label="Geometric centre (not a known source)")]
    figure.legend(handles=legend, loc="outside lower center", ncol=3, fontsize=9)
    figure.suptitle("Function 1: sign and activity answer different questions\nEvery point is measured; colour uses the stated transform, without rounding tiny outputs to zero.")
    return finish_figure(figure, save_to)


def plot_activity_decisions(study, policies, official, pooled, *, input_columns, reference,
                            pool_name="baseline", weight=1.96, side=81, save_to=None):
    """Compare refitted target means and UCB; each target has its own labelled scale."""
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull
    from model_helpers import finish_figure

    grid = np.linspace(0, .999999, side)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True, layout="constrained")
    for column, (name, run) in enumerate(study["studies"].items()):
        mean, sd = run["selected"]["model"].predict(points, return_std=True)
        fields = [mean.reshape(xx.shape), (mean+weight*sd).reshape(xx.shape)]
        low, high = min(field.min() for field in fields), max(field.max() for field in fields)
        if high <= low:
            high = low+max(abs(low)*1e-6, 1e-12)
        train = pooled if run["selected"]["scope"] != "official" else official
        inputs = train[input_columns].to_numpy()
        hull = inputs[ConvexHull(inputs).vertices]
        hull = np.vstack([hull, hull[0]])
        proposal = policies["proposals"][pool_name][name]
        point = proposal["point"]
        for row, (field, label) in enumerate(zip(fields, ("Mean", f"UCB: mean + {weight:g} × SD"))):
            ax = axes[row, column]
            image = ax.contourf(xx, yy, field, levels=np.linspace(low, high, 18), cmap="viridis")
            figure.colorbar(image, ax=ax, shrink=.75)
            ax.plot(hull[:, 0], hull[:, 1], color="white", linestyle=":", linewidth=1.5)
            ax.scatter(*inputs.T, color="black", s=14)
            ax.scatter(*reference, marker="X", color="#69c9ff", edgecolor="black", s=120, zorder=5)
            ax.scatter(*point, marker="*", color="gold", edgecolor="black", s=220, zorder=6)
            ax.scatter(.5, .5, marker="+", color="white", s=120, linewidths=2)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=input_columns[0], ylabel=input_columns[1],
                   title=f"{TARGET_LABELS[name]}\n{label}")
            ax.set_aspect("equal")
    figure.suptitle("Changing the fitted target changes the proposed location\nGold: that target's query. Blue X: prepared signed query. White +: geometric centre. Dotted: training-input hull.")
    figure.supxlabel("Colour scales are shared vertically within a target only. Predictions, SD and UCB are in that target's units; these are not measured surfaces.", fontsize=10)
    return finish_figure(figure, save_to)


def plot_activity_query_shift(data, policies, *, input_columns, reference, pool_name="baseline", save_to=None):
    """Make input movement and distance to the geometric centre directly comparable."""
    import matplotlib.pyplot as plt
    from model_helpers import finish_figure

    colours = {"signed": "#3279a8", "magnitude": "#bd7720", "raw_magnitude": "#8055a0"}
    labels = {"signed": "Signed-score policy", "magnitude": "Log-magnitude policy", "raw_magnitude": "Raw-magnitude policy"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.9), layout="constrained")
    axes[0].scatter(*data[input_columns].to_numpy().T, color="#677e8a", s=30, alpha=.7, label="Measured inputs")
    distances = []
    for name, row in policies["proposals"][pool_name].items():
        point = np.asarray(row["point"])
        axes[0].scatter(*point, color=colours[name], marker="X" if name == "signed" else "*", s=180,
                        edgecolor="black", zorder=5, label=labels[name])
        if name != "signed":
            axes[0].annotate("", xy=point, xytext=reference,
                             arrowprops=dict(arrowstyle="->", color=colours[name], lw=2))
        distances.append(row["distance_to_centre"])
    axes[0].scatter(.5, .5, marker="+", color="black", s=180, label="Geometric centre")
    axes[0].set(xlim=(0,1), ylim=(0,1), xlabel=input_columns[0], ylabel=input_columns[1],
                title="Same data and candidates; different fitted targets")
    axes[0].set_aspect("equal");axes[0].legend(loc="lower right", fontsize=8)
    names = list(policies["proposals"][pool_name])
    bars = axes[1].barh([labels[name] for name in names], distances, color=[colours[name] for name in names])
    axes[1].bar_label(bars, fmt="%.4f", padding=6)
    axes[1].set(xlim=(0, max(distances)*1.22), xlabel="Euclidean input distance to (0.5, 0.5)",
                title="How far is the proposed query from the centre?")
    axes[1].invert_yaxis();axes[1].grid(axis="x", alpha=.2);axes[1].set_axisbelow(True)
    figure.suptitle("Function 1: quantify the pull towards the centre\nDistance describes the query location, not distance to an established contamination source.")
    return finish_figure(figure, save_to)

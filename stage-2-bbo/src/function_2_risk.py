"""Conditional decision risk and a bounded 13-query experiment for Function 2."""

import json
from pathlib import Path

from IPython.display import Markdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from coverage_optimisation import MeanCoverage2D
from function_2_selection import fit_explanation
from initial_exploration import _coordinate_axes, _finish


MODEL_SPECS = (("x1-only", 1, False), ("Local 2D", 2, False), ("Flexible 2D", 2, True))
REFERENCES = {
    "Pure coverage": (0.162244, 0.811282),
    "Coverage/hypothesis blend": (0.144906, 0.811010),
    "Hypothesis discrimination": (0.686100, 0.721359),
    "Right-hand gap": (0.8, 0.4),
    "x1 = 0.7 gap": (0.7, 0.4),
}


def knowledge_gradient(means, update_slopes):
    """Exactly integrate max_i(a_i + b_i Z), Z ~ N(0, 1), on a finite set.

    Construct the upper envelope of lines. This avoids sampling prospective
    outcomes or repeatedly fitting models. Parameters remain fixed.
    """
    means, slopes = np.asarray(means), np.asarray(update_slopes)
    order = np.lexsort((means, slopes))
    # For identical slopes, only the greatest intercept can reach the envelope.
    order = order[np.r_[slopes[order][1:] != slopes[order][:-1], True]]
    hull, starts = [], []
    for index in order:
        crossing = -np.inf
        while hull:
            previous = hull[-1]
            crossing = (means[previous] - means[index]) / (slopes[index] - slopes[previous])
            if crossing > starts[-1]:
                break
            hull.pop()
            starts.pop()
        if not hull:
            crossing = -np.inf
        hull.append(index)
        starts.append(crossing)
    lower = np.asarray(starts)
    upper = np.r_[lower[1:], np.inf]
    selected = np.asarray(hull)
    density = lambda z: np.exp(-z * z / 2) / np.sqrt(2 * np.pi)
    expectation = np.sum(means[selected] * (ndtr(upper) - ndtr(lower))
                         + slopes[selected] * (density(lower) - density(upper)))
    # Round-off can leave an effectively zero gain slightly negative.
    return max(0.0, float(expectation - means.max()))


def condition_on_observation(means, covariance, index, observation, noise_variance):
    """Return a Gaussian posterior update in original output units."""
    column = covariance[:, index].copy()
    observation_variance = covariance[index, index] + noise_variance
    updated_mean = means + column / observation_variance * (observation - means[index])
    updated_covariance = covariance - np.outer(column, column) / observation_variance
    return updated_mean, updated_covariance


def posterior_for_pool(inputs, outputs, pool, spec, noise_sd):
    name, dimensions, separate = spec
    fit = fit_explanation(inputs, outputs, dimensions, noise_sd, separate_lengths=separate)
    unique, inverse = np.unique(pool[:, :dimensions], axis=0, return_inverse=True)
    means, covariance = fit["model"].predict(unique, return_cov=True)
    return {"name": name, "fit": fit, "means": means, "covariance": covariance,
            "pool_indices": inverse, "noise_sd": noise_sd}


def score_queries(posterior, query_count):
    mean, covariance = posterior["means"], posterior["covariance"]
    indices = posterior["pool_indices"][:query_count]
    # x1-only queries at different x2 values are the same statistical experiment.
    values = {int(index): knowledge_gradient(
        mean, covariance[:, index] / np.sqrt(covariance[index, index] + posterior["noise_sd"] ** 2))
        for index in np.unique(indices)}
    return np.array([values[int(index)] for index in indices])


def _posterior_factor(covariance):
    jitter = max(float(np.diag(covariance).max()), 1.0) * 1e-12
    return np.linalg.cholesky(covariance + np.eye(len(covariance)) * jitter)


def estimate_current_risk(posterior, *, draws=4096, seed=17):
    mean = posterior["means"]
    factor = _posterior_factor(posterior["covariance"])
    rng = np.random.default_rng(seed)
    regrets = []
    decision = int(np.argmax(mean))
    for start in range(0, draws, 256):
        count = min(256, draws - start)
        functions = mean[:, None] + factor @ rng.standard_normal((len(mean), count))
        regrets.extend(functions.max(axis=0) - functions[decision])
    regrets = np.asarray(regrets)
    return {"risk": float(regrets.mean()), "mc_se": float(regrets.std(ddof=1) / np.sqrt(draws)),
            "draws": draws}


def simulate_remaining_queries(posterior, first_indices, query_count, *, trials=128, seed=31):
    """Compare first choices with the same twelve-query continuation policy.

    Posterior paths and observation-noise draws are shared across first choices,
    making their differences paired. These are simulated values, never portal
    responses. A final recommendation can be anywhere in the finite pool.
    """
    base_mean, base_covariance = posterior["means"], posterior["covariance"]
    factor = _posterior_factor(base_covariance)
    rng = np.random.default_rng(seed)
    truths = (base_mean[:, None] + factor @ rng.standard_normal((len(base_mean), trials))).T
    noises = rng.normal(0, posterior["noise_sd"], (trials, 13))
    query_indices = np.unique(posterior["pool_indices"][:query_count])
    first = posterior["pool_indices"][first_indices]
    losses = np.empty((len(first), trials, 3))
    for candidate, first_index in enumerate(first):
        for trial in range(trials):
            mean, covariance = base_mean.copy(), base_covariance.copy()
            truth = truths[trial]
            best_true = truth.max()
            for step in range(13):
                if step == 0:
                    index = int(first_index)
                else:
                    # The remaining twelve queries taper from 2 SD of optimism to 0.
                    coefficient = 2 * (12 - step) / 11
                    values = mean[query_indices] + coefficient * np.sqrt(
                        np.maximum(np.diag(covariance)[query_indices], 0))
                    index = int(query_indices[np.argmax(values)])
                observation = truth[index] + noises[trial, step]
                mean, covariance = condition_on_observation(
                    mean, covariance, index, observation, posterior["noise_sd"] ** 2)
                if step in (0, 3, 12):
                    stage = {0: 0, 3: 1, 12: 2}[step]
                    losses[candidate, trial, stage] = best_true - truth[np.argmax(mean)]
    return {"losses": losses, "mean": losses.mean(axis=1),
            "mc_se": losses.std(axis=1, ddof=1) / np.sqrt(trials), "trials": trials,
            "stages": [1, 4, 13]}


def analyse_function_2_risk(data, *, output=None, resolution=25, trials=128):
    if [name for name in data if name.startswith("x")] != ["x1", "x2"]:
        raise ValueError("This finite-set decision study is for two inputs only.")
    inputs, outputs = data[["x1", "x2"]].to_numpy(), data.y.to_numpy()
    axis = np.round(np.linspace(0, .999999, resolution), 6)
    first, second = np.meshgrid(axis, axis)
    grid = np.c_[first.ravel(), second.ravel()]
    queries = np.vstack([grid, np.array(list(REFERENCES.values()))])
    pool = np.vstack([queries, inputs])
    geometry = MeanCoverage2D(inputs)
    models, names, candidate_indices = [], list(REFERENCES), list(range(len(grid), len(queries)))
    with threadpool_limits(limits=2):
        for spec in MODEL_SPECS:
            posterior = posterior_for_pool(inputs, outputs, pool, spec, .05)
            gains = score_queries(posterior, len(queries))
            tied = np.flatnonzero(gains >= gains.max() - 1e-12)
            # Use coverage only to break statistical indifference, e.g. x2 under H1.
            best = min(tied, key=lambda index: geometry(queries[index]))
            names.append(f"KG: {spec[0]}")
            candidate_indices.append(int(best))
            models.append({"posterior": posterior, "gains": gains, "best_index": int(best),
                           "current": estimate_current_risk(posterior)})
        for model in models:
            model["rollout"] = simulate_remaining_queries(
                model["posterior"], candidate_indices, len(queries), trials=trials)
        noise_checks = []
        for noise in (.025, .15):
            for spec in MODEL_SPECS:
                posterior = posterior_for_pool(inputs, outputs, pool, spec, noise)
                gains = score_queries(posterior, len(queries))
                tied = np.flatnonzero(gains >= gains.max() - 1e-12)
                best = min(tied, key=lambda index: geometry(queries[index]))
                noise_checks.append({"model": spec[0], "noise_sd": noise,
                    "best_point": queries[best], "max_gain": float(gains[best]),
                    "candidate_gains": gains[candidate_indices]})
        fine_axis = np.round(np.linspace(0, .999999, 41), 6)
        fine_first, fine_second = np.meshgrid(fine_axis, fine_axis)
        fine_queries = np.vstack([np.c_[fine_first.ravel(), fine_second.ravel()],
                                  np.array(list(REFERENCES.values()))])
        fine_pool = np.vstack([fine_queries, inputs])
        grid_checks = []
        for spec in MODEL_SPECS:
            posterior = posterior_for_pool(inputs, outputs, fine_pool, spec, .05)
            gains = score_queries(posterior, len(fine_queries))
            tied = np.flatnonzero(gains >= gains.max() - 1e-12)
            best = min(tied, key=lambda index: geometry(fine_queries[index]))
            grid_checks.append({"model": spec[0], "point": fine_queries[best],
                                "gain": float(gains[best])})
    worst_current = max(model["current"]["risk"] for model in models)
    worst_after = np.max([model["current"]["risk"] - model["gains"] for model in models], axis=0)
    robust_index = int(np.argmin(worst_after))
    study = {"axis": axis, "grid": grid, "queries": queries, "pool": pool,
             "models": models, "candidate_names": names, "candidate_indices": candidate_indices,
             "noise_checks": noise_checks, "grid_checks": grid_checks,
             "worst_current_risk": worst_current,
             "robust_index": robust_index, "worst_after_risk": worst_after}
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        # Store reproducible results, not large covariance matrices or estimator objects.
        saved = {key: value for key, value in study.items() if key not in ("models", "pool")}
        saved["models"] = [{key: value for key, value in model.items() if key != "posterior"}
                           | {"name": model["posterior"]["name"],
                              "kernel": str(model["posterior"]["fit"]["model"].kernel_),
                              "warnings": model["posterior"]["fit"]["warnings"]}
                           for model in models]
        def encode(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.integer):
                return int(value)
            raise TypeError(type(value).__name__)
        (destination / "risk-study.json").write_text(
            json.dumps(saved, default=encode, indent=2, allow_nan=False), encoding="utf-8")
    return study


def current_risk_table(study):
    rows = [{"Model": m["posterior"]["name"], "Expected shortfall": m["current"]["risk"],
             "Simulation standard error": m["current"]["mc_se"]} for m in study["models"]]
    return pd.DataFrame(rows).style.format({"Expected shortfall": "{:.4f}",
        "Simulation standard error": "{:.4f}"}).hide(axis="index")


def query_value_table(study):
    rows = []
    for name, index in zip(study["candidate_names"], study["candidate_indices"]):
        row = {"First query": name, "x1": study["queries"][index, 0], "x2": study["queries"][index, 1]}
        row.update({model["posterior"]["name"]: model["gains"][index] for model in study["models"]})
        rows.append(row)
    return pd.DataFrame(rows).style.format({"x1": "{:.3f}", "x2": "{:.3f}",
        **{m["posterior"]["name"]: "{:.4f}" for m in study["models"]}}).hide(axis="index")


def final_risk_table(study):
    rows = []
    for candidate, name in enumerate(study["candidate_names"]):
        row = {"First query": name}
        for model in study["models"]:
            mean = model["rollout"]["mean"][candidate, -1]
            error = model["rollout"]["mc_se"][candidate, -1]
            row[model["posterior"]["name"]] = f"{mean:.4f} ± {error:.4f}"
        rows.append(row)
    return pd.DataFrame(rows).style.hide(axis="index")


def risk_findings(study):
    index = study["robust_index"]
    point = study["queries"][index]
    fine = next(item for item in study["grid_checks"] if item["model"] == "Local 2D")
    current, after = study["worst_current_risk"], study["worst_after_risk"][index]
    return Markdown(
        "The risk estimates disagree because the models allow different unseen "
        "behaviour. The local 2D model is the most uncertain about the location of "
        "the optimum and determines the largest residual risk across these models.\n\n"
        "If we minimise that **largest model-specific expected shortfall after one "
        f"query**, the best tested query is **({point[0]:.3f}, {point[1]:.3f})**. "
        f"The largest risk falls from **{current:.4f}** to **{after:.4f}** output units, "
        f"an expected reduction of **{current-after:.4f}**. A finer 41 × 41 grid "
        f"moves the local-model choice to **({fine['point'][0]:.3f}, {fine['point'][1]:.3f})**; "
        "the supported conclusion is an upper-right region, not a six-decimal optimum.\n\n"
        "This conservative rule avoids assigning probabilities to the models. "
        "It still depends on which models are included. The earlier prediction "
        "check favoured the x1-only model over the local 2D model; treating the "
        "latter as the risk to guard against is a judgement, not evidence that it "
        "is the true function.\n\n"
        "Under the x1-only model, x2 does not affect query value. Coverage breaks "
        "such ties when reporting a point."
    )


def rollout_findings(study):
    local = next(model for model in study["models"] if model["posterior"]["name"] == "Local 2D")
    reference = study["candidate_names"].index("KG: Local 2D")
    paired = local["rollout"]["losses"][0, :, -1] - local["rollout"]["losses"][reference, :, -1]
    difference = float(paired.mean())
    error = float(paired.std(ddof=1) / np.sqrt(len(paired)))
    return Markdown(
        "The remaining queries substantially reduce risk for every starting choice. "
        "In these simulations the upper-right first query has the lowest mean "
        "final risk under the local 2D model. Its advantage over starting with pure "
        f"coverage is only **{difference:.4f} ± {error:.4f}** output units "
        "(paired simulation standard error). That comparison does not establish "
        "a clear thirteen-query winner.\n\n"
        "The one-query calculation supports the upper-right region more strongly "
        "than the full-budget experiment separates the starting choices. A one-step "
        "knowledge-gradient choice is not a proven optimal first move for thirteen "
        "queries. The continuation policy, possible functions and noise assumptions "
        "can matter more than small differences between proposed first points."
    )


def risk_noise_table(study):
    rows = [{"Model": item["model"], "Noise SD": item["noise_sd"],
             "x1": item["best_point"][0], "x2": item["best_point"][1],
             "Best one-query risk reduction": item["max_gain"]} for item in study["noise_checks"]]
    return pd.DataFrame(rows).style.format({"Noise SD": "{:.3f}", "x1": "{:.3f}",
        "x2": "{:.3f}", "Best one-query risk reduction": "{:.4f}"}).hide(axis="index")


def plot_risk_reduction(data, study, *, save_to=None):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained")
    maximum = max(model["gains"].max() for model in study["models"])
    n = len(study["axis"])
    for axis, model in zip(axes, study["models"]):
        heatmap = axis.pcolormesh(study["axis"], study["axis"], model["gains"][:n*n].reshape(n, n),
            cmap="viridis", vmin=0, vmax=maximum, shading="nearest", rasterized=True)
        axis.scatter(data.x1, data.x2, s=80, facecolors="white", edgecolors="#263238", linewidths=.7)
        for row in data.itertuples():
            axis.text(row.x1, row.x2, str(row.rank), ha="center", va="center", fontsize=7)
        point = study["queries"][model["best_index"]]
        axis.scatter(*point, s=180, marker="*", color="#ffbf3f", edgecolors="#172126",
                     zorder=5, clip_on=False)
        axis.set_title(f"{model['posterior']['name']}\nBest grid query: ({point[0]:.3f}, {point[1]:.3f})")
        _coordinate_axes(axis, ("x1", "x2"))
    figure.colorbar(heatmap, ax=axes, shrink=.8, label="Expected shortfall reduction (output units)")
    return _finish(figure, save_to)


def plot_risk_over_rounds(study, *, save_to=None):
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), layout="constrained", sharey=True)
    candidates = [0, 2, 6]
    colours = ["#287a69", "#bd531d", "#236b91"]
    for axis, model in zip(axes, study["models"]):
        for index, colour in zip(candidates, colours):
            rollout = model["rollout"]
            axis.errorbar(rollout["stages"], rollout["mean"][index],
                          yerr=rollout["mc_se"][index], marker="o", capsize=3,
                          label=study["candidate_names"][index], color=colour)
        axis.set(title=model["posterior"]["name"], xlabel="Additional queries completed",
                 xticks=[1, 4, 13])
        axis.grid(axis="y", alpha=.2)
    upper = max(np.max(model["rollout"]["mean"][candidates]
                       + model["rollout"]["mc_se"][candidates]) for model in study["models"])
    axes[0].set_ylim(0, 1.08 * upper)
    axes[0].set_ylabel("Simulated expected shortfall")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    return _finish(figure, save_to)

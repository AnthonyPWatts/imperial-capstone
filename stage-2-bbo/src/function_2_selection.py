"""Compare coverage with a specific pair of explanations for Function 2."""

from itertools import product
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from IPython.display import Markdown

from coverage_optimisation import MeanCoverage2D, optimise_average_coverage
from initial_exploration import _coordinate_axes, _finish


WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
NOISE_LEVELS = (0.025, 0.05, 0.1, 0.15)
BOUNDS = [(0.0, 0.999999)] * 2


def fit_explanation(inputs, outputs, dimensions, noise_sd, *, separate_lengths=False):
    """Noise is specified in output units, before internal standardisation."""
    scale = float(np.std(outputs))
    if scale == 0 or noise_sd <= 0:
        raise ValueError("These exploratory fits require varying outputs and positive noise.")
    length = [0.3, 0.3] if separate_lengths else 0.3
    model = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (0.05, 10.0)) * Matern(length, (0.03, 2.0), nu=1.5),
        normalize_y=True, alpha=(noise_sd / scale) ** 2,
        n_restarts_optimizer=4, random_state=7,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(np.asarray(inputs)[:, :dimensions], outputs)
    return {
        "model": model, "dimensions": dimensions, "noise_sd": noise_sd,
        "warnings": [str(item.message) for item in caught],
    }


def predict_explanation(fit, points):
    """Return means and variances for a new noisy observation, in output units."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    mean, latent_sd = fit["model"].predict(points[:, :fit["dimensions"]], return_std=True)
    return mean, latent_sd ** 2 + fit["noise_sd"] ** 2


def standardised_disagreement(means, variances):
    """Squared mean separation relative to combined predictive spread.

    This is an acquisition heuristic, not a test statistic with a reference
    distribution: the fits share data and are not independent estimates.
    """
    means, variances = np.asarray(means), np.asarray(variances)
    return (means[0] - means[1]) ** 2 / (variances[0] + variances[1])


def mix_scores(coverage, disagreement, coverage_max, weight):
    if not 0 <= weight <= 1:
        raise ValueError("Coverage weight must lie between zero and one.")
    coverage_score = np.asarray(coverage) / coverage_max if coverage_max > 0 else 0.0
    # A fixed mapping preserves weak absolute disagreement across model choices.
    hypothesis_score = np.asarray(disagreement) / (1 + np.asarray(disagreement))
    return weight * coverage_score + (1 - weight) * hypothesis_score


def _disagreement(fits, points):
    predictions = [predict_explanation(fit, points) for fit in fits]
    return standardised_disagreement([p[0] for p in predictions], [p[1] for p in predictions])


def _refine_maximum(score, axis, grid_scores):
    """Refine every local maximum on the plotting grid, including boundary ones."""
    surface = np.asarray(grid_scores).reshape(len(axis), len(axis))
    local = np.argwhere(surface == maximum_filter(surface, size=3, mode="nearest"))
    starts = [np.array([axis[column], axis[row]]) for row, column in local]
    results = []
    for start in starts:
        result = minimize(lambda point: -float(score(point)), start,
                          method="Nelder-Mead", bounds=BOUNDS,
                          options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 500})
        if not result.success:
            raise RuntimeError(f"Selection refinement failed: {result.message}")
        results.append(result)
    best = min(results, key=lambda result: result.fun)
    lower = np.clip(np.floor(best.x * 1e6) / 1e6, 0, 0.999999)
    upper = np.clip(np.ceil(best.x * 1e6) / 1e6, 0, 0.999999)
    point = np.asarray(max(product(*zip(lower, upper)), key=score))
    return {"point": point, "score": float(score(point)), "unrounded_score": -float(best.fun),
            "local_starts": len(starts), "grid_max": float(surface.max())}


def _comparison(inputs, outputs, coverage, coverage_max, axis, grid, gains, noise_sd,
                *, separate_lengths=False):
    fits = [fit_explanation(inputs, outputs, dimensions, noise_sd,
                            separate_lengths=separate_lengths and dimensions == 2)
            for dimensions in (1, 2)]
    disagreement = _disagreement(fits, grid)
    hypothesis_max = _refine_maximum(
        lambda point: _disagreement(fits, point)[0], axis, disagreement)
    # Keep the raw peak to compare the strength of disagreement across assumptions.
    disagreement_max = hypothesis_max["unrounded_score"]
    selections = []
    for weight in WEIGHTS:
        scores = mix_scores(gains, disagreement, coverage_max, weight)

        def score(point):
            gain = 1 - coverage(point) / coverage.mean_before
            separation = _disagreement(fits, point)[0]
            return mix_scores(gain, separation, coverage_max, weight)

        selected = _refine_maximum(score, axis, scores)
        selected.update({"weight": weight,
                         "coverage_gain": 1 - coverage(selected["point"]) / coverage.mean_before,
                         "disagreement": float(_disagreement(fits, selected["point"])[0])})
        selected["predictions"] = [
            {"mean": float(mean[0]), "observation_sd": float(np.sqrt(variance[0]))}
            for mean, variance in [predict_explanation(fit, selected["point"]) for fit in fits]
        ]
        selections.append(selected)
    return {"noise_sd": noise_sd, "fits": fits, "disagreement": disagreement,
            "disagreement_max": disagreement_max, "selections": selections}


def analyse_function_2_selection(data, *, output=None, resolution=81):
    if [name for name in data if name.startswith("x")] != ["x1", "x2"]:
        raise ValueError("This hypothesis comparison is for two inputs only.")
    inputs, outputs = data[["x1", "x2"]].to_numpy(), data["y"].to_numpy()
    coverage = MeanCoverage2D(inputs)
    coverage_result = optimise_average_coverage(data)
    coverage_max = 1 - coverage(coverage_result["unrounded_point"]) / coverage.mean_before
    axis = np.linspace(0, 0.999999, resolution)
    first, second = np.meshgrid(axis, axis)
    grid = np.column_stack([first.ravel(), second.ravel()])
    gains = np.array([1 - coverage(point) / coverage.mean_before for point in grid])
    comparisons = [_comparison(inputs, outputs, coverage, coverage_max, axis, grid, gains, noise)
                   for noise in NOISE_LEVELS]
    main = next(item for item in comparisons if item["noise_sd"] == 0.05)
    # A less restrictive 2D fit checks how much our two-model contrast imposes.
    flexible = _comparison(inputs, outputs, coverage, coverage_max, axis, grid, gains, 0.05,
                           separate_lengths=True)
    errors = []
    for dimensions in (1, 2):
        predictions = []
        for index in range(len(outputs)):
            keep = np.arange(len(outputs)) != index
            fit = fit_explanation(inputs[keep], outputs[keep], dimensions, 0.05)
            predictions.append(predict_explanation(fit, inputs[index])[0][0])
        errors.append(float(np.sqrt(np.mean((outputs - predictions) ** 2))))
    mean_predictions = [(outputs.sum() - y) / (len(outputs) - 1) for y in outputs]
    errors.append(float(np.sqrt(np.mean((outputs - mean_predictions) ** 2))))
    examples = []
    for point in ((0.8, 0.4), (0.7, 0.4)):
        examples.append({"point": np.array(point),
                         "coverage_gain": 1 - coverage(point) / coverage.mean_before,
                         "disagreement": float(_disagreement(main["fits"], point)[0])})
    study = {"axis": axis, "grid": grid, "coverage_gains": gains,
             "coverage_max": coverage_max, "coverage_result": coverage_result,
             "main": main, "noise_comparisons": comparisons, "flexible": flexible,
             "loo_rmse": errors, "examples": examples}
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)

        def serialisable(value):
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, GaussianProcessRegressor):
                return str(value.kernel_)
            raise TypeError(type(value).__name__)

        (destination / "selection-study.json").write_text(
            json.dumps(study, default=serialisable, indent=2, allow_nan=False), encoding="utf-8")
    return study


def selection_balance_table(study):
    rows = [{"Coverage weight": row["weight"], "x1": row["point"][0], "x2": row["point"][1],
             "Coverage gain": row["coverage_gain"],
             "Hypothesis score": row["disagreement"] / (1 + row["disagreement"])}
            for row in study["main"]["selections"]]
    return pd.DataFrame(rows).style.format({"Coverage weight": "{:.0%}", "x1": "{:.6f}",
        "x2": "{:.6f}", "Coverage gain": "{:.2%}", "Hypothesis score": "{:.3f}"}).hide(axis="index")


def selection_noise_table(study):
    rows = []
    for comparison in study["noise_comparisons"]:
        selected = comparison["selections"][2]
        rows.append({"Assumed noise SD": comparison["noise_sd"], "x1": selected["point"][0],
                     "x2": selected["point"][1], "Coverage gain": selected["coverage_gain"],
                     "Maximum raw disagreement": comparison["disagreement_max"]})
    return pd.DataFrame(rows).style.format({"Assumed noise SD": "{:.3f}", "x1": "{:.6f}",
        "x2": "{:.6f}", "Coverage gain": "{:.2%}", "Maximum raw disagreement": "{:.3f}"}).hide(axis="index")


def selection_candidate_table(study):
    points = [study["main"]["selections"][2], *study["examples"]]
    rows = []
    for label, candidate in zip(("50:50 blend", "Right-hand gap", "x1 = 0.7 gap"), points):
        predictions = [predict_explanation(fit, candidate["point"])[0][0] for fit in study["main"]["fits"]]
        rows.append({"Candidate": label, "x1": candidate["point"][0], "x2": candidate["point"][1],
                     "Coverage gain": candidate["coverage_gain"],
                     "x1-only mean": predictions[0], "Local 2D mean": predictions[1]})
    return pd.DataFrame(rows).style.format({"x1": "{:.6f}", "x2": "{:.6f}",
        "Coverage gain": "{:.2%}", "x1-only mean": "{:.3f}", "Local 2D mean": "{:.3f}"}).hide(axis="index")


def selection_fit_table(study):
    return pd.DataFrame({"Predictor": ["H1: x1-only ridge", "H2: local two-input features",
                                       "Mean of the other nine outputs"],
                         "Leave-one-out RMSE": study["loo_rmse"]}).style.format(
        {"Leave-one-out RMSE": "{:.3f}"}).hide(axis="index")


def selection_findings(study):
    main = study["main"]
    selected = main["selections"][2]
    point = selected["point"]
    test = main["selections"][0]
    return Markdown(
        f"With **50% weight on each score**, the numerical selection is "
        f"**({point[0]:.6f}, {point[1]:.6f})**. It reduces mean sampling distance by "
        f"**{selected['coverage_gain']:.2%}**. The two fitted means there are "
        f"**{selected['predictions'][0]['mean']:.3f}** and "
        f"**{selected['predictions'][1]['mean']:.3f}**. This is an illustrative policy "
        "choice, not a settled submission.\n\n"
        f"Pure hypothesis discrimination selects **({test['point'][0]:.6f}, "
        f"{test['point'][1]:.6f})**, gaining only **{test['coverage_gain']:.2%}** in "
        "average coverage. It targets disagreement near the upper observations. "
        "The blend favours the upper-left because it offers broad coverage and "
        "also challenges the x1-only explanation: does the low response observed "
        "at a similar x1 extend into this unsampled region?\n\n"
        "The right-hand gaps remain useful alternatives, but they are not forced "
        "to win by the scoring rule. Predicted means below are model outputs, not "
        "new observations."
    )


def selection_sensitivity_findings(study):
    main, flexible = study["main"], study["flexible"]
    point = flexible["selections"][2]["point"]
    return Markdown(
        "A further check lets the two-input fit use a different length scale "
        "for each input. It then resembles the x1-only ridge much more closely. "
        f"The maximum raw disagreement falls from **{main['disagreement_max']:.3f}** "
        f"to **{flexible['disagreement_max']:.3f}**, and the largest hypothesis "
        f"score falls from **{main['disagreement_max']/(1+main['disagreement_max']):.3f}** "
        f"to **{flexible['disagreement_max']/(1+flexible['disagreement_max']):.3f}**. "
        "The fixed transformation keeps that weaker disagreement visibly weak.\n\n"
        "Its fitted x2 length scale reaches the allowed upper bound of 2. "
        "This suggests weak x2 variation within that model, not proof that x2 "
        "does not matter. The 50:50 selection under this alternative is "
        f"**({point[0]:.6f}, {point[1]:.6f})**.\n\n"
        "H1 also predicts the held-out observations better than the local H2 fit; "
        "H2 performs slightly worse than the mean baseline. Ten held-out errors "
        "cannot settle the function's structure. These checks show that the "
        "hypothesis heatmap depends on which explanations we choose to contrast. "
        "A single query can challenge those explanations, but cannot prove "
        "independence from x2 or identify a global optimum."
    )


def _observations(axis, data):
    axis.scatter(data.x1, data.x2, s=135, facecolors="white", edgecolors="#263238", linewidths=.8)
    for row in data.itertuples():
        axis.text(row.x1, row.x2, str(row.rank), ha="center", va="center", fontsize=8, color="#172126")
    _coordinate_axes(axis, ("x1", "x2"))


def plot_hypothesis_surfaces(data, study, *, save_to=None):
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 5.1), layout="constrained")
    means = [predict_explanation(fit, study["grid"])[0] for fit in study["main"]["fits"]]
    lower, upper = min(data.y.min(), np.min(means)), max(data.y.max(), np.max(means))
    for axis, mean, title in zip(axes, means, ("H1: x1-only ridge", "H2: local two-input features")):
        mesh = axis.pcolormesh(study["axis"], study["axis"], mean.reshape(len(study["axis"]), -1),
                              shading="nearest", cmap="viridis", vmin=lower, vmax=upper, rasterized=True)
        _observations(axis, data)
        axis.set_title(title)
    figure.colorbar(mesh, ax=axes, shrink=.85, label="Predicted mean output (same scale)")
    return _finish(figure, save_to)


def plot_selection_heatmaps(data, study, *, save_to=None):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.9), layout="constrained")
    main = study["main"]
    coverage = study["coverage_gains"] / study["coverage_max"]
    hypothesis = main["disagreement"] / (1 + main["disagreement"])
    surfaces = (coverage, hypothesis, .5 * (coverage + hypothesis))
    selected = (main["selections"][-1], main["selections"][0], main["selections"][2])
    for axis, values, row, title in zip(axes, surfaces, selected,
            ("Coverage benefit", "Distinguish H1 from H2", "50:50 blend")):
        mesh = axis.pcolormesh(study["axis"], study["axis"], values.reshape(len(study["axis"]), -1),
                              shading="nearest", cmap="viridis", vmin=0, vmax=1, rasterized=True)
        _observations(axis, data)
        axis.scatter(*row["point"], s=200, marker="*", color="#ffbf3f", edgecolors="#172126", zorder=5)
        axis.set_title(f"{title}\n({row['point'][0]:.3f}, {row['point'][1]:.3f})")
    figure.colorbar(mesh, ax=axes, shrink=.8, label="Score on a common 0–1 scale")
    return _finish(figure, save_to)

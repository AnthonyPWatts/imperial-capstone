"""Review Function 2's first return using its existing noisy-response models."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / "src"))
sys.path.insert(0, str(STAGE / "scripts"))
from coverage_optimisation import MeanCoverage2D, optimise_average_coverage
from function_2_risk import MODEL_SPECS, posterior_for_pool, score_queries
from function_2_selection import fit_explanation, predict_explanation
from import_round_results import read_return


def prediction_checks(inputs, outputs):
    """Refit the original three hypotheses on each fold at noise SD 0.05."""
    checks = []
    for spec in (None, *MODEL_SPECS):
        predictions = []
        for index in range(len(outputs)):
            keep = np.arange(len(outputs)) != index
            if spec is None:
                prediction = outputs[keep].mean()
            else:
                _, dimensions, separate = spec
                fit = fit_explanation(inputs[keep], outputs[keep], dimensions, 0.05,
                                      separate_lengths=separate)
                prediction = predict_explanation(fit, inputs[index])[0][0]
            predictions.append(float(prediction))
        errors = np.asarray(predictions) - outputs
        checks.append({"name": "Training mean" if spec is None else spec[0],
                       "predictions": predictions, "rmse": float(np.sqrt(np.mean(errors ** 2))),
                       "mae": float(np.mean(np.abs(errors)))})
    return checks


def query_comparison(inputs, outputs, coverage_point, *, resolution, noise_sd):
    """One-query KG on a finite pool with portal-precision query coordinates."""
    axis = np.round(np.linspace(0, .999999, resolution), 6)
    first, second = np.meshgrid(axis, axis)
    # Initial inputs have more precision than the portal accepts. Their rounded
    # neighbours are near-repeats; original inputs remain decision alternatives.
    rounded_inputs = np.clip(np.round(inputs, 6), 0, .999999)
    queries = np.unique(np.vstack([np.c_[first.ravel(), second.ravel()], rounded_inputs,
                                   coverage_point, [0.7, 0.4], [0.8, 0.4]]), axis=0)
    pool = np.vstack([queries, inputs])
    geometry = MeanCoverage2D(inputs)
    rows = []
    for spec in MODEL_SPECS:
        posterior = posterior_for_pool(inputs, outputs, pool, spec, noise_sd)
        gains = score_queries(posterior, len(queries))
        tied = np.flatnonzero(gains >= gains.max() - 1e-12)
        # x2 is statistically irrelevant in the x1-only model; coverage breaks ties.
        index = min(tied, key=lambda i: geometry(queries[i]))
        point = queries[index]
        mean, variance = predict_explanation(posterior["fit"], point)
        best_observed_index = np.flatnonzero(np.all(queries == rounded_inputs[np.argmax(outputs)], axis=1))[0]
        coverage_index = np.flatnonzero(np.all(queries == coverage_point, axis=1))[0]
        rows.append({"model": spec[0], "noise_sd": noise_sd, "resolution": resolution,
                     "pool_size": len(np.unique(pool, axis=0)), "query_count": len(queries), "point": point.tolist(),
                     "portal_input": "-".join(f"{v:.6f}" for v in point),
                     "repeated_input": bool(np.any(np.all(inputs == point, axis=1))),
                     "knowledge_gradient": float(gains[index]),
                     "predicted_mean": float(mean[0]), "observation_sd": float(np.sqrt(variance[0])),
                     "coverage_gain": float(1 - geometry(point) / geometry.mean_before),
                     "incumbent_rounded_neighbour_kg": float(gains[best_observed_index]),
                     "pure_coverage_kg": float(gains[coverage_index]),
                     "kernel": str(posterior["fit"]["model"].kernel_),
                     "warnings": posterior["fit"]["warnings"]})
    return rows


def main():
    source = STAGE / "data/rounds/round-01"
    points, outputs = read_return(source)
    initial = STAGE / "data/initial_data/function_2"
    xp, yp = initial / "initial_inputs.npy", initial / "initial_outputs.npy"
    x, y = np.load(xp, allow_pickle=False), np.load(yp, allow_pickle=False)
    point, result = points[1], outputs[1]
    updated_x, updated_y = np.vstack([x, point]), np.append(y, result)
    accumulated = STAGE / "data/accumulated/function_2"
    if not (np.array_equal(np.load(accumulated / "inputs.npy", allow_pickle=False)[:len(updated_x)], updated_x)
            and np.array_equal(np.load(accumulated / "outputs.npy", allow_pickle=False)[:len(updated_y)], updated_y)):
        raise ValueError("Accumulated observations disagree with the first-round snapshot.")
    destination = STAGE.parent / ".runtime/function-2-round-01"
    destination.mkdir(parents=True, exist_ok=True)
    before, after = MeanCoverage2D(x).mean_before, MeanCoverage2D(updated_x).mean_before
    print("Refitting the three original hypotheses and held-out baselines...", flush=True)
    with threadpool_limits(limits=2):
        old_checks, checks = prediction_checks(x, y), prediction_checks(updated_x, updated_y)
        forecasts = []
        for name, dimensions, separate in MODEL_SPECS:
            fit = fit_explanation(x, y, dimensions, 0.05, separate_lengths=separate)
            mean, variance = predict_explanation(fit, point)
            sd = np.sqrt(variance[0])
            forecasts.append({"model": name, "mean": float(mean[0]), "observation_sd": float(sd),
                              "standardised_error": float((result - mean[0]) / sd),
                              "kernel": str(fit["model"].kernel_), "warnings": fit["warnings"]})
        print("Checking coverage and updated one-query KG (25 and 41 point axes)...", flush=True)
        coverage = optimise_average_coverage(pd.DataFrame(updated_x, columns=["x1", "x2"]))
        choices = query_comparison(updated_x, updated_y, coverage["point"], resolution=41, noise_sd=.05)
        grid_check = query_comparison(updated_x, updated_y, coverage["point"], resolution=25, noise_sd=.05)
        # Same fine pool for the noise comparisons, avoiding confounded grid changes.
        sensitivity = []
        for noise in (.025, .15):
            print(f"Checking the same candidate pool at assumed noise SD {noise}...", flush=True)
            sensitivity.extend(query_comparison(updated_x, updated_y, coverage["point"], resolution=41, noise_sd=noise))

    distances = np.linalg.norm(x - point, axis=1)
    summary = {"function": 2, "round": 1, "point": point.tolist(), "output": float(result),
               "observations_before": len(x), "observations_after": len(updated_x),
               "rank_after": int(1 + np.count_nonzero(updated_y > result)),
               "best_before": float(y.max()), "best_after": float(updated_y.max()),
               "best_point": x[np.argmax(y)].tolist(), "difference_from_best": float(result - y.max()),
               "nearest_initial_point": x[np.argmin(distances)].tolist(),
               "nearest_initial_distance": float(distances.min()),
               "coordinate_shift_from_best": (point - x[np.argmax(y)]).tolist(),
               "mean_distance_before": before, "mean_distance_after": after,
               "coverage_reduction": 1 - after / before,
               "pure_coverage_point": coverage["point"].tolist(),
               "pure_coverage_gain": float(coverage["reduction"]),
               "noise_sd": .05, "checks_before": old_checks, "checks_after": checks,
               "retrospective_predictions_from_initial_data": forecasts,
               "next_query_comparison": choices, "grid_check": grid_check,
               "noise_sensitivity": sensitivity,
               "next_query_status": "conditional_candidates_not_submitted",
               "runtime": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")}}
    paths = [xp, yp, source / "inputs.txt", source / "outputs.txt", source / "email-body.txt",
             source / "source.json", STAGE / "submissions/round-01-proposals.json", Path(__file__),
             STAGE / "scripts/import_round_results.py", STAGE / "src/coverage_optimisation.py",
             STAGE / "src/function_2_risk.py", STAGE / "src/function_2_selection.py"]
    summary["source_sha256"] = {str(p.relative_to(STAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax = axes[0]
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", vmin=y.min(), vmax=y.max(), s=85,
                         edgecolors="#23313c")
    ax.scatter(*x[np.argmax(y)], marker="o", s=185, facecolors="none", edgecolors="#263238", label="Best observed: 0.6112")
    ax.scatter(*point, marker="*", s=190, c="#b92e44", label="Round 1: 0.2834", zorder=4)
    ax.set(xlabel="x1", ylabel="x2", xlim=(0, 1), ylim=(0, 1), aspect="equal", title="A lower response near the observed best")
    ax.legend(loc="upper left", fontsize=9)
    fig.colorbar(scatter, ax=ax, shrink=.8, label="Observed score (higher is better)")
    ax = axes[1]
    for index, row in enumerate(forecasts):
        ax.errorbar(row["mean"], index, xerr=1.96 * row["observation_sd"], fmt="o", color="#315c9d", capsize=5)
    ax.axvline(result, color="#b92e44", label="Actual return", linewidth=2)
    ax.set(yticks=range(3), yticklabels=[row["model"] for row in forecasts],
           xlabel="Predicted noisy score: mean +/- 1.96 assumed SD", title="Predictions fitted only to the initial data", ylim=(-.6, 2.6))
    ax.legend(loc="upper left")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(destination / "result-and-predictions.png", dpi=160)
    plt.close(fig)

    axis = np.linspace(0, .999999, 101)
    first, second = np.meshgrid(axis, axis)
    grid = np.c_[first.ravel(), second.ravel()]
    fits = [fit_explanation(a, b, 2, .05) for a, b in ((x, y), (updated_x, updated_y))]
    surfaces = [predict_explanation(fit, grid)[0].reshape(first.shape) for fit in fits]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    low, high = min(s.min() for s in surfaces), max(s.max() for s in surfaces)
    for ax, surface, values, title in zip(axes, surfaces, (x, updated_x), ("Before: 10 observations", "After: 11 observations")):
        artist = ax.imshow(surface, origin="lower", extent=(0, 1, 0, 1), vmin=low, vmax=high, cmap="viridis")
        ax.scatter(values[:, 0], values[:, 1], c="white", edgecolors="#23313c", s=30)
        ax.set(xlabel="x1", ylabel="x2", title=title)
    axes[1].scatter(*point, marker="*", s=160, c="#f39398", edgecolors="#23313c")
    local_choice = next(row for row in choices if row["model"] == "Local 2D")
    axes[1].scatter(*local_choice["point"], marker="D", s=100, facecolors="none", edgecolors="white", label="Conditional next KG point")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.colorbar(artist, ax=axes, shrink=.85, label="Local 2D GP fitted mean (not the true surface)")
    fig.savefig(destination / "local-model-before-after.png", dpi=160)
    plt.close(fig)
    summary["artefact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(destination.glob("*.png"))}
    (destination / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    compact = {key: summary[key] for key in ("point", "output", "rank_after", "difference_from_best", "nearest_initial_distance",
                                            "mean_distance_before", "mean_distance_after", "coverage_reduction", "pure_coverage_point", "pure_coverage_gain")}
    compact["prediction_checks"] = {key: [{k: row[k] for k in ("name", "rmse", "mae")} for row in summary[key]]
                                    for key in ("checks_before", "checks_after")}
    compact["forecasts"] = forecasts
    compact["next_query_comparison"] = choices
    compact["grid_check"] = grid_check
    compact["noise_sensitivity"] = sensitivity
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

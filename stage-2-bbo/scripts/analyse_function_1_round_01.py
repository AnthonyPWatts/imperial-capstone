"""Reproduce the bounded Function 1 review after the first emailed result."""

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
from scipy.spatial import cKDTree

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / "src"))
sys.path.insert(0, str(STAGE / "scripts"))
from coverage_optimisation import MeanCoverage2D, optimise_average_coverage
from import_round_results import read_return
from multifunction_risk import MODEL_NAMES, fit_model, prediction_checks


def main():
    source = STAGE / "data/rounds/round-01"
    points, outputs = read_return(source)
    initial = STAGE / "data/initial_data/function_1"
    inputs_path, outputs_path = initial / "initial_inputs.npy", initial / "initial_outputs.npy"
    x, y = np.load(inputs_path, allow_pickle=False), np.load(outputs_path, allow_pickle=False)
    point, result = points[0], outputs[0]
    updated_x, updated_y = np.vstack([x, point]), np.append(y, result)
    accumulated = STAGE / "data/accumulated/function_1"
    # Explicit snapshot: later rounds must not silently enter this first-round review.
    assert np.array_equal(np.load(accumulated / "inputs.npy", allow_pickle=False)[:len(updated_x)], updated_x)
    assert np.array_equal(np.load(accumulated / "outputs.npy", allow_pickle=False)[:len(updated_y)], updated_y)
    output = STAGE.parent / ".runtime/function-1-round-01"
    output.mkdir(parents=True, exist_ok=True)
    before = MeanCoverage2D(x).mean_before
    after = MeanCoverage2D(updated_x).mean_before
    data = pd.DataFrame(updated_x, columns=["x1", "x2"])
    print("Checking the next coverage point...", flush=True)
    coverage = optimise_average_coverage(data)
    next_point = coverage["point"]
    assert np.all((next_point >= 0) & (next_point < 1))
    assert not np.any(np.all(updated_x == next_point, axis=1))
    print("Refitting the existing model families for held-out checks...", flush=True)
    # Recompute both snapshots with identical current code and noise assumptions.
    old_checks = prediction_checks(x, y)
    checks = prediction_checks(updated_x, updated_y)
    for values, target in ((old_checks, y), (checks, updated_y)):
        values.append({"name": "Zero", "rmse": float(np.sqrt(np.mean(target ** 2))),
                       "mae": float(np.mean(np.abs(target)))})
    forecasts = []
    for name in MODEL_NAMES:
        fit = fit_model(x, y, name)
        mean, std = fit["model"].predict(point.reshape(1, -1), return_std=True)
        observation_sd = np.hypot(std[0], fit["noise_sd"])
        forecasts.append({"name": name, "mean": float(mean[0]),
                          "observation_sd": float(observation_sd),
                          "standardised_error": float((result - mean[0]) / observation_sd),
                          "warnings": fit["warnings"]})
    distances = np.linalg.norm(x - point, axis=1)
    summary = {
        "function": 1, "round": 1, "point": point.tolist(), "output": float(result),
        "observations_before": len(x), "observations_after": len(updated_x),
        "best_before": float(y.max()), "best_after": float(updated_y.max()),
        "best_point": x[np.argmax(y)].tolist(), "previous_minimum": float(y.min()),
        "change_from_previous_minimum": float(result - y.min()),
        "negative_magnitude_ratio": float(abs(result / y.min())),
        "negative_location_distance": float(distances[np.argmin(y)]),
        "nearest_initial_point": x[np.argmin(distances)].tolist(),
        "nearest_initial_distance": float(distances.min()),
        "mean_distance_before": before, "mean_distance_after": after,
        "coverage_reduction": 1 - after / before,
        "next_coverage_point": next_point.tolist(),
        "next_coverage_portal_input": "-".join(f"{v:.6f}" for v in next_point),
        "next_coverage_mean_distance": coverage["mean_after"],
        "next_coverage_reduction": coverage["reduction"],
        "next_point_status": "conditional_coverage_candidate_not_submitted",
        "noise_fraction": 0.1, "checks_before": old_checks, "checks_after": checks,
        "retrospective_predictions_from_initial_data": forecasts,
        "runtime": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
    }
    paths = [inputs_path, outputs_path, source / "inputs.txt", source / "outputs.txt",
             source / "email-body.txt", source / "source.json", Path(__file__),
             STAGE / "scripts/import_round_results.py", STAGE / "src/coverage_optimisation.py",
             STAGE / "src/multifunction_risk.py", STAGE / "src/function_2_risk.py"]
    summary["source_sha256"] = {str(p.relative_to(STAGE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    plt.rcParams.update({"font.size": 11})
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    ax = axes[0]
    ax.scatter(x[:, 0], x[:, 1], s=70, c="#8898a5", label="Initial observations")
    for index, label, colour in ((np.argmax(y), "Best unchanged", "#216e49"), (np.argmin(y), "Previous minimum", "#b16916")):
        ax.scatter(*x[index], s=110, c=colour, label=label)
    ax.scatter(*point, s=180, marker="*", c="#bb273a", label="Round 1: new minimum")
    ax.scatter(*next_point, s=100, marker="D", facecolors="none", edgecolors="#315c9d", label="Next coverage candidate")
    ax.set(xlabel="x1", ylabel="x2", xlim=(0, 1), ylim=(0, 1), aspect="equal", title="Observed locations and the new result")
    ax.legend(loc="lower right", fontsize=8.5)
    ranks = pd.Series(updated_y).rank(ascending=False, method="min").to_numpy()
    ax = axes[1]
    for positive, marker, colour, label in ((True, "o", "#216e49", "Positive"), (False, "v", "#b16916", "Negative")):
        keep = updated_y > 0 if positive else updated_y < 0
        ax.scatter(ranks[keep], np.log10(np.abs(updated_y[keep])), marker=marker, c=colour, s=65, label=label)
    ax.scatter(ranks[-1], np.log10(abs(result)), marker="*", c="#bb273a", s=180, label="Round 1 (negative)")
    ax.set(xlabel="Output rank (1 = best)", ylabel="log10 of absolute output", title="Magnitude with signs retained", xticks=np.arange(1, 12))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower left")
    figure.savefig(output / "result-and-scale.png", dpi=160)
    plt.close(figure)

    centres = (np.arange(201) + 0.5) / 201
    gx, gy = np.meshgrid(centres, centres)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    surfaces = [cKDTree(values).query(grid)[0].reshape(gx.shape) for values in (x, updated_x)]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, surface, values, title in zip(axes, surfaces, (x, updated_x), ("Before: 10 observations", "After: 11 observations")):
        artist = ax.imshow(surface, origin="lower", extent=(0, 1, 0, 1), cmap="viridis", vmin=0, vmax=surfaces[0].max())
        ax.scatter(values[:, 0], values[:, 1], c="white", edgecolors="#172126", s=35)
        ax.set(title=title, xlabel="x1", ylabel="x2")
    axes[1].scatter(*point, marker="*", s=170, c="#fa8b80", edgecolors="#172126")
    figure.colorbar(artist, ax=axes, label="Distance to nearest observation (geometric coverage)", shrink=0.85)
    figure.savefig(output / "coverage-before-after.png", dpi=160)
    plt.close(figure)
    summary["artefact_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.png"))}
    (output / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("source_sha256", "artefact_sha256", "runtime")}, indent=2))


if __name__ == "__main__":
    main()

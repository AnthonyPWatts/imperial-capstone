"""Bounded knowledge-gradient studies across the capstone's input dimensions.

The finite-set KG integral and posterior simulation primitives are shared with
the earlier Function 2 study. No course function is evaluated by this module.
"""

from itertools import product
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Hyperparameter, Kernel, Matern
from threadpoolctl import threadpool_limits

from function_2_risk import estimate_current_risk, score_queries, simulate_remaining_queries


MODEL_NAMES = ("Additive", "Local", "Flexible")
UPPER = 0.999999


def study_fingerprint(data):
    """Include both the study and the shared KG/posterior calculations."""
    digest = hashlib.sha256(data.to_numpy().tobytes())
    for source in (Path(__file__), Path(__file__).with_name("function_2_risk.py")):
        digest.update(source.name.encode("utf-8"))
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


class AdditiveMatern(Kernel):
    """Average of independent one-input Matérn 3/2 kernels; no interactions."""

    def __init__(self, length_scale=(0.3, 0.3), length_scale_bounds=(0.03, 2.0)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds,
                              len(self.length_scale))

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is not None and eval_gradient:
            raise ValueError("Kernel gradients require Y=None.")
        Y = X if Y is None else np.atleast_2d(Y)
        dimensions = X.shape[1]
        if len(self.length_scale) != dimensions:
            raise ValueError("One length scale is required for every input.")
        covariance = np.zeros((len(X), len(Y)))
        gradient = []
        for axis, length in enumerate(self.length_scale):
            distance = np.sqrt(3) * np.abs(X[:, axis, None] - Y[None, :, axis]) / length
            decay = np.exp(-distance)
            covariance += (1 + distance) * decay / dimensions
            if eval_gradient:
                gradient.append(distance ** 2 * decay / dimensions)
        if eval_gradient:
            derivative = (np.stack(gradient, axis=-1) if not self.hyperparameter_length_scale.fixed
                          else np.empty((*covariance.shape, 0)))
            return covariance, derivative
        return covariance

    def diag(self, X):
        return np.ones(len(X))

    def is_stationary(self):
        return True

    def __repr__(self):
        return f"AdditiveMatern(length_scale={np.asarray(self.length_scale).round(4).tolist()})"


def fit_model(inputs, outputs, name, noise_fraction=0.1, *, seed=7):
    dimensions = inputs.shape[1]
    if name == "Additive":
        kernel = AdditiveMatern((0.3,) * dimensions)
    elif name == "Local":
        kernel = Matern(0.3, (0.03, 2.0), nu=1.5)
    elif name == "Flexible":
        kernel = Matern([0.3] * dimensions, (0.03, 2.0), nu=1.5)
    else:
        raise ValueError(name)
    model = GaussianProcessRegressor(
        kernel=ConstantKernel(1.0, (0.05, 10.0)) * kernel,
        normalize_y=True, alpha=noise_fraction ** 2,
        n_restarts_optimizer=3, random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(inputs, outputs)
    return {"model": model, "name": name, "noise_fraction": noise_fraction,
            "noise_sd": float(noise_fraction * np.std(outputs)),
            "warnings": [str(item.message) for item in caught]}


def prediction_checks(inputs, outputs, noise_fraction=0.1):
    """Refit on each training fold; include a simple held-out mean baseline."""
    checks = []
    for name in ("Training mean", *MODEL_NAMES):
        predictions = []
        for index in range(len(inputs)):
            keep = np.arange(len(inputs)) != index
            if name == "Training mean":
                prediction = outputs[keep].mean()
            else:
                fit = fit_model(inputs[keep], outputs[keep], name, noise_fraction)
                prediction = fit["model"].predict(inputs[index:index + 1])[0]
            predictions.append(float(prediction))
        errors = np.asarray(predictions) - outputs
        checks.append({"name": name, "predictions": predictions,
                       "rmse": float(np.sqrt(np.mean(errors ** 2))),
                       "mae": float(np.mean(np.abs(errors)))})
    return checks


def candidate_pool(inputs, outputs, *, fine=False):
    """A grid up to 3D; space-filling and local candidates above 3D.

    Both pools include observations, corners and perturbations around the best
    three observations. Fine pools add a different global Sobol sample as well
    as more local points. Six decimals are a file/portal format, not certainty.
    """
    dimensions = inputs.shape[1]
    seed = 29 if fine else 17
    if dimensions <= 3:
        resolution = (13 if fine else 9) if dimensions == 3 else (41 if fine else 25)
        axis = np.round(np.linspace(0, UPPER, resolution), 6)
        global_points = np.array(list(product(axis, repeat=dimensions)))
    else:
        global_points = qmc.Sobol(dimensions, scramble=True, seed=seed).random_base2(10 if fine else 9)
    corners = np.array(list(product((0.0, UPPER), repeat=dimensions)))
    local = []
    for offset, index in enumerate(np.argsort(outputs)[-3:]):
        points = qmc.Sobol(dimensions, scramble=True, seed=seed + 100 + offset).random_base2(7 if fine else 6)
        local.append(np.clip(inputs[index] + 0.4 * (points - 0.5), 0, UPPER))
        # Explicitly test movements in one coordinate with the others held fixed.
        for dimension in range(dimensions):
            for delta in (-0.2, -0.1, 0.1, 0.2):
                point = inputs[index].copy()
                point[dimension] = np.clip(point[dimension] + delta, 0, UPPER)
                local.append(point[None, :])
    points = np.vstack([global_points, corners, *local, inputs])
    return np.unique(np.round(np.clip(points, 0, UPPER), 6), axis=0)


def coverage_scores(inputs, queries, *, power=13, seed=41):
    """Fractional decrease of uniform-domain mean nearest-sample distance."""
    domain = qmc.Sobol(inputs.shape[1], scramble=True, seed=seed).random_base2(power)
    before = cKDTree(inputs).query(domain)[0]
    baseline = before.mean()
    scores = np.empty(len(queries))
    for start in range(0, len(queries), 32):
        points = queries[start:start + 32]
        distances = np.sqrt(np.sum((domain[:, None, :] - points[None, :, :]) ** 2, axis=-1))
        scores[start:start + len(points)] = np.maximum(before[:, None] - distances, 0).mean(axis=0) / baseline
    return scores, float(baseline)


def posterior_on_pool(fit, pool):
    means, covariance = fit["model"].predict(pool, return_cov=True)
    return {"name": fit["name"], "fit": fit, "means": means, "covariance": covariance,
            "pool_indices": np.arange(len(pool)), "noise_sd": fit["noise_sd"]}


def _scored_models(fits, pool, *, risk_draws=0):
    models = []
    for fit in fits:
        posterior = posterior_on_pool(fit, pool)
        gains = score_queries(posterior, len(pool))
        item = {"name": fit["name"], "posterior": posterior,
                "gains": gains, "best_index": int(np.argmax(gains)),
                "kernel": str(fit["model"].kernel_), "warnings": fit["warnings"]}
        if risk_draws:
            item["current"] = estimate_current_risk(posterior, draws=risk_draws)
        models.append(item)
    return models


def _save(study, output):
    """Save results without Python estimators or covariance matrices."""
    if output is None:
        return
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    saved = {key: value for key, value in study.items() if key not in ("models", "fits")}
    saved["models"] = [{key: value for key, value in model.items() if key != "posterior"}
                        for model in study["models"]]
    def encode(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        raise TypeError(type(value).__name__)
    (destination / "risk-study.json").write_text(
        json.dumps(saved, default=encode, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def analyse_risk(data, function_id, *, output=None):
    """Fit, score, and check the proposed query against a larger decision pool."""
    started = perf_counter()
    columns = [name for name in data if name.startswith("x")]
    inputs, outputs = data[columns].to_numpy(), data.y.to_numpy()
    with threadpool_limits(limits=2):
        checks = prediction_checks(inputs, outputs)
        fits = [fit_model(inputs, outputs, name) for name in MODEL_NAMES]
        coarse = candidate_pool(inputs, outputs)
        fine = candidate_pool(inputs, outputs, fine=True)
        if function_id == 1:
            coverage_reference = np.array([[0.468429, 0.435440]])
            coarse = np.unique(np.vstack([coarse, coverage_reference]), axis=0)
            fine = np.unique(np.vstack([fine, coverage_reference]), axis=0)
        coarse_models = _scored_models(fits, coarse)
        fine_models = _scored_models(fits, fine)
        grid_checks = [{"model": name,
                        "coarse_point": coarse[small["best_index"]],
                        "coarse_gain": float(small["gains"].max()),
                        "fine_point": fine[large["best_index"]],
                        "fine_gain": float(large["gains"].max())}
                       for name, small, large in zip(MODEL_NAMES, coarse_models, fine_models)]
        # Keep the rollout pool bounded, while including the finer search's winners.
        extra = np.array([fine[m["best_index"]] for m in fine_models])
        pool = np.unique(np.vstack([coarse, extra]), axis=0)
        del coarse_models, fine_models
        coverage, before = coverage_scores(inputs, pool)
        models = _scored_models(fits, pool, risk_draws=4096)
        residual = np.stack([m["current"]["risk"] - m["gains"] for m in models])
        worst_after = residual.max(axis=0)
        robust_index = int(np.argmin(worst_after))
        # Keep a coverage start, a region suggested by raw observations, each
        # model's winner, and the conservative choice. Merge duplicate points.
        best_observed = inputs[np.argmax(outputs)]
        near_best = np.linalg.norm(pool - best_observed, axis=1)
        eligible = np.where((near_best > 0.08) & (near_best < 0.25))[0]
        consensus_mean = np.mean([m["posterior"]["means"] for m in models], axis=0)
        local_index = int(eligible[np.argmax(consensus_mean[eligible])])
        selected = [("Coverage", int(np.argmax(coverage))), ("Near best", local_index)]
        if function_id == 1:
            previous_index = int(cKDTree(pool).query([0.468429, 0.435440])[1])
            selected.append(("Existing coverage proposal", previous_index))
        selected += [("KG " + m["name"], m["best_index"]) for m in models]
        selected += [("Conservative KG", robust_index)]
        candidate_names, candidate_indices = [], []
        for name, index in selected:
            if index in candidate_indices:
                place = candidate_indices.index(index)
                candidate_names[place] += " / " + name
            else:
                candidate_names.append(name)
                candidate_indices.append(index)
        # Independent denser integration checks coverage, not modelled response.
        checked_coverage, checked_before = coverage_scores(
            inputs, pool[candidate_indices], power=16, seed=101)
    study = {
        "fingerprint": study_fingerprint(data),
        "function_id": function_id, "dimensions": len(columns), "observations": len(inputs),
        "output_sd": float(np.std(outputs)), "noise_sd": fits[0]["noise_sd"],
        "checks": checks, "fits": fits, "pool": pool, "models": models,
        "coarse_count": len(coarse), "fine_count": len(fine), "grid_checks": grid_checks,
        "coverage": coverage, "coverage_mean_before": before,
        "checked_coverage": checked_coverage, "checked_coverage_mean_before": checked_before,
        "candidate_names": candidate_names, "candidate_indices": candidate_indices,
        "robust_index": robust_index, "worst_after": worst_after,
        "elapsed_one_step_seconds": perf_counter() - started,
    }
    _save(study, output)
    return study


def restore_study(data, output):
    """Restore this run's saved numerical results, rebuilding only fitted objects.

    This is an explicit continuation of a computed study, not automatic cache
    invalidation. Use analyse_risk again if data, models or search settings change.
    """
    study = json.loads((Path(output) / "risk-study.json").read_text(encoding="utf-8"))
    for key in ("pool", "coverage", "checked_coverage", "worst_after"):
        study[key] = np.asarray(study[key])
    inputs = data[[name for name in data if name.startswith("x")]].to_numpy()
    with threadpool_limits(limits=2):
        study["fits"] = [fit_model(inputs, data.y.to_numpy(), name) for name in MODEL_NAMES]
        for model, fit in zip(study["models"], study["fits"]):
            model["gains"] = np.asarray(model["gains"])
            model["posterior"] = posterior_on_pool(fit, study["pool"])
            if "rollout" in model:
                for key in ("losses", "mean", "mc_se"):
                    model["rollout"][key] = np.asarray(model["rollout"][key])
    return study


def add_noise_checks(data, study, *, output=None):
    started = perf_counter()
    inputs = data[[name for name in data if name.startswith("x")]].to_numpy()
    rows = []
    with threadpool_limits(limits=2):
        for fraction in (0.03, 0.3):
            for name in MODEL_NAMES:
                fit = fit_model(inputs, data.y.to_numpy(), name, fraction)
                model = _scored_models([fit], study["pool"])[0]
                winner = model["best_index"]
                recommended = study["robust_index"]
                rows.append({"model": name, "noise_fraction": fraction,
                             "noise_sd": fit["noise_sd"], "point": study["pool"][winner],
                             "gain": float(model["gains"][winner]),
                             "recommended_gain": float(model["gains"][recommended]),
                             "warnings": fit["warnings"]})
    study["noise_checks"] = rows
    study["elapsed_noise_seconds"] = perf_counter() - started
    _save(study, output)


def add_rollouts(study, *, output=None, trials=64):
    started = perf_counter()
    with threadpool_limits(limits=2):
        for model in study["models"]:
            model["rollout"] = simulate_remaining_queries(
                model["posterior"], study["candidate_indices"], len(study["pool"]), trials=trials)
    study["elapsed_rollout_seconds"] = perf_counter() - started
    _save(study, output)


def ensure_risk_results(data, function_id, output, *, progress=None):
    """Compute a complete study once; reuse it only for identical data and code."""
    path = Path(output) / "risk-study.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if (saved.get("fingerprint") == study_fingerprint(data)
                and "elapsed_rollout_seconds" in saved and "noise_checks" in saved):
            return saved
    report = progress if progress is not None else lambda message: None
    report(f"Function {function_id}: fits, prediction checks and one-query search")
    study = analyse_risk(data, function_id, output=output)
    report(f"Function {function_id}: noise sensitivity")
    add_noise_checks(data, study, output=output)
    report(f"Function {function_id}: thirteen-query policy experiment")
    add_rollouts(study, output=output)
    report(f"Function {function_id}: complete")
    return json.loads(path.read_text(encoding="utf-8"))

"""Weekly GP decisions and neural comparators, with explicit targets and budgets."""

from collections import Counter
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from model_review_helpers import (
    build_review_candidates, query_prediction, resolve_unmodelled_coordinates,
    score_review_candidates, weight_sensitivity,
)
from surrogate_study_helpers import (
    finite_difference_gradient, network_input_gradient,
)


def select_gp(run, gp_names, *, metric="LOO RMSE"):
    """Select only a declared GP; neural fit variation is not posterior uncertainty."""
    if metric not in ("LOO RMSE", "LOO MAE") or not gp_names:
        raise ValueError("Declare GP names and a held-out error criterion")
    table = run["table"].set_index("Model")
    name = table.loc[list(gp_names), metric].idxmin()
    return dict(name=name, model=run["fitted"][name], metric=metric,
                error=float(table.loc[name, metric]))


def paired_peer_review(models, official, peers, target, peer_target, official_run, *, verbose=True):
    """Score the same official targets; the held-out official row never enters training.

    Only GP candidates need a second fit. All official outputs remain the owner's
    progress record even when a fit uses retained, self-reported peer training rows.
    """
    official, peers = np.asarray(official), np.asarray(peers)
    target, peer_target = np.asarray(target), np.asarray(peer_target)
    if (official.ndim != 2 or peers.ndim != 2 or official.shape[1] != peers.shape[1]
            or target.shape != (len(official),) or peer_target.shape != (len(peers),)
            or not len(peers)):
        raise ValueError("Supply matching official and peer inputs and targets")
    prediction, fitted, rows, warning_details = {}, {}, [], {}
    for name, estimator in models.items():
        predicted, counts = np.empty(len(target)), Counter()
        for held in range(len(target)):
            train = np.arange(len(target)) != held
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fold = clone(estimator).fit(np.vstack([official[train], peers]),
                                             np.r_[target[train], peer_target])
                predicted[held] = fold.predict(official[held:held+1])[0]
            counts.update(f"{w.category.__name__}: {w.message}" for w in caught)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted[name] = clone(estimator).fit(np.vstack([official, peers]), np.r_[target, peer_target])
        counts.update(f"{w.category.__name__}: {w.message}" for w in caught)
        if not np.isfinite(predicted).all():
            raise ValueError("Non-finite peer-assisted predictions")
        own = official_run["predictions"][name]
        rows.append({"Model": name, "Official LOO RMSE": root_mean_squared_error(target, own),
                     "Peer-assisted LOO RMSE": root_mean_squared_error(target, predicted),
                     "Official LOO MAE": mean_absolute_error(target, own),
                     "Peer-assisted LOO MAE": mean_absolute_error(target, predicted),
                     "Fit warnings": sum(counts.values())})
        prediction[name], warning_details[name] = predicted, dict(counts)
        if verbose:
            print(f"Peer-assisted official-target review complete: {name}", flush=True)
    return dict(table=pd.DataFrame(rows), predictions=prediction, fitted=fitted, warnings=warning_details)


def choose_training_scope(official_run, gp_names, *, peer_run=None, metric="LOO RMSE"):
    """Compare each scope's best declared GP on identical official targets.

    Equal errors retain the official fit. This is model selection on a small,
    adaptively sampled set, not an independent estimate of the selected policy.
    """
    own = select_gp(official_run, gp_names, metric=metric)
    own.update(scope="official", official_target_error=own["error"])
    if peer_run is None:
        return own
    column = "Peer-assisted " + metric
    peer_table = peer_run["table"].set_index("Model")
    name = peer_table.loc[list(gp_names), column].idxmin()
    error = float(peer_table.loc[name, column])
    if error < own["error"]:
        return dict(name=name, model=peer_run["fitted"][name], metric=metric, error=error,
                    scope="official_plus_retained_peers", official_target_error=error)
    return own


def neural_architecture_review(run, metadata, *, metric="LOO RMSE"):
    """Select an architecture by average seed error, and score its seed-mean predictions."""
    if metric not in ("LOO RMSE", "LOO MAE") or not metadata:
        raise ValueError("Declare neural models and an error criterion")
    table = run["table"].set_index("Model")
    seed_rows = [{"Model": name, **spec, metric: float(table.loc[name, metric])}
                 for name, spec in metadata.items()]
    seeds = pd.DataFrame(seed_rows)
    architecture = seeds.groupby("architecture")[metric].agg(["mean", "std", "min", "max"])
    architecture = architecture.sort_values(["mean", "architecture"]).reset_index()
    selected = str(architecture.iloc[0].architecture)
    names = [name for name, spec in metadata.items() if spec["architecture"] == selected]
    predictions = np.mean([run["predictions"][name] for name in names], axis=0)
    return dict(table=architecture, seeds=seeds, architecture=selected, names=names,
                models=[run["fitted"][name] for name in names], predictions=predictions, metric=metric)


def make_candidate_review(inputs, anchor, *, input_columns, search_settings,
                          minimum_distance=0., excluded_inputs=None):
    """Use the existing broad/local search, plus a finite-pool geometric comparator.

    Coverage here means the farthest eligible point in the frozen pool. It is
    explicitly a proxy, not a rerun of continuous minimax coverage optimisation.
    """
    inputs = np.asarray(inputs, float)
    observed = inputs if excluded_inputs is None else np.asarray(excluded_inputs, float)
    pool = build_review_candidates(observed, anchor, np.empty((0, inputs.shape[1])),
                                   input_columns=input_columns, **search_settings)
    pool["nearest_observed_distance"] = cdist(pool[list(input_columns)], observed).min(axis=1)
    pool = pool.loc[pool.nearest_observed_distance >= minimum_distance].reset_index(drop=True)
    if pool.empty or not np.isfinite(minimum_distance) or minimum_distance < 0:
        raise ValueError("No eligible candidates or invalid minimum distance")
    coverage = pool.loc[pool.nearest_observed_distance.idxmax(), list(input_columns)].to_numpy(dtype=float)
    return pool, coverage


def review_gp_query(selected, candidates, inputs, target, *, input_columns, weights,
                    weight=1.96, feature_sets=None, precision=6, minimum_distance=0.,
                    training_inputs=None):
    """Score exact six-decimal inputs and preserve spacing after any geometric tie-break."""
    model = selected["model"]
    scores = score_review_candidates(model, candidates, input_columns=input_columns,
                                     weight=weight, precision=precision)
    axes = (feature_sets or {}).get(selected["name"], tuple(range(len(input_columns))))
    point = scores.iloc[0][list(input_columns)].to_numpy(dtype=float)
    if len(axes) != len(input_columns):
        point = resolve_unmodelled_coordinates(point, inputs, modelled_axes=axes, precision=precision)
    if cdist(point[None], inputs).min() < minimum_distance:
        raise ValueError("Geometric tie-break violated the minimum query distance")
    fitted_inputs = inputs if training_inputs is None else training_inputs
    proposal = query_prediction(model, point, fitted_inputs, target, weight=weight, precision=precision)
    proposal["nearest_observed_distance"] = float(cdist(point[None], inputs).min())
    sensitivity = weight_sensitivity(scores, inputs, input_columns=input_columns,
        modelled_axes=axes, weights=weights, precision=precision)
    return dict(scores=scores, proposal=proposal, weight_sensitivity=sensitivity)


def neural_candidate_review(networks, candidates, *, input_columns):
    """Give each architecture the same candidates; seed spread is labelled as variation."""
    points = candidates[list(input_columns)].to_numpy(dtype=float)
    predicted = np.asarray([model.predict(points) for model in networks["models"]])
    mean, spread = predicted.mean(axis=0), predicted.std(axis=0, ddof=1)
    index = int(np.argmax(mean))
    return dict(point=points[index].tolist(), predicted_target=float(mean[index]),
                seed_prediction_sd=float(spread[index]), architecture=networks["architecture"],
                portal_input="-".join(f"{x:.6f}" for x in points[index]),
                criterion="mean prediction across all declared seeds; not GP UCB")


def neural_gradient_review(networks, anchor, *, relative_tolerance=.005):
    """Check derivatives, explicitly separating a ReLU kink from a smooth probe.

    A central difference across an activation boundary need not match either
    branch derivative. Keep the anchor check visible; validate the implementation
    at a fixed nearby point only when a near-zero ReLU activation explains it.
    """
    rows = []
    for name, model in zip(networks["names"], networks["models"]):
        gradient = network_input_gradient(model, anchor)[0]
        numerical = finite_difference_gradient(model.predict, np.atleast_2d(anchor))[0]
        relative = float(np.max(np.abs(gradient-numerical) / np.maximum(1., np.abs(gradient))))
        anchor_passes = relative <= relative_tolerance
        check_point, check_error, status = np.asarray(anchor), relative, "anchor"
        pipeline = model.regressor_
        network = pipeline.named_steps["model"]
        if not anchor_passes and network.activation == "relu":
            activation = pipeline.named_steps["scale"].transform(np.atleast_2d(anchor))
            margins = []
            for weights, bias in zip(network.coefs_[:-1], network.intercepts_[:-1]):
                linear = activation @ weights + bias
                margins.append(float(np.abs(linear).min()))
                activation = np.maximum(linear, 0)
            if min(margins) <= 1e-6:
                check_point = np.clip(np.asarray(anchor) + np.linspace(.0001, .0002, len(anchor)), 0, .999999)
                analytic = network_input_gradient(model, check_point)[0]
                numeric = finite_difference_gradient(model.predict, check_point, step=1e-7)[0]
                check_error = float(np.max(np.abs(analytic-numeric) / np.maximum(1., np.abs(analytic))))
                status = "nearby smooth probe; anchor is near a ReLU kink"
        rows.append(dict(Model=name, maximum_relative_error=check_error,
                         passes=check_error <= relative_tolerance, check_location=status,
                         check_point=check_point.tolist(), anchor_relative_error=relative,
                         anchor_check_passes=anchor_passes, gradient=gradient.tolist()))
    return pd.DataFrame(rows)

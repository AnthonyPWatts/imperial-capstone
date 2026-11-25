"""Cohort surrogate experiments with official, portal-grouped validation targets."""

from collections import Counter, defaultdict
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process.kernels import Matern, Product
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from surrogate_study_helpers import make_study_models


GP_MATERN = "GP Matérn 3/2"
GP_PEER_NOISE = "GP higher peer noise"
GP_MEDIAN = "GP median duplicate scores"
GP_WARP = "GP asinh target"
NEURAL = "Tanh-8 seed ensemble"
SVR_STANDARD = "SVR standard C=1"
SVR_ROBUST = "SVR robust C=1"
SVR_BLEND = "SVR scaling blend"
MIXED_BLEND = "GP–SVR–MLP mean"
GP_SELECTION = "Nested GP selection"
MODEL_SELECTION = "Nested model selection"
NESTED_BLEND = "Nested GP blend"


def grouped_official_folds(points, indices, *, splits=None, seed=None):
    """Split whole six-decimal portal groups, returning original official indices."""
    indices = np.asarray(indices, dtype=int)
    _, groups = np.unique(np.asarray(points)[indices].round(6), axis=0, return_inverse=True)
    group_count = int(groups.max()) + 1
    folds = KFold(n_splits=group_count if splits is None else min(splits, group_count),
                  shuffle=seed is not None, random_state=seed)
    for train, test in folds.split(np.arange(group_count)):
        yield indices[np.isin(groups, train)], indices[np.isin(groups, test)]


def _training_peers(matching_peer, train):
    omitted = np.ones(len(matching_peer), dtype=bool)
    omitted[train] = False
    return ~matching_peer[omitted].any(axis=0)


def _median_repeats(x, y):
    """Use one representative original input and the median per portal group."""
    _, first, inverse, counts = np.unique(x.round(6), axis=0, return_index=True,
                                          return_inverse=True, return_counts=True)
    labels = np.argsort(first)
    values = y[first[labels]].copy()
    for position, label in enumerate(labels):
        if counts[label] > 1:
            values[position] = np.median(y[inverse == label])
    return x[first[labels]], values


def _gp_step(estimator):
    return estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator


def _matern_gp(reference):
    estimator = clone(reference)
    gp = _gp_step(estimator)
    old = gp.kernel
    rbf = old.k2 if isinstance(old, Product) else old
    kernel = Matern(rbf.length_scale, rbf.length_scale_bounds, nu=1.5)
    gp.set_params(kernel=old.k1 * kernel if isinstance(old, Product) else kernel)
    return estimator


def _regression_estimators(dimensions):
    def svr(scale, c):
        return TransformedTargetRegressor(
            regressor=make_pipeline(scale, SVR(C=c, epsilon=.1, gamma="scale")),
            transformer=StandardScaler(),
        )

    return {
        "Pooled mean": DummyRegressor(),
        "Standard ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.)),
        "Quadratic ridge": make_pipeline(PolynomialFeatures(2, include_bias=False),
                                         StandardScaler(), Ridge(alpha=1.)),
        "PCA ridge": make_pipeline(StandardScaler(), PCA(n_components=.95, svd_solver="full"),
                                    Ridge(alpha=1.)),
        "5-NN distance weighting": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "RBF kernel ridge": make_pipeline(StandardScaler(),
            KernelRidge(alpha=.1, kernel="rbf", gamma=1 / dimensions)),
        "SVR standard C=0.3": svr(StandardScaler(), .3),
        SVR_STANDARD: svr(StandardScaler(), 1.),
        "SVR standard C=3": svr(StandardScaler(), 3.),
        SVR_ROBUST: svr(RobustScaler(), 1.),
        "Depth-3 regression tree": DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=42),
        "Random forest": RandomForestRegressor(n_estimators=48, min_samples_leaf=2,
                                               max_features=.8, random_state=42, n_jobs=1),
        "Shallow gradient boosting": GradientBoostingRegressor(n_estimators=48, learning_rate=.06,
            max_depth=2, min_samples_leaf=3, random_state=42),
    }


def _cluster_ridge(x, y, test):
    """Split training geometry in two and use a regularised fit in each region."""
    scale = StandardScaler().fit(x)
    training = scale.transform(x)
    locations = scale.transform(test)
    clusters = KMeans(n_clusters=2, n_init=5, random_state=42).fit(training)
    membership = clusters.predict(locations)
    prediction = np.empty(len(test))
    for group in np.unique(membership):
        members = clusters.labels_ == group
        if members.sum() < 4:
            members = np.ones(len(x), dtype=bool)
        fit = Ridge(alpha=1.).fit(training[members], y[members])
        prediction[membership == group] = fit.predict(locations[membership == group])
    return prediction


def _top_quartile(estimator, x, y, test):
    """Estimate a score from training-only class probabilities and class means."""
    high = y >= np.quantile(y, .75)
    if np.all(high):
        return np.full(len(test), y.mean())
    fit = clone(estimator).fit(x, high)
    probability = fit.predict_proba(test)[:, 1]
    low_mean, high_mean = y[~high].mean(), y[high].mean()
    return low_mean + probability * (high_mean - low_mean)


def _warped_gp(estimator, x, y, test):
    """Use a fold-local signed asinh warp and invert the predicted transformed mean."""
    centre = np.median(y)
    lower, upper = np.quantile(y, [.25, .75])
    scale = upper - lower or y.std() or 1.
    fitted = clone(estimator).fit(x, np.arcsinh((y - centre) / scale))
    return centre + scale * np.sinh(fitted.predict(test))


def _fit_predict(estimator, x, y, test):
    return clone(estimator).fit(x, y).predict(test)


def _capture(name, fit, warning_counts):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions = np.asarray(fit(), dtype=float)
    warning_counts[name].update(f"{w.category.__name__}: {w.message}" for w in caught)
    if not np.isfinite(predictions).all():
        raise ValueError(f"{name} produced a non-finite prediction")
    return predictions


def _loss(target, predicted, metric):
    residual = np.asarray(predicted) - target
    return float(np.abs(residual).mean() if metric == "MAE" else np.sqrt(np.square(residual).mean()))


def _ranking(target, predicted):
    if not np.ptp(target) or not np.ptp(predicted):
        return None
    return float(spearmanr(target, predicted).statistic)


def _pivot_predictions(folds, names, size, predict):
    result = {name: np.full(size, np.nan) for name in names}
    for train, test in folds:
        values = predict(train, test)
        for name in names:
            result[name][test] = values[name]
    return result


def _partition_result(y, predictions, test, metric):
    actual = y[test]
    winner = int(test[np.argmax(actual)])
    selected = {}
    for name, values in predictions.items():
        pick = int(test[np.argmax(values)])
        selected[name] = dict(selected_index=pick, best_index=winner,
                              observed_gap=float(y[winner] - y[pick]),
                              error=_loss(actual, values, metric))
    return selected


def _study_bank(size, gp_models, reference_name):
    added = _regression_estimators(size)
    reference = gp_models[reference_name]
    estimators = {**gp_models, GP_MATERN: _matern_gp(reference), **added}
    networks, _, metadata = make_study_models(size,
        architectures=[{"name": "tanh-8", "layers": [8], "activation": "tanh"}],
        seeds=[11, 42, 73], gp_restarts=0)
    classifiers = {
        "Top-quartile logistic": make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=500)),
        "Top-quartile Gaussian NB": make_pipeline(StandardScaler(), GaussianNB()),
    }
    contextual = (GP_PEER_NOISE, GP_MEDIAN, GP_WARP, "K-means local ridge", *classifiers, NEURAL,
                  SVR_BLEND, MIXED_BLEND)
    names = (*estimators, *contextual)
    gp_names = (*gp_models, GP_MATERN, GP_PEER_NOISE, GP_MEDIAN, GP_WARP)
    alternative_names = (*added, "K-means local ridge", *classifiers, NEURAL, SVR_BLEND)
    def predict(x, y, query, official_rows, warning_counts):
        result = {}
        for name, model in estimators.items():
            result[name] = _capture(name, lambda: _fit_predict(model, x, y, query), warning_counts)

        weighted = clone(reference)
        step = _gp_step(weighted)
        step.set_params(alpha=np.r_[np.full(official_rows, step.alpha),
                                    np.full(len(x) - official_rows, step.alpha * 9.)])
        result[GP_PEER_NOISE] = _capture(GP_PEER_NOISE, lambda: weighted.fit(x, y).predict(query), warning_counts)
        median_x, median_y = _median_repeats(x, y)
        result[GP_MEDIAN] = _capture(GP_MEDIAN, lambda: _fit_predict(reference, median_x, median_y, query),
                                    warning_counts)
        result[GP_WARP] = _capture(GP_WARP, lambda: _warped_gp(reference, x, y, query), warning_counts)
        result["K-means local ridge"] = _capture("K-means local ridge", lambda: _cluster_ridge(x, y, query),
                                                warning_counts)
        for name, model in classifiers.items():
            result[name] = _capture(name, lambda: _top_quartile(model, x, y, query), warning_counts)
        result[NEURAL] = np.mean([_capture(NEURAL, lambda: _fit_predict(networks[name], x, y, query), warning_counts)
                                  for name in metadata], axis=0)
        result[SVR_BLEND] = (result[SVR_STANDARD] + result[SVR_ROBUST]) / 2
        result[MIXED_BLEND] = (result[reference_name] + result[SVR_BLEND] + result[NEURAL]) / 3
        return result

    return names, gp_names, alternative_names, predict


def _selected_methods(predictions, target, metric, gp_names, alternative_names):
    scores = {name: _loss(target, values, metric) for name, values in predictions.items()}
    gp = min(gp_names, key=scores.__getitem__)
    partner = min(alternative_names, key=scores.__getitem__)
    weight = min((.25, .5, .75), key=lambda w: _loss(
        target, w * predictions[gp] + (1 - w) * predictions[partner], metric))
    return gp, partner, weight, min(scores, key=scores.__getitem__)


def predict_official_blend(gp_models, reference_name, inputs, target, query, *, metric, seed):
    """Choose and refit the study's blend with official training rows only."""
    names, gp_names, alternative_names, predict = _study_bank(inputs.shape[1], gp_models, reference_name)
    warning_counts = defaultdict(Counter)

    def fold(train, test):
        return predict(inputs[train], target[train], inputs[test], len(train), warning_counts)

    splits = grouped_official_folds(inputs, np.arange(len(inputs)), splits=4, seed=seed)
    internal = _pivot_predictions(splits, names, len(inputs), fold)
    gp, partner, weight, _ = _selected_methods(internal, target, metric, gp_names, alternative_names)
    fitted = predict(inputs, target, query, len(inputs), warning_counts)
    choice = dict(gp=gp, partner=partner, gp_weight=weight, inner_split_seed=seed,
                  warnings={name: dict(counts) for name, counts in warning_counts.items() if counts})
    return weight * fitted[gp] + (1 - weight) * fitted[partner], choice


def run_cohort_surrogate_study(review, gp_models, reference_name, official_target, peer_target,
                               *, metric, seeds=(42, 73)):
    """Screen fixed methods, then validate model and blend selection on outer folds."""
    if metric not in ("MAE", "RMSE") or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require MAE/RMSE and distinct outer split seeds")

    columns = review["input_columns"]
    official = review["own_data"]
    own = official[columns].to_numpy(dtype=float)
    peers = review["data"].loc[review["peer_mask"], columns].to_numpy(dtype=float)
    target, peer_y = np.asarray(official_target, float), np.asarray(peer_target, float)
    if target.shape != (len(own),) or peer_y.shape != (len(peers),):
        raise ValueError("Target arrays must align with the official and peer snapshots")

    indices = np.arange(len(own))
    matching_peer = np.all(own.round(6)[:, None, :] == peers.round(6)[None, :, :], axis=2)
    names, gp_names, alternative_names, predict = _study_bank(len(columns), gp_models, reference_name)
    warning_counts = defaultdict(Counter)
    fit_partitions = 0

    def fold(train, test):
        nonlocal fit_partitions
        fit_partitions += 1
        allowed = _training_peers(matching_peer, train)
        x = np.vstack([own[train], peers[allowed]])
        y = np.r_[target[train], peer_y[allowed]]
        return predict(x, y, own[test], len(train), warning_counts)

    loo = _pivot_predictions(grouped_official_folds(own, indices), names, len(own), fold)
    returned = official["source_round"].to_numpy() > 0
    screen = []
    for name, values in loo.items():
        if not np.isfinite(values).all():
            raise ValueError("An official leave-one-out target is missing a prediction")
        screen.append(dict(method=name, error=_loss(target, values, metric),
                           returned_error=_loss(target[returned], values[returned], metric) if returned.any() else None,
                           rank_correlation=_ranking(target, values),
                           absolute_error_wins=int(np.sum(np.abs(values - target) < np.abs(loo[reference_name] - target)))))
    screen.sort(key=lambda row: row["error"])

    policies = ("Saved GP family", GP_SELECTION, MODEL_SELECTION, NESTED_BLEND)
    repeats = []
    for seed in seeds:
        out = {policy: np.full(len(own), np.nan) for policy in policies}
        details = []
        for number, (train, test) in enumerate(grouped_official_folds(own, indices, splits=5, seed=seed), 1):
            internal = _pivot_predictions(grouped_official_folds(own, train, splits=4, seed=seed + number),
                                           names, len(own), fold)
            gp, alternate, weight, selected = _selected_methods(
                {name: values[train] for name, values in internal.items()}, target[train],
                metric, gp_names, alternative_names)
            external = fold(train, test)
            current = dict(zip(policies, (external[reference_name], external[gp], external[selected],
                                           weight * external[gp] + (1 - weight) * external[alternate])))
            for policy, values in current.items():
                out[policy][test] = values
            detail = _partition_result(target, current, test, metric)
            details.append(dict(fold=number, test_indices=test.tolist(), selected_gp=gp,
                                selected_model=selected, blend_alternate=alternate, gp_weight=weight,
                                peer_rows_withheld=int(matching_peer[test].any(axis=0).sum()), results=detail))
        results = {}
        for policy, predictions in out.items():
            if not np.isfinite(predictions).all():
                raise ValueError("An outer fold did not score every official target")
            results[policy] = dict(error=_loss(target, predictions, metric),
                returned_error=_loss(target[returned], predictions[returned], metric) if returned.any() else None,
                mean_selection_gap=float(np.mean([d["results"][policy]["observed_gap"] for d in details])),
                picks_best=sum(d["results"][policy]["observed_gap"] == 0 for d in details),
                predictions=predictions.tolist())
        repeats.append(dict(seed=int(seed), folds=details, policies=results))
    return dict(metric=metric, reference_model=reference_name, fixed_screen=screen, nested=repeats,
                loo_predictions={name: values.tolist() for name, values in loo.items()},
                official_ids=official["observation_id"].tolist(), official_targets=target.tolist(),
                fit_partitions=fit_partitions, matching_peer_counts=matching_peer.sum(axis=1).tolist(),
                warnings={name: dict(counts) for name, counts in warning_counts.items() if counts})


def nested_summary(result):
    """Summarise split sensitivity while retaining each repeat in the run record."""
    repeats = result["nested"]
    rows = []
    for name in repeats[0]["policies"]:
        policy = [repeat["policies"][name] for repeat in repeats]
        rows.append(dict(policy=name, mean_error=np.mean([row["error"] for row in policy]),
                         mean_returned_error=np.mean([row["returned_error"] for row in policy])
                         if policy[0]["returned_error"] is not None else None,
                         mean_selection_gap=np.mean([row["mean_selection_gap"] for row in policy]),
                         best_picks=sum(row["picks_best"] for row in policy),
                         **{f"seed_{repeat['seed']}_error": row["error"]
                            for repeat, row in zip(repeats, policy, strict=True)}))
    return pd.DataFrame(rows)

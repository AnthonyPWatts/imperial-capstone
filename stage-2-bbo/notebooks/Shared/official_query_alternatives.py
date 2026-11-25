"""Refit supported official-only alternatives and compare their Week 9 queries."""

from collections import Counter, defaultdict

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import clone

from cohort_surrogate_study import (
    GP_MATERN, NESTED_BLEND, _capture, _fit_predict, _loss, _matern_gp,
    _partition_result, _regression_estimators, grouped_official_folds,
    nested_summary, predict_official_blend,
)
from prediction_review_helpers import make_candidate_review, review_gp_query


SAVED_GP = "Saved GP family"


def _outer_validation(inputs, target, returned, models, gp_models, reference, alternatives, metric):
    names = (SAVED_GP, *alternatives)
    repeats = []
    warnings = defaultdict(Counter)
    for seed in (42, 73):
        out = {name: np.full(len(inputs), np.nan) for name in names}
        details = []
        splits = grouped_official_folds(inputs, np.arange(len(inputs)), splits=5, seed=seed)
        for number, (train, test) in enumerate(splits, 1):
            x, y, query = inputs[train], target[train], inputs[test]
            for name in names:
                if name == NESTED_BLEND:
                    values, choice = predict_official_blend(
                        gp_models, reference, x, y, query, metric=metric, seed=seed + number)
                    for method, counts in choice["warnings"].items():
                        warnings[method].update(counts)
                else:
                    values = _capture(name, lambda: _fit_predict(models[name], x, y, query), warnings)
                out[name][test] = values
            details.append(_partition_result(target, {name: out[name][test] for name in names}, test, metric))
        policies = {}
        for name, predicted in out.items():
            if not np.isfinite(predicted).all():
                raise ValueError("Every official input must receive an outer-fold prediction")
            gaps = [fold[name]["observed_gap"] for fold in details]
            policies[name] = dict(error=_loss(target, predicted, metric),
                returned_error=_loss(target[returned], predicted[returned], metric) if returned.any() else None,
                mean_selection_gap=float(np.mean(gaps)), picks_best=sum(gap == 0 for gap in gaps))
        repeats.append(dict(seed=seed, policies=policies))
    rows = nested_summary({"nested": repeats}).to_dict(orient="records")
    return rows, {name: dict(counts) for name, counts in warnings.items() if counts}


def _archived_point(label, archive, weight):
    return dict(source=label, model=archive["model"], point=archive["point"],
                portal_input=archive["portal_input"], criterion=f"GP UCB ({weight:g})",
                predicted_target=archive["predicted_output"], latent_sd=archive["posterior_std"],
                acquisition_score=archive["ucb"])


def review_official_alternatives(inputs, target, returned, gp_models, official_gp, original, cohort,
                                 alternatives, *, columns, search, feature_sets, metric, minimum_distance,
                                 weight, weights):
    """Use official rows for estimation and search, then score archived comparison points."""
    reference = original["model"]
    if reference != official_gp["name"]:
        raise ValueError("The original proposal and official GP fit must use the same model")
    models = {SAVED_GP: gp_models[reference], **_regression_estimators(len(columns)),
              GP_MATERN: _matern_gp(gp_models[reference])}
    validation, validation_warnings = _outer_validation(inputs, target, returned, models, gp_models,
        reference, alternatives, metric)
    proposals = [_archived_point("Original My data only GP", original, weight)]
    if cohort is not None:
        proposals.append(_archived_point("Full-cohort GP", cohort, weight))
    archived = np.asarray([row["point"] for row in proposals])
    original_mean, original_sd = official_gp["model"].predict(archived[:1], return_std=True)
    np.testing.assert_allclose([original_mean[0], original_sd[0]], [original["predicted_output"],
        original["posterior_std"]], rtol=1e-10, atol=1e-9)

    anchor = inputs[int(np.argmax(target))]
    pool, _ = make_candidate_review(inputs, anchor, input_columns=columns, search_settings=search,
                                    minimum_distance=minimum_distance)
    candidates = pool[list(columns)].to_numpy(dtype=float)
    query = np.vstack([candidates, archived])
    precision = search["precision"]
    for name in alternatives:
        details = {}
        if name == GP_MATERN:
            fitted = clone(models[name]).fit(inputs, target)
            # Route the cloned kernel through the original feature subset and tie-break.
            search_result = review_gp_query(dict(name=reference, model=fitted), pool, inputs, target,
                input_columns=columns, weights=weights, weight=weight, feature_sets=feature_sets,
                precision=precision, minimum_distance=minimum_distance)
            best = search_result["proposal"]
            point, mean = best["point"], best["predicted_output"]
            sd, score, criterion = best["posterior_std"], best["ucb"], f"GP UCB ({weight:g})"
            reference_values = fitted.predict(archived).tolist()
            details["modelled_coordinates"] = [columns[i] for i in feature_sets.get(reference, range(len(columns)))]
        else:
            if name == NESTED_BLEND:
                predictions, details = predict_official_blend(
                    gp_models, reference, inputs, target, query, metric=metric, seed=42)
            else:
                predictions = _fit_predict(models[name], inputs, target, query)
            if not np.isfinite(predictions).all():
                raise ValueError("Expected finite alternative predictions")
            index = int(np.argmax(predictions[:len(candidates)]))
            point, mean = candidates[index].tolist(), float(predictions[index])
            sd, score, criterion = None, mean, "Predicted mean"
            reference_values = predictions[len(candidates):].tolist()
        proposals.append(dict(source="My data only alternative", model=name, point=point,
            portal_input="-".join(f"{value:.{precision}f}" for value in point), criterion=criterion,
            predicted_target=mean, latent_sd=sd, acquisition_score=score, details=details,
            predicted_target_at_original_gp=reference_values[0],
            predicted_target_at_full_cohort_gp=reference_values[1] if cohort is not None else None))

    all_points = np.asarray([row["point"] for row in proposals])
    gp_means, gp_sds = official_gp["model"].predict(all_points, return_std=True)
    nearest = cdist(all_points, inputs).min(axis=1)
    known = set(map(tuple, inputs.round(precision)))
    for index, row in enumerate(proposals):
        point = all_points[index]
        row.update(nearest_official_distance=float(nearest[index]),
            distance_to_original_gp=float(np.linalg.norm(point - archived[0])),
            distance_to_full_cohort_gp=float(np.linalg.norm(point - archived[1])) if cohort is not None else None,
            original_gp_mean=float(gp_means[index]), original_gp_latent_sd=float(gp_sds[index]),
            original_gp_ucb=float(gp_means[index] + weight * gp_sds[index]))
        if row["source"] == "My data only alternative" and (
            tuple(point) in known or nearest[index] < minimum_distance
            or not np.array_equal(point, point.round(precision))
            or not ((point >= 0) & (point <= 1 - 10.0 ** -precision)).all()
        ):
            raise ValueError("Alternative proposal violates the portal or spacing rules")
    return dict(candidate_count=len(pool), minimum_query_distance=minimum_distance,
                acquisition_weight=weight, validation=validation, validation_warnings=validation_warnings,
                proposals=proposals)

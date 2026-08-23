"""Confirm two stable low-weight residual voters beside the deep archive."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
from functools import partial as _partial
import time as _time
from typing import Callable as _Callable

import numpy as _np
import pandas as _pd

from data_partitioning import PartitionedData
from final_model import ordered_probabilities
from final_model import validate_probabilities
from geography_features import GEOGRAPHY_POLICIES
from geography_features import make_geography_preprocessor
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import median_selected_iterations
from model_evaluation import make_random_forest_pipeline
from target_encoding_features import TARGET_ENCODING_POLICIES
from target_encoding_features import make_target_encoding_preprocessor


DEEP_RESIDUAL_WEIGHTS = {
    "deep_archive": 0.925,
    "location_identity_blend": 0.025,
    "grid_10km_blend": 0.050,
}
COMPONENT_WEIGHTS = (0.55, 0.45)


@_dataclass(frozen=True)
class DeepResidualLocalConfirmation:
    """Residual component and complete recipe probabilities on local test."""

    probabilities: dict[str, _np.ndarray]
    component_seconds: _pd.Series
    selected_iterations: _pd.Series


def blend_deep_residual_candidate(
    components: dict[str, _np.ndarray],
) -> _np.ndarray:
    """Blend the fixed deep, location-identity and 10 km grid recipe."""

    missing = set(DEEP_RESIDUAL_WEIGHTS).difference(components)
    if missing:
        raise KeyError(f"Missing deep-residual components: {sorted(missing)!r}.")
    expected_rows = len(components["deep_archive"])
    for key in DEEP_RESIDUAL_WEIGHTS:
        validate_probabilities(components[key], expected_rows)
    probabilities = sum(
        DEEP_RESIDUAL_WEIGHTS[key] * components[key]
        for key in DEEP_RESIDUAL_WEIGHTS
    )
    validate_probabilities(probabilities, expected_rows)
    return probabilities


def fit_deep_residual_on_local_test(
    partitioned_data: PartitionedData,
    location_trial,
    grid_trial,
    deep_archive_probabilities: _np.ndarray,
    *,
    gpu_fitter: _Callable = fit_gpu_candidate_probabilities,
    random_forest_factory: _Callable[..., object] = make_random_forest_pipeline,
) -> DeepResidualLocalConfirmation:
    """Fit both residual ensembles on development and score local test."""

    for trial in (location_trial, grid_trial):
        for evaluation in (trial.xgboost, trial.random_forest):
            if evaluation.cross_validation_fingerprint != (
                partitioned_data.cross_validation_fingerprint
            ):
                raise ValueError("Residual confirmation uses different folds.")
    validate_probabilities(
        deep_archive_probabilities,
        len(partitioned_data.y_local_test),
    )
    if location_trial.policy != TARGET_ENCODING_POLICIES["location_identity"]:
        raise ValueError("Residual confirmation requires location target encoding.")
    if grid_trial.policy != GEOGRAPHY_POLICIES["grid_10km"]:
        raise ValueError("Residual confirmation requires the 10 km grid policy.")

    location = _fit_policy_blend(
        partitioned_data,
        location_trial.xgboost,
        preprocessor_factory=_partial(
            make_target_encoding_preprocessor,
            location_trial.policy,
        ),
        feature_policy=location_trial.policy.key,
        label="location identity",
        gpu_fitter=gpu_fitter,
        random_forest_factory=random_forest_factory,
    )
    grid = _fit_policy_blend(
        partitioned_data,
        grid_trial.xgboost,
        preprocessor_factory=_partial(
            make_geography_preprocessor,
            grid_trial.policy,
        ),
        feature_policy=grid_trial.policy.key,
        label="10 km grid",
        gpu_fitter=gpu_fitter,
        random_forest_factory=random_forest_factory,
    )
    components = {
        "deep_archive": deep_archive_probabilities,
        "location_identity_blend": location["blend"],
        "grid_10km_blend": grid["blend"],
    }
    candidate = blend_deep_residual_candidate(components)
    return DeepResidualLocalConfirmation(
        probabilities={
            **components,
            "deep_residual_candidate": candidate,
            "location_identity_xgboost": location["xgboost"],
            "location_identity_random_forest": location["random_forest"],
            "grid_10km_xgboost": grid["xgboost"],
            "grid_10km_random_forest": grid["random_forest"],
        },
        component_seconds=_pd.Series(
            {
                "location-identity XGBoost": location["xgboost_seconds"],
                "location-identity Random Forest": location["forest_seconds"],
                "10 km grid XGBoost": grid["xgboost_seconds"],
                "10 km grid Random Forest": grid["forest_seconds"],
            },
            name="fit and predict seconds",
        ),
        selected_iterations=_pd.Series(
            {
                "location-identity XGBoost": location["iterations"],
                "10 km grid XGBoost": grid["iterations"],
            },
            name="selected iterations",
            dtype="int64",
        ),
    )


def _fit_policy_blend(
    partitioned_data: PartitionedData,
    xgboost_evaluation,
    *,
    preprocessor_factory: _Callable[[], object],
    feature_policy: str,
    label: str,
    gpu_fitter: _Callable,
    random_forest_factory: _Callable[..., object],
) -> dict[str, object]:
    iterations = median_selected_iterations(xgboost_evaluation)
    spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name=f"XGBoost depth 8 child 1 [{label}]",
        feature_policy=feature_policy,
    )
    xgboost, xgboost_seconds = gpu_fitter(
        spec,
        partitioned_data.X_development,
        partitioned_data.y_development,
        partitioned_data.X_local_test,
        iterations=iterations,
        preprocessor_factory=preprocessor_factory,
    )
    forest = random_forest_factory(preprocessor_factory=preprocessor_factory)
    started = _time.perf_counter()
    forest.fit(partitioned_data.X_development, partitioned_data.y_development)
    random_forest = ordered_probabilities(
        forest,
        partitioned_data.X_local_test,
    )
    forest_seconds = _time.perf_counter() - started
    blend = COMPONENT_WEIGHTS[0] * xgboost + COMPONENT_WEIGHTS[1] * random_forest
    for probabilities in (xgboost, random_forest, blend):
        validate_probabilities(probabilities, len(partitioned_data.y_local_test))
    return {
        "xgboost": xgboost,
        "random_forest": random_forest,
        "blend": blend,
        "xgboost_seconds": xgboost_seconds,
        "forest_seconds": forest_seconds,
        "iterations": iterations,
    }

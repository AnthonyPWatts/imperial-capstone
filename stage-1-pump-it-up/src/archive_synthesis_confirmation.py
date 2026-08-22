"""Confirm the fixed archive-derived synthesis on the used local test."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
from functools import partial as _partial
import time as _time
from typing import Callable as _Callable

import numpy as _np
import pandas as _pd

from categorical_frequency_forest_evaluation import CategoricalFrequencyForestTrial
from categorical_frequency_forest_evaluation import (
    make_categorical_frequency_preprocessor,
)
from data_partitioning import PartitionedData
from final_model import ordered_probabilities
from final_model import validate_probabilities
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import median_selected_iterations
from model_evaluation import make_random_forest_pipeline
from spatial_height_imputation_evaluation import SpatialHeightTrial
from spatial_height_imputation_evaluation import make_spatial_height_preprocessor


ARCHIVE_SYNTHESIS_WEIGHTS = {
    "xgboost": 0.33,
    "spatial_xgboost": 0.11,
    "random_forest": 0.18,
    "frequency_random_forest": 0.09,
    "spatial_random_forest": 0.09,
    "identity_catboost": 0.20,
}


@_dataclass(frozen=True)
class ArchiveSynthesisLocalConfirmation:
    """New component and complete recipe probabilities on local test."""

    probabilities: dict[str, _np.ndarray]
    component_seconds: _pd.Series
    spatial_xgboost_iterations: int


def blend_archive_synthesis(
    components: dict[str, _np.ndarray],
) -> _np.ndarray:
    """Blend the six fixed components with the algebraic synthesis weights."""

    missing = set(ARCHIVE_SYNTHESIS_WEIGHTS).difference(components)
    if missing:
        raise KeyError(f"Missing archive-synthesis components: {sorted(missing)!r}.")
    expected_rows = len(components["xgboost"])
    for key in ARCHIVE_SYNTHESIS_WEIGHTS:
        validate_probabilities(components[key], expected_rows)
    probabilities = sum(
        ARCHIVE_SYNTHESIS_WEIGHTS[key] * components[key]
        for key in ARCHIVE_SYNTHESIS_WEIGHTS
    )
    validate_probabilities(probabilities, expected_rows)
    return probabilities


def fit_archive_synthesis_on_local_test(
    partitioned_data: PartitionedData,
    spatial_trial: SpatialHeightTrial,
    frequency_trial: CategoricalFrequencyForestTrial,
    cached_components: dict[str, _np.ndarray],
    *,
    gpu_fitter: _Callable = fit_gpu_candidate_probabilities,
    random_forest_factory: _Callable[..., object] = make_random_forest_pipeline,
) -> ArchiveSynthesisLocalConfirmation:
    """Fit the three uncached representation components on development."""

    for evaluation in (
        spatial_trial.xgboost,
        spatial_trial.random_forest,
        frequency_trial.random_forest,
    ):
        if evaluation.cross_validation_fingerprint != (
            partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Archive confirmation uses different frozen folds.")
    for key in ("xgboost", "random_forest", "identity_catboost"):
        if key not in cached_components:
            raise KeyError(f"Missing cached archive component: {key!r}.")
        validate_probabilities(
            cached_components[key],
            len(partitioned_data.y_local_test),
        )

    iterations = median_selected_iterations(spatial_trial.xgboost)
    spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [spatial height imputation]",
        feature_policy="ten-neighbour spatial GPS-height imputation",
    )
    spatial_xgboost, xgboost_seconds = gpu_fitter(
        spec,
        partitioned_data.X_development,
        partitioned_data.y_development,
        partitioned_data.X_local_test,
        iterations=iterations,
        preprocessor_factory=make_spatial_height_preprocessor,
    )
    spatial_forest, spatial_forest_seconds = _fit_forest(
        partitioned_data,
        _partial(
            random_forest_factory,
            preprocessor_factory=make_spatial_height_preprocessor,
        ),
    )
    frequency_forest, frequency_forest_seconds = _fit_forest(
        partitioned_data,
        _partial(
            random_forest_factory,
            preprocessor_factory=make_categorical_frequency_preprocessor,
        ),
    )
    components = {
        **cached_components,
        "spatial_xgboost": spatial_xgboost,
        "spatial_random_forest": spatial_forest,
        "frequency_random_forest": frequency_forest,
    }
    archive = blend_archive_synthesis(components)
    return ArchiveSynthesisLocalConfirmation(
        probabilities={**components, "archive_synthesis": archive},
        component_seconds=_pd.Series(
            {
                "spatial-height XGBoost": xgboost_seconds,
                "spatial-height Random Forest": spatial_forest_seconds,
                "frequency Random Forest": frequency_forest_seconds,
            },
            name="fit and predict seconds",
        ),
        spatial_xgboost_iterations=iterations,
    )


def _fit_forest(
    partitioned_data: PartitionedData,
    factory: _Callable[[], object],
) -> tuple[_np.ndarray, float]:
    started = _time.perf_counter()
    forest = factory()
    forest.fit(partitioned_data.X_development, partitioned_data.y_development)
    probabilities = ordered_probabilities(forest, partitioned_data.X_local_test)
    validate_probabilities(probabilities, len(partitioned_data.y_local_test))
    return probabilities, _time.perf_counter() - started

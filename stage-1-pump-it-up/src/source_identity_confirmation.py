"""Confirm source-plus-class recipes against the used local test."""

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
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import median_selected_iterations
from model_evaluation import make_random_forest_pipeline
from physical_hierarchy_evaluation import PhysicalHierarchyTrial
from physical_hierarchy_features import PHYSICAL_HIERARCHY_POLICIES
from physical_hierarchy_features import make_physical_hierarchy_preprocessor


SOURCE_PLUS_CLASS = PHYSICAL_HIERARCHY_POLICIES["source"]["source_plus_class"]
SOURCE_WEIGHTS = (0.55, 0.45)
SOURCE_IDENTITY_WEIGHTS = (0.44, 0.36, 0.20)


@_dataclass(frozen=True)
class SourceIdentityLocalConfirmation:
    """Source-policy component and recipe probabilities on local test."""

    probabilities: dict[str, _np.ndarray]
    component_seconds: _pd.Series
    selected_iterations: int


def blend_source_identity_recipes(
    components: dict[str, _np.ndarray],
) -> dict[str, _np.ndarray]:
    """Build the source baseline and fixed identity cross."""

    required = {
        "source_xgboost",
        "source_random_forest",
        "identity_catboost",
    }
    missing = required.difference(components)
    if missing:
        raise KeyError(f"Missing source/identity components: {sorted(missing)!r}.")
    expected_rows = len(components["source_xgboost"])
    for values in components.values():
        validate_probabilities(values, expected_rows)
    source = (
        SOURCE_WEIGHTS[0] * components["source_xgboost"]
        + SOURCE_WEIGHTS[1] * components["source_random_forest"]
    )
    source_identity = (
        SOURCE_IDENTITY_WEIGHTS[0] * components["source_xgboost"]
        + SOURCE_IDENTITY_WEIGHTS[1] * components["source_random_forest"]
        + SOURCE_IDENTITY_WEIGHTS[2] * components["identity_catboost"]
    )
    for values in (source, source_identity):
        validate_probabilities(values, expected_rows)
    return {
        "source_plus_class_55_45": source,
        "source_plus_class_identity_44_36_20": source_identity,
    }


def fit_source_identity_recipes_on_local_test(
    partitioned_data: PartitionedData,
    source_trial: PhysicalHierarchyTrial,
    identity_probabilities: _np.ndarray,
    *,
    gpu_fitter: _Callable = fit_gpu_candidate_probabilities,
    random_forest_factory: _Callable[[], object] | None = None,
) -> SourceIdentityLocalConfirmation:
    """Fit source-plus-class components on development and score local test."""

    for evaluation in (source_trial.xgboost, source_trial.random_forest):
        if evaluation.cross_validation_fingerprint != (
            partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Source local confirmation uses different frozen folds.")
    if source_trial.policy != SOURCE_PLUS_CLASS:
        raise ValueError("Source local confirmation requires source-plus-class.")
    validate_probabilities(identity_probabilities, len(partitioned_data.y_local_test))

    preprocessor_factory = _partial(
        make_physical_hierarchy_preprocessor,
        SOURCE_PLUS_CLASS,
    )
    iterations = median_selected_iterations(source_trial.xgboost)
    spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [source:source_plus_class]",
        feature_policy="source:source_plus_class",
    )
    xgboost_probabilities, xgboost_seconds = gpu_fitter(
        spec,
        partitioned_data.X_development,
        partitioned_data.y_development,
        partitioned_data.X_local_test,
        iterations=iterations,
        preprocessor_factory=preprocessor_factory,
    )
    forest_factory = random_forest_factory or _partial(
        make_random_forest_pipeline,
        preprocessor_factory=preprocessor_factory,
    )
    forest = forest_factory()
    forest_started = _time.perf_counter()
    forest.fit(partitioned_data.X_development, partitioned_data.y_development)
    forest_probabilities = ordered_probabilities(
        forest,
        partitioned_data.X_local_test,
    )
    forest_seconds = _time.perf_counter() - forest_started
    components = {
        "source_xgboost": xgboost_probabilities,
        "source_random_forest": forest_probabilities,
        "identity_catboost": identity_probabilities,
    }
    recipes = blend_source_identity_recipes(components)
    return SourceIdentityLocalConfirmation(
        probabilities={**components, **recipes},
        component_seconds=_pd.Series(
            {
                "source-plus-class XGBoost": xgboost_seconds,
                "source-plus-class Random Forest": forest_seconds,
            },
            name="fit and predict seconds",
        ),
        selected_iterations=iterations,
    )

"""Confirm the fixed deep-XGBoost archive substitution on the used local test."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
from typing import Callable as _Callable

import numpy as _np
import pandas as _pd

from data_partitioning import PartitionedData
from final_model import validate_probabilities
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from top_common_identity_evaluation import ARCHIVED_DEPTH_17_ITERATIONS
from top_common_identity_evaluation import make_top_common_identity_preprocessor


DEEP_XGBOOST_SEED = 20260822
DEEP_ARCHIVE_WEIGHTS = {
    "deep_xgboost": 0.33,
    "spatial_xgboost": 0.11,
    "random_forest": 0.18,
    "frequency_random_forest": 0.09,
    "spatial_random_forest": 0.09,
    "identity_catboost": 0.20,
}


@_dataclass(frozen=True)
class DeepArchiveLocalConfirmation:
    """Deep component and complete recipe probabilities on the local test."""

    probabilities: dict[str, _np.ndarray]
    component_seconds: _pd.Series


def blend_deep_archive(components: dict[str, _np.ndarray]) -> _np.ndarray:
    """Blend the unchanged deep-XGBoost archive-substitution recipe."""

    missing = set(DEEP_ARCHIVE_WEIGHTS).difference(components)
    if missing:
        raise KeyError(f"Missing deep-archive components: {sorted(missing)!r}.")
    expected_rows = len(components["deep_xgboost"])
    for key in DEEP_ARCHIVE_WEIGHTS:
        validate_probabilities(components[key], expected_rows)
    probabilities = sum(
        DEEP_ARCHIVE_WEIGHTS[key] * components[key]
        for key in DEEP_ARCHIVE_WEIGHTS
    )
    validate_probabilities(probabilities, expected_rows)
    return probabilities


def fit_deep_archive_on_local_test(
    partitioned_data: PartitionedData,
    cached_components: dict[str, _np.ndarray],
    *,
    gpu_fitter: _Callable = fit_gpu_candidate_probabilities,
) -> DeepArchiveLocalConfirmation:
    """Fit the fixed deep component on development and score the local test."""

    required = set(DEEP_ARCHIVE_WEIGHTS).difference({"deep_xgboost"})
    missing = required.difference(cached_components)
    if missing:
        raise KeyError(f"Missing cached archive components: {sorted(missing)!r}.")
    for key in required:
        validate_probabilities(
            cached_components[key],
            len(partitioned_data.y_local_test),
        )

    spec = _replace(
        make_xgboost_spec(
            variant="archived depth 17",
            seed=DEEP_XGBOOST_SEED,
        ),
        name=(
            "XGBoost archived depth 17 "
            f"[top-50 deferred identities; seed {DEEP_XGBOOST_SEED}]"
        ),
        feature_policy="accepted_plus_top_50_deferred_identities",
    )
    deep_xgboost, seconds = gpu_fitter(
        spec,
        partitioned_data.X_development,
        partitioned_data.y_development,
        partitioned_data.X_local_test,
        iterations=ARCHIVED_DEPTH_17_ITERATIONS,
        preprocessor_factory=make_top_common_identity_preprocessor,
    )
    components = {**cached_components, "deep_xgboost": deep_xgboost}
    candidate = blend_deep_archive(components)
    return DeepArchiveLocalConfirmation(
        probabilities={
            **components,
            "deep_archive_substitution": candidate,
        },
        component_seconds=_pd.Series(
            {"deep top-50 XGBoost": seconds},
            name="fit and predict seconds",
        ),
    )

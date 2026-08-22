"""Evaluate CatBoost ordered categoricals over deferred identities."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace

import pandas as _pd

from data_partitioning import PartitionedData
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import engineer_initial_features
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_catboost_spec
from model_evaluation import CandidateEvaluation
from target_encoding_features import normalise_identity


DEFERRED_IDENTITY_SOURCES = (
    "funder",
    "installer",
    "wpt_name",
    "subvillage",
    "ward",
    "scheme_name",
)
DEFERRED_IDENTITY_FEATURES = tuple(
    f"{feature}_identity" for feature in DEFERRED_IDENTITY_SOURCES
)


@_dataclass(frozen=True)
class CatBoostIdentityTrial:
    """One native-categorical candidate and its representation size."""

    evaluation: CandidateEvaluation
    engineered_features: int
    categorical_features: int


def engineer_complete_identity_catboost_features(
    X: _pd.DataFrame,
) -> tuple[_pd.DataFrame, tuple[str, ...]]:
    """Return accepted features plus all six normalised deferred identities."""

    engineered = engineer_initial_features(X).copy()
    for source, feature in zip(
        DEFERRED_IDENTITY_SOURCES,
        DEFERRED_IDENTITY_FEATURES,
        strict=True,
    ):
        engineered[feature] = normalise_identity(X[source]).astype(str)
    categorical = (*CATEGORICAL_FEATURES, *DEFERRED_IDENTITY_FEATURES)
    for feature in CATEGORICAL_FEATURES:
        engineered[feature] = engineered[feature].astype(str)
    if tuple(engineered.columns[-len(DEFERRED_IDENTITY_FEATURES) :]) != (
        DEFERRED_IDENTITY_FEATURES
    ):
        raise ValueError("Deferred CatBoost identity order changed.")
    return engineered, categorical


def evaluate_complete_identity_catboost(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> CatBoostIdentityTrial:
    """Evaluate one depth-8 CatBoost with native deferred identities."""

    spec = _replace(
        make_catboost_spec(variant="d8"),
        name="CatBoost d8 [complete deferred identities]",
        feature_policy="accepted plus six deferred identities",
    )
    evaluation = evaluate_gpu_candidate(
        spec,
        partitioned_data,
        cross_validation,
        catboost_feature_engineer=(
            engineer_complete_identity_catboost_features
        ),
    )
    first_training_positions, _ = next(cross_validation.split())
    engineered, categorical = engineer_complete_identity_catboost_features(
        partitioned_data.X_development.iloc[first_training_positions]
    )
    return CatBoostIdentityTrial(
        evaluation=evaluation,
        engineered_features=engineered.shape[1],
        categorical_features=len(categorical),
    )

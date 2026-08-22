"""Evaluate exact recording dates as native CatBoost survey batches."""

from __future__ import annotations

from dataclasses import replace as _replace

import pandas as _pd

from catboost_identity_evaluation import CatBoostIdentityTrial
from catboost_identity_evaluation import engineer_complete_identity_catboost_features
from data_partitioning import PartitionedData
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_catboost_spec


RECORDING_BATCH_FEATURE = "recording_batch_identity"


def engineer_recording_batch_catboost_features(
    X: _pd.DataFrame,
) -> tuple[_pd.DataFrame, tuple[str, ...]]:
    """Add the exact, normalised survey date to complete identities."""

    engineered, categorical = engineer_complete_identity_catboost_features(X)
    parsed = _pd.to_datetime(X["date_recorded"], errors="raise")
    batch = parsed.dt.strftime("%Y-%m-%d")
    engineered[RECORDING_BATCH_FEATURE] = batch.astype(str)
    if engineered.columns[-1] != RECORDING_BATCH_FEATURE:
        raise ValueError("Recording-batch CatBoost feature order changed.")
    return engineered, (*categorical, RECORDING_BATCH_FEATURE)


def evaluate_recording_batch_catboost(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> CatBoostIdentityTrial:
    """Evaluate one depth-8 complete-identity CatBoost with exact date."""

    spec = _replace(
        make_catboost_spec(variant="d8"),
        name="CatBoost d8 [complete identities plus recording batch]",
        feature_policy="accepted plus six identities and exact recording date",
    )
    evaluation = evaluate_gpu_candidate(
        spec,
        partitioned_data,
        cross_validation,
        catboost_feature_engineer=engineer_recording_batch_catboost_features,
    )
    first_training_positions, _ = next(cross_validation.split())
    engineered, categorical = engineer_recording_batch_catboost_features(
        partitioned_data.X_development.iloc[first_training_positions]
    )
    return CatBoostIdentityTrial(
        evaluation=evaluation,
        engineered_features=engineered.shape[1],
        categorical_features=len(categorical),
    )

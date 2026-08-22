"""Evaluate fixed quantity-by-physical native categorical interactions."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace

import pandas as _pd

from catboost_identity_evaluation import engineer_complete_identity_catboost_features
from data_partitioning import PartitionedData
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_catboost_spec
from model_evaluation import CandidateEvaluation
from target_encoding_features import normalise_identity


PHYSICAL_STATE_INTERACTION_SOURCES = (
    "extraction_type",
    "source",
    "waterpoint_type",
)
PHYSICAL_STATE_INTERACTION_FEATURES = tuple(
    f"quantity_{source}_interaction"
    for source in PHYSICAL_STATE_INTERACTION_SOURCES
)


@_dataclass(frozen=True)
class PhysicalStateInteractionTrial:
    """Complete-identity CatBoost with one physical-state layer."""

    catboost: CandidateEvaluation
    engineered_features: int
    categorical_features: int


def engineer_physical_state_interaction_catboost_features(
    X: _pd.DataFrame,
) -> tuple[_pd.DataFrame, tuple[str, ...]]:
    """Add the three fixed quantity-by-physical composites."""

    engineered, categorical = engineer_complete_identity_catboost_features(X)
    quantity = normalise_identity(X["quantity"])
    for source, feature in zip(
        PHYSICAL_STATE_INTERACTION_SOURCES,
        PHYSICAL_STATE_INTERACTION_FEATURES,
        strict=True,
    ):
        engineered[feature] = quantity.str.cat(
            normalise_identity(X[source]),
            sep="::",
        ).astype(str)
    if tuple(engineered.columns[-len(PHYSICAL_STATE_INTERACTION_FEATURES) :]) != (
        PHYSICAL_STATE_INTERACTION_FEATURES
    ):
        raise ValueError("Physical-state CatBoost interaction order changed.")
    return engineered, (*categorical, *PHYSICAL_STATE_INTERACTION_FEATURES)


def evaluate_physical_state_interaction_trial(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> PhysicalStateInteractionTrial:
    """Evaluate one depth-8 native-identity interaction candidate."""

    catboost_spec = _replace(
        make_catboost_spec(variant="d8"),
        name="CatBoost d8 [identities plus physical-state interactions]",
        feature_policy=(
            "accepted plus six identities and three quantity interactions"
        ),
    )
    evaluation = evaluate_gpu_candidate(
        catboost_spec,
        partitioned_data,
        cross_validation,
        catboost_feature_engineer=(
            engineer_physical_state_interaction_catboost_features
        ),
    )
    first_training_positions, _ = next(cross_validation.split())
    engineered, categorical = (
        engineer_physical_state_interaction_catboost_features(
            partitioned_data.X_development.iloc[first_training_positions]
        )
    )
    return PhysicalStateInteractionTrial(
        catboost=evaluation,
        engineered_features=engineered.shape[1],
        categorical_features=len(categorical),
    )

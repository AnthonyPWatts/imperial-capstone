"""Evaluate multi-resolution coordinate grids as native CatBoost categories."""

from __future__ import annotations

from dataclasses import replace as _replace

import numpy as _np
import pandas as _pd

from catboost_identity_evaluation import CatBoostIdentityTrial
from catboost_identity_evaluation import engineer_complete_identity_catboost_features
from data_partitioning import PartitionedData
from feature_engineering import TANZANIA_LATITUDE_BOUNDS
from feature_engineering import TANZANIA_LONGITUDE_BOUNDS
from feature_engineering import valid_tanzania_coordinates
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_catboost_spec
from model_evaluation import CandidateEvaluation


SPATIAL_GRID_MISSING_CATEGORY = "__missing__"
SPATIAL_GRID_POLICY_VERSION = 1
SPATIAL_GRID_GATE_ACCURACY = 0.001
SPATIAL_GRID_GATE_FOLD_WINS = 3
SPATIAL_GRID_GATE_WORST_FOLD = -0.0025
SPATIAL_GRID_FEATURES = (
    "coordinate_grid_0_05_degrees",
    "coordinate_grid_0_05_degrees_half_offset",
    "coordinate_grid_0_025_degrees",
    "coordinate_grid_0_025_degrees_half_offset",
)
_SPATIAL_GRID_POLICIES = (
    (SPATIAL_GRID_FEATURES[0], 0.05, 0.0),
    (SPATIAL_GRID_FEATURES[1], 0.05, 0.025),
    (SPATIAL_GRID_FEATURES[2], 0.025, 0.0),
    (SPATIAL_GRID_FEATURES[3], 0.025, 0.0125),
)


def spatial_grid_policy_signature() -> dict[str, object]:
    """Return the complete versioned contract for the deterministic grids."""

    return {
        "version": SPATIAL_GRID_POLICY_VERSION,
        "features": list(SPATIAL_GRID_FEATURES),
        "cells": [
            {
                "feature": feature,
                "degrees": degrees,
                "longitude_offset": offset,
                "latitude_offset": offset,
            }
            for feature, degrees, offset in _SPATIAL_GRID_POLICIES
        ],
        "binning": "floor((coordinate - offset) / degrees)",
        "missing_category": SPATIAL_GRID_MISSING_CATEGORY,
        "tanzania_longitude_bounds": list(TANZANIA_LONGITUDE_BOUNDS),
        "tanzania_latitude_bounds": list(TANZANIA_LATITUDE_BOUNDS),
    }


def engineer_spatial_grid_catboost_features(
    X: _pd.DataFrame,
) -> tuple[_pd.DataFrame, tuple[str, ...]]:
    """Append fixed regular and half-cell-offset coordinate grids."""

    engineered, categorical = engineer_complete_identity_catboost_features(X)
    longitude = _pd.to_numeric(X["longitude"], errors="coerce")
    latitude = _pd.to_numeric(X["latitude"], errors="coerce")
    valid = valid_tanzania_coordinates(longitude, latitude)
    for feature, degrees, offset in _SPATIAL_GRID_POLICIES:
        engineered[feature] = _coordinate_grid(
            longitude,
            latitude,
            valid,
            degrees=degrees,
            offset=offset,
        )
    if tuple(engineered.columns[-len(SPATIAL_GRID_FEATURES) :]) != (
        SPATIAL_GRID_FEATURES
    ):
        raise ValueError("Spatial-grid CatBoost feature order changed.")
    return engineered, (*categorical, *SPATIAL_GRID_FEATURES)


def evaluate_spatial_grid_catboost(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> CatBoostIdentityTrial:
    """Evaluate depth-8 complete-identity CatBoost with four fixed grids."""

    evaluation = evaluate_gpu_candidate(
        make_spatial_grid_catboost_spec(),
        partitioned_data,
        cross_validation,
        catboost_feature_engineer=engineer_spatial_grid_catboost_features,
    )
    first_training_positions, _ = next(cross_validation.split())
    engineered, categorical = engineer_spatial_grid_catboost_features(
        partitioned_data.X_development.iloc[first_training_positions]
    )
    return CatBoostIdentityTrial(
        evaluation=evaluation,
        engineered_features=engineered.shape[1],
        categorical_features=len(categorical),
    )


def make_spatial_grid_catboost_spec():
    """Return the unchanged depth-8 specification for the grid candidate."""

    return _replace(
        make_catboost_spec(variant="d8"),
        name="CatBoost d8 [complete identities plus spatial grids]",
        feature_policy=(
            "accepted plus six identities and four fixed coordinate grids"
        ),
    )


def summarise_spatial_grid_catboost_comparison(
    complete_identity: CandidateEvaluation,
    spatial_grid: CandidateEvaluation,
) -> tuple[_pd.DataFrame, _pd.DataFrame]:
    """Return the locked candidate summary and paired fold deltas."""

    if (
        complete_identity.cross_validation_fingerprint
        != spatial_grid.cross_validation_fingerprint
    ):
        raise ValueError("CatBoost candidates use different cross-validation folds.")
    identity_accuracy = complete_identity.fold_metrics["accuracy"]
    spatial_accuracy = spatial_grid.fold_metrics["accuracy"]
    if not identity_accuracy.index.equals(spatial_accuracy.index):
        raise ValueError("CatBoost candidate fold labels do not align.")
    if len(identity_accuracy) != 5:
        raise ValueError("Spatial-grid comparison requires exactly five folds.")
    values = _np.concatenate(
        [
            identity_accuracy.to_numpy(dtype="float64"),
            spatial_accuracy.to_numpy(dtype="float64"),
        ]
    )
    if not _np.isfinite(values).all():
        raise ValueError("CatBoost candidate fold accuracies must be finite.")

    fold_delta = spatial_accuracy - identity_accuracy
    accuracy_change = float(spatial_accuracy.mean() - identity_accuracy.mean())
    fold_wins = int(fold_delta.gt(0).sum())
    worst_fold_change = float(fold_delta.min())
    passes_gate = (
        _meets_minimum(accuracy_change, SPATIAL_GRID_GATE_ACCURACY)
        and fold_wins >= SPATIAL_GRID_GATE_FOLD_WINS
        and _meets_minimum(
            worst_fold_change,
            SPATIAL_GRID_GATE_WORST_FOLD,
        )
    )
    candidate_summary = _pd.DataFrame(
        [
            {
                "candidate": "complete_identity_catboost",
                "model_name": complete_identity.model_name,
                "mean_accuracy": float(identity_accuracy.mean()),
                "accuracy_change_vs_identity": 0.0,
                "fold_wins_vs_identity": 0,
                "worst_fold_change_vs_identity": 0.0,
                "passes_gate": False,
            },
            {
                "candidate": "spatial_grid_catboost",
                "model_name": spatial_grid.model_name,
                "mean_accuracy": float(spatial_accuracy.mean()),
                "accuracy_change_vs_identity": accuracy_change,
                "fold_wins_vs_identity": fold_wins,
                "worst_fold_change_vs_identity": worst_fold_change,
                "passes_gate": passes_gate,
            },
        ]
    ).set_index("candidate")
    fold_summary = _pd.DataFrame(
        {
            "complete_identity_accuracy": identity_accuracy,
            "spatial_grid_accuracy": spatial_accuracy,
            "spatial_grid_change": fold_delta,
            "spatial_grid_wins": fold_delta.gt(0),
        }
    )
    fold_summary.index.name = "validation_fold"
    return candidate_summary, fold_summary


def _meets_minimum(value: float, minimum: float) -> bool:
    return value >= minimum or bool(
        _np.isclose(value, minimum, rtol=0.0, atol=1e-12)
    )


def _coordinate_grid(
    longitude: _pd.Series,
    latitude: _pd.Series,
    valid: _pd.Series,
    *,
    degrees: float,
    offset: float,
) -> _pd.Series:
    result = _pd.Series(
        SPATIAL_GRID_MISSING_CATEGORY,
        index=longitude.index,
        dtype="string",
    )
    longitude_bin = _np.floor(
        (longitude.loc[valid] - offset) / degrees
    ).astype("int64")
    latitude_bin = _np.floor(
        (latitude.loc[valid] - offset) / degrees
    ).astype("int64")
    result.loc[valid] = longitude_bin.astype("string").str.cat(
        latitude_bin.astype("string"),
        sep="::",
    )
    return result.astype(str)

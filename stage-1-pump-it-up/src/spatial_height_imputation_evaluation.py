"""Evaluate fold-fitted spatial reconstruction of missing GPS height."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.neighbors import BallTree as _BallTree
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.preprocessing import StandardScaler as _StandardScaler
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from data_partitioning import PartitionedData
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from feature_engineering import engineer_initial_features
from feature_engineering import valid_tanzania_coordinates
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_preprocessing import RARE_CATEGORY_MINIMUM


SPATIAL_HEIGHT_NEIGHBOURS = 10
_DISTANCE_FLOOR_RADIANS = 1e-8


@_dataclass(frozen=True)
class SpatialHeightModel:
    """Training-partition measured heights and their coordinate index."""

    tree: _BallTree
    heights: _np.ndarray
    global_median: float


@_dataclass(frozen=True)
class SpatialHeightTrial:
    """Accepted components refitted with spatial height reconstruction."""

    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    engineered_features: int
    transformed_features_fold_1: int


class SpatialHeightFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Learn measured-height neighbours from the current fit partition."""

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "SpatialHeightFeatureEngineer":
        engineer_initial_features(X)
        self.spatial_height_model_ = fit_spatial_height_model(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(MODEL_FEATURES, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "spatial_height_model_")
        return engineer_spatial_height_features(X, self.spatial_height_model_)

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def fit_spatial_height_model(X: _pd.DataFrame) -> SpatialHeightModel:
    """Index non-zero measured heights with valid Tanzanian coordinates."""

    longitude = _pd.to_numeric(X["longitude"], errors="coerce")
    latitude = _pd.to_numeric(X["latitude"], errors="coerce")
    height = _pd.to_numeric(X["gps_height"], errors="coerce")
    measured_height = height.notna() & height.ne(0)
    indexed = measured_height & valid_tanzania_coordinates(longitude, latitude)
    if indexed.sum() < SPATIAL_HEIGHT_NEIGHBOURS:
        raise ValueError("Too few measured GPS heights for spatial imputation.")
    coordinates = _np.radians(
        _np.column_stack(
            [
                latitude.loc[indexed].to_numpy(dtype="float64"),
                longitude.loc[indexed].to_numpy(dtype="float64"),
            ]
        )
    )
    return SpatialHeightModel(
        tree=_BallTree(coordinates, metric="haversine"),
        heights=height.loc[indexed].to_numpy(dtype="float64"),
        global_median=float(height.loc[measured_height].median()),
    )


def engineer_spatial_height_features(
    X: _pd.DataFrame,
    model: SpatialHeightModel,
) -> _pd.DataFrame:
    """Fill missing height from distance-weighted measured neighbours."""

    engineered = engineer_initial_features(X)
    missing = engineered["gps_height"].isna()
    if not missing.any():
        return engineered

    longitude = _pd.to_numeric(X["longitude"], errors="coerce")
    latitude = _pd.to_numeric(X["latitude"], errors="coerce")
    spatially_valid = missing & valid_tanzania_coordinates(longitude, latitude)
    engineered.loc[missing, "gps_height"] = model.global_median
    if spatially_valid.any():
        coordinates = _np.radians(
            _np.column_stack(
                [
                    latitude.loc[spatially_valid].to_numpy(dtype="float64"),
                    longitude.loc[spatially_valid].to_numpy(dtype="float64"),
                ]
            )
        )
        neighbour_count = min(SPATIAL_HEIGHT_NEIGHBOURS, len(model.heights))
        distances, indices = model.tree.query(
            coordinates,
            k=neighbour_count,
            return_distance=True,
        )
        weights = 1.0 / _np.maximum(distances, _DISTANCE_FLOOR_RADIANS)
        predictions = (
            (model.heights[indices] * weights).sum(axis=1)
            / weights.sum(axis=1)
        )
        engineered.loc[spatially_valid, "gps_height"] = predictions
    if engineered["gps_height"].isna().any():
        raise ValueError("Spatial height imputation left missing values.")
    return engineered.loc[:, MODEL_FEATURES]


def make_spatial_height_preprocessor(
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build accepted preprocessing with fold-fitted spatial height."""

    numeric_steps: list[tuple[str, object]] = [
        (
            "median_imputation",
            _SimpleImputer(strategy="median", add_indicator=True),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("standardisation", _StandardScaler()))
    columns = _ColumnTransformer(
        transformers=[
            (
                "numeric",
                _Pipeline(steps=numeric_steps),
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                _OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=RARE_CATEGORY_MINIMUM,
                    sparse_output=sparse_output,
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", SpatialHeightFeatureEngineer()),
            ("column_preprocessing", columns),
        ]
    )


def evaluate_spatial_height_trial(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> SpatialHeightTrial:
    """Evaluate accepted XGBoost and forest with spatial height imputation."""

    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [spatial height imputation]",
        feature_policy="ten-neighbour spatial GPS-height imputation",
    )
    xgboost = evaluate_gpu_candidate(
        xgboost_spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_spatial_height_preprocessor,
    )
    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_spatial_height_preprocessor,
        model_name="Random Forest [spatial height imputation]",
    )
    training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[training_positions]
    preprocessor = make_spatial_height_preprocessor()
    transformed = preprocessor.fit_transform(
        X_training,
        partitioned_data.y_development.iloc[training_positions],
    )
    return SpatialHeightTrial(
        xgboost=xgboost,
        random_forest=random_forest,
        engineered_features=len(MODEL_FEATURES),
        transformed_features_fold_1=transformed.shape[1],
    )

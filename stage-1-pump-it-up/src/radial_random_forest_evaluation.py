"""Evaluate radial distance in the accepted Random Forest representation."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.preprocessing import StandardScaler as _StandardScaler
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from data_partitioning import PartitionedData
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from feature_engineering import engineer_initial_features
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_preprocessing import RARE_CATEGORY_MINIMUM
from radial_distance_features import RADIAL_DISTANCE_FEATURE
from radial_distance_features import geodesic_origin_distance_km


RADIAL_RANDOM_FOREST_MODEL_FEATURES = (*MODEL_FEATURES, RADIAL_DISTANCE_FEATURE)
RADIAL_RANDOM_FOREST_NUMERIC_FEATURES = (*NUMERIC_FEATURES, RADIAL_DISTANCE_FEATURE)


@_dataclass(frozen=True)
class RadialRandomForestTrial:
    """Accepted Random Forest with one deterministic radial feature."""

    random_forest: CandidateEvaluation
    engineered_features: int
    transformed_features_fold_1: int


class RadialRandomForestFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Append great-circle origin distance to accepted features."""

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "RadialRandomForestFeatureEngineer":
        engineered = engineer_radial_random_forest_features(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(engineered.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "feature_names_out_")
        engineered = engineer_radial_random_forest_features(X)
        if tuple(engineered.columns) != tuple(self.feature_names_out_):
            raise ValueError("Radial Random Forest columns changed after fitting.")
        return engineered

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def engineer_radial_random_forest_features(X: _pd.DataFrame) -> _pd.DataFrame:
    """Return accepted features plus one geodesic origin distance."""

    engineered = engineer_initial_features(X)
    engineered[RADIAL_DISTANCE_FEATURE] = geodesic_origin_distance_km(X)
    if tuple(engineered.columns) != RADIAL_RANDOM_FOREST_MODEL_FEATURES:
        raise ValueError("Radial Random Forest feature order changed.")
    return engineered


def make_radial_random_forest_preprocessor(
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build accepted one-hot preprocessing with radial distance."""

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
                list(RADIAL_RANDOM_FOREST_NUMERIC_FEATURES),
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
            ("feature_engineering", RadialRandomForestFeatureEngineer()),
            ("column_preprocessing", columns),
        ]
    )


def evaluate_radial_random_forest(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> RadialRandomForestTrial:
    """Evaluate one accepted-spec forest with radial distance."""

    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_radial_random_forest_preprocessor,
        model_name="Random Forest [accepted plus origin distance]",
    )
    training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[training_positions]
    preprocessor = make_radial_random_forest_preprocessor()
    transformed = preprocessor.fit_transform(
        X_training,
        partitioned_data.y_development.iloc[training_positions],
    )
    return RadialRandomForestTrial(
        random_forest=random_forest,
        engineered_features=len(RADIAL_RANDOM_FOREST_MODEL_FEATURES),
        transformed_features_fold_1=transformed.shape[1],
    )

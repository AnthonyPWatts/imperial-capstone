"""Evaluate explicit recording-time categories beside elapsed days."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace

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
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_preprocessing import RARE_CATEGORY_MINIMUM


TEMPORAL_CATEGORICAL_FEATURES = (
    "recorded_year",
    "recorded_month",
    "recorded_year_month",
)
TEMPORAL_MODEL_FEATURES = (*MODEL_FEATURES, *TEMPORAL_CATEGORICAL_FEATURES)
TEMPORAL_ALL_CATEGORICAL_FEATURES = (
    *CATEGORICAL_FEATURES,
    *TEMPORAL_CATEGORICAL_FEATURES,
)


@_dataclass(frozen=True)
class TemporalFeatureTrial:
    """XGBoost and Random Forest evidence for the one temporal policy."""

    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    engineered_features: int
    transformed_features_fold_1: int


class TemporalFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Add explicit recording-time categories to the accepted feature frame."""

    def fit(self, X: _pd.DataFrame, y: object = None) -> "TemporalFeatureEngineer":
        engineered = engineer_temporal_features(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(engineered.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "feature_names_out_")
        engineered = engineer_temporal_features(X)
        if tuple(engineered.columns) != tuple(self.feature_names_out_):
            raise ValueError("Temporal feature columns changed after fitting.")
        return engineered

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def engineer_temporal_features(X: _pd.DataFrame) -> _pd.DataFrame:
    """Return accepted features plus year, month and year-month categories."""

    engineered = engineer_initial_features(X)
    recorded = _pd.to_datetime(X["date_recorded"], errors="raise")
    if recorded.isna().any():
        raise ValueError("Recording-time categories require complete dates.")
    engineered["recorded_year"] = recorded.dt.strftime("%Y")
    engineered["recorded_month"] = recorded.dt.strftime("%m")
    engineered["recorded_year_month"] = recorded.dt.strftime("%Y-%m")
    if tuple(engineered.columns) != TEMPORAL_MODEL_FEATURES:
        raise ValueError("Temporal feature order changed.")
    return engineered


def make_temporal_preprocessor(
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build fold-fitted preprocessing for the one temporal policy."""

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
                list(TEMPORAL_ALL_CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", TemporalFeatureEngineer()),
            ("column_preprocessing", columns),
        ]
    )


def evaluate_temporal_feature_trial(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> TemporalFeatureTrial:
    """Evaluate accepted XGBoost and Random Forest with temporal categories."""

    preprocessor_factory = make_temporal_preprocessor
    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [explicit recording time]",
        feature_policy="accepted plus recording year, month and year-month",
    )
    xgboost = evaluate_gpu_candidate(
        xgboost_spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=preprocessor_factory,
    )
    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=preprocessor_factory,
        model_name="Random Forest [explicit recording time]",
    )
    training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[training_positions]
    preprocessor = preprocessor_factory()
    transformed = preprocessor.fit_transform(
        X_training,
        partitioned_data.y_development.iloc[training_positions],
    )
    return TemporalFeatureTrial(
        xgboost=xgboost,
        random_forest=random_forest,
        engineered_features=len(TEMPORAL_MODEL_FEATURES),
        transformed_features_fold_1=transformed.shape[1],
    )

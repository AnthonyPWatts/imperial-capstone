"""Evaluate one shared region-by-physical interaction layer."""

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

from catboost_identity_evaluation import CatBoostIdentityTrial
from catboost_identity_evaluation import engineer_complete_identity_catboost_features
from data_partitioning import PartitionedData
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from feature_engineering import engineer_initial_features
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_catboost_spec
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_preprocessing import RARE_CATEGORY_MINIMUM
from target_encoding_features import normalise_identity


REGIONAL_INTERACTION_SOURCES = (
    "extraction_type",
    "source",
    "waterpoint_type",
)
REGIONAL_INTERACTION_FEATURES = tuple(
    f"region_{source}_interaction" for source in REGIONAL_INTERACTION_SOURCES
)
REGIONAL_MODEL_FEATURES = (*MODEL_FEATURES, *REGIONAL_INTERACTION_FEATURES)
REGIONAL_CATEGORICAL_FEATURES = (
    *CATEGORICAL_FEATURES,
    *REGIONAL_INTERACTION_FEATURES,
)


@_dataclass(frozen=True)
class RegionalInteractionTrial:
    """All promoted component families with one shared interaction policy."""

    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    catboost: CandidateEvaluation
    engineered_features: int
    transformed_features_fold_1: int
    categorical_features: int


class RegionalInteractionFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Add three deterministic region-by-physical category composites."""

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "RegionalInteractionFeatureEngineer":
        engineered = engineer_regional_interaction_features(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(engineered.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "feature_names_out_")
        engineered = engineer_regional_interaction_features(X)
        if tuple(engineered.columns) != tuple(self.feature_names_out_):
            raise ValueError("Regional interaction columns changed after fitting.")
        return engineered

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def engineer_regional_interaction_features(X: _pd.DataFrame) -> _pd.DataFrame:
    """Return accepted features plus the three fixed regional composites."""

    engineered = engineer_initial_features(X)
    _append_regional_interactions(engineered, X)
    if tuple(engineered.columns) != REGIONAL_MODEL_FEATURES:
        raise ValueError("Regional interaction feature order changed.")
    return engineered


def engineer_regional_interaction_catboost_features(
    X: _pd.DataFrame,
) -> tuple[_pd.DataFrame, tuple[str, ...]]:
    """Add the same shared interaction layer to complete native identities."""

    engineered, categorical = engineer_complete_identity_catboost_features(X)
    _append_regional_interactions(engineered, X)
    if tuple(engineered.columns[-len(REGIONAL_INTERACTION_FEATURES) :]) != (
        REGIONAL_INTERACTION_FEATURES
    ):
        raise ValueError("Regional CatBoost interaction order changed.")
    return engineered, (*categorical, *REGIONAL_INTERACTION_FEATURES)


def make_regional_interaction_preprocessor(
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build accepted preprocessing with the shared regional layer."""

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
                list(REGIONAL_CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", RegionalInteractionFeatureEngineer()),
            ("column_preprocessing", columns),
        ]
    )


def evaluate_regional_interaction_trial(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> RegionalInteractionTrial:
    """Evaluate all three promoted families with the one shared layer."""

    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [regional physical interactions]",
        feature_policy="accepted plus three region-by-physical interactions",
    )
    xgboost = evaluate_gpu_candidate(
        xgboost_spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_regional_interaction_preprocessor,
    )
    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_regional_interaction_preprocessor,
        model_name="Random Forest [regional physical interactions]",
    )
    catboost_spec = _replace(
        make_catboost_spec(variant="d8"),
        name="CatBoost d8 [identities plus regional physical interactions]",
        feature_policy=(
            "accepted plus six identities and three regional interactions"
        ),
    )
    catboost_trial = evaluate_gpu_candidate(
        catboost_spec,
        partitioned_data,
        cross_validation,
        catboost_feature_engineer=engineer_regional_interaction_catboost_features,
    )
    first_training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[first_training_positions]
    preprocessor = make_regional_interaction_preprocessor()
    transformed = preprocessor.fit_transform(
        X_training,
        partitioned_data.y_development.iloc[first_training_positions],
    )
    cat_engineered, categorical = engineer_regional_interaction_catboost_features(
        X_training
    )
    if cat_engineered.shape[1] != (
        len(MODEL_FEATURES)
        + len(REGIONAL_INTERACTION_FEATURES)
        + 6
    ):
        raise ValueError("Unexpected regional identity CatBoost feature count.")
    return RegionalInteractionTrial(
        xgboost=xgboost,
        random_forest=random_forest,
        catboost=catboost_trial,
        engineered_features=len(REGIONAL_MODEL_FEATURES),
        transformed_features_fold_1=transformed.shape[1],
        categorical_features=len(categorical),
    )


def _append_regional_interactions(
    destination: _pd.DataFrame,
    X: _pd.DataFrame,
) -> None:
    region = normalise_identity(X["region"])
    for source, feature in zip(
        REGIONAL_INTERACTION_SOURCES,
        REGIONAL_INTERACTION_FEATURES,
        strict=True,
    ):
        destination[feature] = region.str.cat(
            normalise_identity(X[source]),
            sep="::",
        ).astype(str)

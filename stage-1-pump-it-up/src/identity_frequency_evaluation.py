"""Evaluate fold-fitted occurrence frequencies for all deferred identities."""

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
from target_encoding_features import normalise_identity


IDENTITY_SOURCES = (
    "funder",
    "installer",
    "wpt_name",
    "subvillage",
    "ward",
    "scheme_name",
)
IDENTITY_FREQUENCY_FEATURES = tuple(
    f"{source}_log_frequency" for source in IDENTITY_SOURCES
)
IDENTITY_FREQUENCY_MODEL_FEATURES = (
    *MODEL_FEATURES,
    *IDENTITY_FREQUENCY_FEATURES,
)
IDENTITY_FREQUENCY_NUMERIC_FEATURES = (
    *NUMERIC_FEATURES,
    *IDENTITY_FREQUENCY_FEATURES,
)


@_dataclass(frozen=True)
class IdentityFrequencyTrial:
    """XGBoost and Random Forest evidence for all identity frequencies."""

    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    engineered_features: int
    transformed_features_fold_1: int


class IdentityFrequencyFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Learn identity-count mappings only from the current fit partition."""

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "IdentityFrequencyFeatureEngineer":
        engineer_initial_features(X)
        self.frequency_maps_ = {
            source: normalise_identity(X[source]).value_counts().to_dict()
            for source in IDENTITY_SOURCES
        }
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(
            IDENTITY_FREQUENCY_MODEL_FEATURES,
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "frequency_maps_")
        return engineer_identity_frequency_features(X, self.frequency_maps_)

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def engineer_identity_frequency_features(
    X: _pd.DataFrame,
    frequency_maps: dict[str, dict[str, int]],
) -> _pd.DataFrame:
    """Return accepted features plus fold-fitted log occurrence counts."""

    if set(frequency_maps) != set(IDENTITY_SOURCES):
        raise ValueError("Identity frequency mappings do not match the policy.")
    engineered = engineer_initial_features(X)
    for source, feature in zip(
        IDENTITY_SOURCES,
        IDENTITY_FREQUENCY_FEATURES,
        strict=True,
    ):
        counts = normalise_identity(X[source]).map(frequency_maps[source]).fillna(0)
        engineered[feature] = _np.log1p(counts.astype("float64"))
    if tuple(engineered.columns) != IDENTITY_FREQUENCY_MODEL_FEATURES:
        raise ValueError("Identity frequency feature order changed.")
    return engineered


def make_identity_frequency_preprocessor(
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build fold-fitted preprocessing for all six identity frequencies."""

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
                list(IDENTITY_FREQUENCY_NUMERIC_FEATURES),
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
            ("feature_engineering", IdentityFrequencyFeatureEngineer()),
            ("column_preprocessing", columns),
        ]
    )


def evaluate_identity_frequency_trial(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> IdentityFrequencyTrial:
    """Evaluate accepted tree specifications with all identity frequencies."""

    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [all identity frequencies]",
        feature_policy="accepted plus six fold-fitted identity frequencies",
    )
    xgboost = evaluate_gpu_candidate(
        xgboost_spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_identity_frequency_preprocessor,
    )
    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_identity_frequency_preprocessor,
        model_name="Random Forest [all identity frequencies]",
    )
    training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[training_positions]
    preprocessor = make_identity_frequency_preprocessor()
    transformed = preprocessor.fit_transform(
        X_training,
        partitioned_data.y_development.iloc[training_positions],
    )
    return IdentityFrequencyTrial(
        xgboost=xgboost,
        random_forest=random_forest,
        engineered_features=len(IDENTITY_FREQUENCY_MODEL_FEATURES),
        transformed_features_fold_1=transformed.shape[1],
    )

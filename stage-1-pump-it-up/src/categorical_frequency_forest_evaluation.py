"""Evaluate a frequency-only categorical representation with Random Forest."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.preprocessing import StandardScaler as _StandardScaler
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from data_partitioning import PartitionedData
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from feature_engineering import DEFERRED_HIERARCHY_FEATURES
from feature_engineering import NUMERIC_FEATURES
from feature_engineering import engineer_initial_features
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from target_encoding_features import normalise_identity


# The official archived 0.8265 solution counted every factor level and replaced
# support below five with one shared sentinel before fitting a random forest.
# This policy reproduces that representation without using validation features.
FREQUENCY_MINIMUM_SUPPORT = 5
FREQUENCY_SOURCE_FEATURES = (
    "date_recorded",
    *CATEGORICAL_FEATURES,
    *DEFERRED_HIGH_CARDINALITY_FEATURES,
    *DEFERRED_HIERARCHY_FEATURES,
)
FREQUENCY_FEATURES = tuple(
    f"{source}_occurrence_count" for source in FREQUENCY_SOURCE_FEATURES
)
FREQUENCY_MODEL_FEATURES = (*NUMERIC_FEATURES, *FREQUENCY_FEATURES)


@_dataclass(frozen=True)
class CategoricalFrequencyForestTrial:
    """Cross-validated evidence for the frequency-only forest."""

    random_forest: CandidateEvaluation
    engineered_features: int
    transformed_features_fold_1: int


class CategoricalFrequencyFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Replace every categorical source value with fold-fitted support."""

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "CategoricalFrequencyFeatureEngineer":
        engineer_initial_features(X)
        self.frequency_maps_ = {
            source: normalise_identity(X[source]).value_counts().to_dict()
            for source in FREQUENCY_SOURCE_FEATURES
        }
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(
            FREQUENCY_MODEL_FEATURES,
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "frequency_maps_")
        engineered = engineer_initial_features(X).loc[:, list(NUMERIC_FEATURES)]
        for source, feature in zip(
            FREQUENCY_SOURCE_FEATURES,
            FREQUENCY_FEATURES,
            strict=True,
        ):
            counts = (
                normalise_identity(X[source])
                .map(self.frequency_maps_[source])
                .fillna(0)
                .astype("int64")
            )
            engineered[feature] = counts.where(
                counts.ge(FREQUENCY_MINIMUM_SUPPORT),
                -1,
            )
        if tuple(engineered.columns) != FREQUENCY_MODEL_FEATURES:
            raise ValueError("Categorical frequency feature order changed.")
        return engineered

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def make_categorical_frequency_preprocessor(
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build numeric preprocessing for the frequency-only representation."""

    del sparse_output
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
                list(FREQUENCY_MODEL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            (
                "feature_engineering",
                CategoricalFrequencyFeatureEngineer(),
            ),
            ("column_preprocessing", columns),
        ]
    )


def evaluate_categorical_frequency_forest(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> CategoricalFrequencyForestTrial:
    """Evaluate one accepted-spec forest using only categorical support."""

    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_categorical_frequency_preprocessor,
        model_name="Random Forest [all categorical occurrence counts]",
    )
    training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[training_positions]
    preprocessor = make_categorical_frequency_preprocessor()
    transformed = preprocessor.fit_transform(
        X_training,
        partitioned_data.y_development.iloc[training_positions],
    )
    return CategoricalFrequencyForestTrial(
        random_forest=random_forest,
        engineered_features=len(FREQUENCY_MODEL_FEATURES),
        transformed_features_fold_1=transformed.shape[1],
    )

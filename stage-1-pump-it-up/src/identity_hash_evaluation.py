"""Evaluate target-free hashing for all six deferred identity fields."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.feature_extraction import FeatureHasher as _FeatureHasher
from sklearn.pipeline import FeatureUnion as _FeatureUnion
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from data_partitioning import PartitionedData
from feature_engineering import engineer_initial_features
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from identity_frequency_evaluation import IDENTITY_SOURCES
from model_evaluation import CandidateEvaluation
from model_preprocessing import make_initial_preprocessor
from target_encoding_features import normalise_identity


IDENTITY_HASH_FEATURES = 4096


@_dataclass(frozen=True)
class IdentityHashTrial:
    """XGBoost evidence for the one fixed identity hash policy."""

    xgboost: CandidateEvaluation
    hash_features: int
    transformed_features_fold_1: int
    maximum_hash_nonzeros_per_row: int


class IdentityTokenExtractor(_BaseEstimator, _TransformerMixin):
    """Return six field-prefixed normalised tokens for each source row."""

    def fit(self, X: _pd.DataFrame, y: object = None) -> "IdentityTokenExtractor":
        engineer_initial_features(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.identity_sources_ = IDENTITY_SOURCES
        return self

    def transform(self, X: _pd.DataFrame) -> list[list[str]]:
        _check_is_fitted(self, "identity_sources_")
        engineer_initial_features(X)
        columns = [
            (
                source
                + "="
                + normalise_identity(X[source]).astype(str)
            ).to_numpy()
            for source in self.identity_sources_
        ]
        return [list(tokens) for tokens in zip(*columns, strict=True)]


def make_identity_hash_transformer() -> _Pipeline:
    """Build the fixed target-free shared identity hash block."""

    return _Pipeline(
        steps=[
            ("identity_tokens", IdentityTokenExtractor()),
            (
                "identity_hash",
                _FeatureHasher(
                    n_features=IDENTITY_HASH_FEATURES,
                    input_type="string",
                    alternate_sign=False,
                ),
            ),
        ]
    )


def make_identity_hash_preprocessor() -> _FeatureUnion:
    """Append the hash block to accepted sparse preprocessing."""

    return _FeatureUnion(
        transformer_list=[
            ("accepted", make_initial_preprocessor()),
            ("identity_hash", make_identity_hash_transformer()),
        ],
        n_jobs=1,
    )


def evaluate_identity_hash_trial(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> IdentityHashTrial:
    """Evaluate accepted XGBoost with one fixed shared identity hash space."""

    spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [4096 identity hashes]",
        feature_policy="accepted plus six identities in one 4096-column hash",
    )
    xgboost = evaluate_gpu_candidate(
        spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_identity_hash_preprocessor,
    )
    training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[training_positions]
    transformed = make_identity_hash_preprocessor().fit_transform(
        X_training,
        partitioned_data.y_development.iloc[training_positions],
    )
    hash_block = make_identity_hash_transformer().fit_transform(X_training)
    return IdentityHashTrial(
        xgboost=xgboost,
        hash_features=IDENTITY_HASH_FEATURES,
        transformed_features_fold_1=transformed.shape[1],
        maximum_hash_nonzeros_per_row=int(hash_block.getnnz(axis=1).max()),
    )

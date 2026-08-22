"""Evaluate compact top-common identity indicators with accepted XGBoost."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace

import numpy as _np
import pandas as _pd
from scipy import sparse as _sparse
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.pipeline import FeatureUnion as _FeatureUnion
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from data_partitioning import PartitionedData
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from feature_engineering import EXPECTED_SOURCE_FEATURES
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_preprocessing import make_initial_preprocessor


TOP_COMMON_VALUES = 50
MISSING_IDENTITY = "__missing__"


@_dataclass(frozen=True)
class TopCommonIdentityTrial:
    """OOF evidence for one fixed common-identity representation."""

    xgboost: CandidateEvaluation
    indicator_features: int
    transformed_features_fold_1: int
    training_coverage_fold_1: float


class TopCommonIdentityEncoder(_BaseEstimator, _TransformerMixin):
    """Learn fixed-width indicators for each field's most common values."""

    def __init__(self, *, top_k: int = TOP_COMMON_VALUES):
        self.top_k = top_k

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "TopCommonIdentityEncoder":
        _validate_source_frame(X)
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        selected_values: dict[str, tuple[str, ...]] = {}
        feature_names = []
        for feature in DEFERRED_HIGH_CARDINALITY_FEATURES:
            values = _identity_values(X[feature])
            counts = (
                values.value_counts(dropna=False)
                .rename_axis("value")
                .reset_index(name="count")
            )
            counts["sort_value"] = counts["value"].astype(str)
            counts = counts.sort_values(
                ["count", "sort_value"],
                ascending=[False, True],
                kind="stable",
            )
            selected = tuple(counts["value"].iloc[: self.top_k])
            selected_values[feature] = selected
            feature_names.extend(
                f"{feature}__top_{position:02d}"
                for position in range(1, len(selected) + 1)
            )
        self.selected_values_ = selected_values
        self.value_positions_ = {
            feature: {value: position for position, value in enumerate(values)}
            for feature, values in selected_values.items()
        }
        self.feature_offsets_ = {
            feature: sum(
                len(selected_values[prior])
                for prior in DEFERRED_HIGH_CARDINALITY_FEATURES[:feature_position]
            )
            for feature_position, feature in enumerate(
                DEFERRED_HIGH_CARDINALITY_FEATURES
            )
        }
        self.feature_names_out_ = _np.asarray(feature_names, dtype=object)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _sparse.csr_matrix:
        _check_is_fitted(
            self,
            ("selected_values_", "value_positions_", "feature_offsets_"),
        )
        _validate_source_frame(X)
        row_positions: list[int] = []
        column_positions: list[int] = []
        for feature in DEFERRED_HIGH_CARDINALITY_FEATURES:
            mapped = _identity_values(X[feature]).map(self.value_positions_[feature])
            present = mapped.notna().to_numpy()
            rows = _np.flatnonzero(present)
            columns = mapped.iloc[rows].to_numpy(dtype="int64")
            row_positions.extend(rows.tolist())
            column_positions.extend(
                (columns + self.feature_offsets_[feature]).tolist()
            )
        data = _np.ones(len(row_positions), dtype="float32")
        return _sparse.csr_matrix(
            (data, (row_positions, column_positions)),
            shape=(len(X), len(self.feature_names_out_)),
            dtype="float32",
        )

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def make_top_common_identity_preprocessor() -> _FeatureUnion:
    """Combine accepted preprocessing with six fixed-width identity blocks."""

    return _FeatureUnion(
        transformer_list=[
            ("accepted", make_initial_preprocessor()),
            ("top_common_identity", TopCommonIdentityEncoder()),
        ]
    )


def evaluate_top_common_identity_xgboost(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> TopCommonIdentityTrial:
    """Evaluate one archive-derived top-50 identity XGBoost candidate."""

    spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [top-50 deferred identities]",
        feature_policy="accepted_plus_top_50_deferred_identities",
    )
    evaluation = evaluate_gpu_candidate(
        spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_top_common_identity_preprocessor,
    )
    first_training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[first_training_positions]
    preprocessor = make_top_common_identity_preprocessor().fit(
        X_training,
        partitioned_data.y_development.iloc[first_training_positions],
    )
    identity_encoder = dict(preprocessor.transformer_list)["top_common_identity"]
    identity_matrix = identity_encoder.transform(X_training)
    return TopCommonIdentityTrial(
        xgboost=evaluation,
        indicator_features=identity_matrix.shape[1],
        transformed_features_fold_1=len(preprocessor.get_feature_names_out()),
        training_coverage_fold_1=float(
            identity_matrix.getnnz(axis=1).sum()
            / (len(X_training) * len(DEFERRED_HIGH_CARDINALITY_FEATURES))
        ),
    )


def _identity_values(values: _pd.Series) -> _pd.Series:
    return values.astype("string").fillna(MISSING_IDENTITY)


def _validate_source_frame(X: _pd.DataFrame) -> None:
    if not isinstance(X, _pd.DataFrame):
        raise TypeError("Top-common identity encoding requires a DataFrame.")
    if tuple(X.columns) != EXPECTED_SOURCE_FEATURES:
        raise ValueError("Top-common identity encoding received unexpected columns.")

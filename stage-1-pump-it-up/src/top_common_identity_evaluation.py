"""Evaluate compact top-common identity indicators with accepted XGBoost."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
import time as _time

import numpy as _np
import pandas as _pd
from scipy import sparse as _sparse
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.pipeline import FeatureUnion as _FeatureUnion
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from categorical_frequency_forest_evaluation import (
    CategoricalFrequencyFeatureEngineer,
)
from categorical_frequency_forest_evaluation import FREQUENCY_FEATURES
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from feature_engineering import EXPECTED_SOURCE_FEATURES
from gpu_model_evaluation import CLASS_LABELS
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_preprocessing import make_initial_preprocessor
from spatial_density_features import SpatialDensityTransformer
from target_encoding_features import normalise_identity as _normalise_identity


TOP_COMMON_VALUES = 50
ARCHIVED_DEPTH_17_ITERATIONS = 600
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

    def __init__(
        self,
        *,
        top_k: int = TOP_COMMON_VALUES,
        normalise: bool = False,
    ):
        self.top_k = top_k
        self.normalise = normalise

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "TopCommonIdentityEncoder":
        _validate_source_frame(X)
        if not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not isinstance(self.normalise, bool):
            raise TypeError("normalise must be Boolean.")
        selected_values: dict[str, tuple[str, ...]] = {}
        feature_names = []
        for feature in DEFERRED_HIGH_CARDINALITY_FEATURES:
            values = _identity_values(X[feature], normalise=self.normalise)
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
            mapped = _identity_values(
                X[feature],
                normalise=self.normalise,
            ).map(self.value_positions_[feature])
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


def make_normalised_top_common_identity_preprocessor() -> _FeatureUnion:
    """Combine accepted preprocessing with normalised top-50 identities."""

    return _FeatureUnion(
        transformer_list=[
            ("accepted", make_initial_preprocessor()),
            (
                "top_common_identity",
                TopCommonIdentityEncoder(normalise=True),
            ),
        ]
    )


def make_top_common_spatial_density_preprocessor() -> _FeatureUnion:
    """Combine accepted, common-identity and waterpoint-density features."""

    return _FeatureUnion(
        transformer_list=[
            ("accepted", make_initial_preprocessor()),
            ("top_common_identity", TopCommonIdentityEncoder()),
            ("spatial_density", SpatialDensityTransformer()),
        ]
    )


def make_top_common_occurrence_preprocessor() -> _FeatureUnion:
    """Append all categorical occurrence counts to accepted/common identity."""

    occurrence_counts = _Pipeline(
        steps=[
            ("feature_engineering", CategoricalFrequencyFeatureEngineer()),
            (
                "count_selection",
                _ColumnTransformer(
                    transformers=[
                        ("occurrence_counts", "passthrough", list(FREQUENCY_FEATURES))
                    ],
                    remainder="drop",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )
    return _FeatureUnion(
        transformer_list=[
            ("accepted", make_initial_preprocessor()),
            ("top_common_identity", TopCommonIdentityEncoder()),
            ("categorical_occurrence", occurrence_counts),
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


def evaluate_archived_deep_xgboost(
    partitioned_data: PartitionedData,
    cross_validation: object,
    *,
    seed: int,
    preprocessor_factory: object = make_top_common_identity_preprocessor,
    representation_name: str = "top-50 deferred identities",
    feature_policy: str = "accepted_plus_top_50_deferred_identities",
) -> CandidateEvaluation:
    """Fit the archived depth-17, 600-tree specification on every fold."""

    spec = _replace(
        make_xgboost_spec(variant="archived depth 17", seed=seed),
        name=(
            f"XGBoost archived depth 17 [{representation_name}; "
            f"seed {seed}]"
        ),
        feature_policy=feature_policy,
    )
    probability_values = _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostics = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        started = _time.perf_counter()
        probabilities, fit_seconds = fit_gpu_candidate_probabilities(
            spec,
            partitioned_data.X_development.iloc[training_positions],
            partitioned_data.y_development.iloc[training_positions],
            partitioned_data.X_development.iloc[validation_positions],
            iterations=ARCHIVED_DEPTH_17_ITERATIONS,
            preprocessor_factory=preprocessor_factory,
        )
        probability_values[validation_positions] = probabilities
        diagnostics.append(
            {
                "validation_fold": fold_number,
                "selected_iterations": ARCHIVED_DEPTH_17_ITERATIONS,
                "inner_fit_rows": len(training_positions),
                "inner_stop_rows": 0,
                "stopping_seconds": 0.0,
                "refit_predict_seconds": fit_seconds,
                "total_seconds": _time.perf_counter() - started,
            }
        )
        print(
            f"Completed {spec.name} fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS} in "
            f"{diagnostics[-1]['total_seconds']:.1f} seconds.",
            flush=True,
        )
    return build_candidate_evaluation(
        model_name=spec.name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )


def _identity_values(
    values: _pd.Series,
    *,
    normalise: bool,
) -> _pd.Series:
    if normalise:
        return _normalise_identity(values)
    return values.astype("string").fillna(MISSING_IDENTITY)


def _validate_source_frame(X: _pd.DataFrame) -> None:
    if not isinstance(X, _pd.DataFrame):
        raise TypeError("Top-common identity encoding requires a DataFrame.")
    if tuple(X.columns) != EXPECTED_SOURCE_FEATURES:
        raise ValueError("Top-common identity encoding received unexpected columns.")

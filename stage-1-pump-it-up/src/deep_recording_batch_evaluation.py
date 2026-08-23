"""Evaluate exact recording batches with the archived deep XGBoost."""

from __future__ import annotations

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.pipeline import FeatureUnion as _FeatureUnion
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from feature_engineering import EXPECTED_SOURCE_FEATURES
from model_preprocessing import RARE_CATEGORY_MINIMUM
from top_common_identity_evaluation import TopCommonIdentityEncoder
from model_preprocessing import make_initial_preprocessor


RECORDING_BATCH_FEATURE = "recording_batch_identity"


class RecordingBatchEncoder(_BaseEstimator, _TransformerMixin):
    """Fold-fit a sparse one-hot block over exact recording dates."""

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "RecordingBatchEncoder":
        dates = _recording_dates(X)
        self.encoder_ = _OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=RARE_CATEGORY_MINIMUM,
            sparse_output=True,
        ).fit(dates)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame):
        _check_is_fitted(self, "encoder_")
        return self.encoder_.transform(_recording_dates(X))

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "encoder_")
        return self.encoder_.get_feature_names_out([RECORDING_BATCH_FEATURE])


def make_top_common_recording_batch_preprocessor() -> _FeatureUnion:
    """Combine accepted, top-common identity and exact-date features."""

    return _FeatureUnion(
        transformer_list=[
            ("accepted", make_initial_preprocessor()),
            ("top_common_identity", TopCommonIdentityEncoder()),
            ("recording_batch", RecordingBatchEncoder()),
        ]
    )


def _recording_dates(X: _pd.DataFrame) -> _pd.DataFrame:
    if not isinstance(X, _pd.DataFrame):
        raise TypeError("Recording-batch encoding requires a DataFrame.")
    if tuple(X.columns) != EXPECTED_SOURCE_FEATURES:
        raise ValueError("Recording-batch encoding received unexpected columns.")
    parsed = _pd.to_datetime(X["date_recorded"], errors="raise")
    if parsed.isna().any():
        raise ValueError("date_recorded must be complete for recording batches.")
    return _pd.DataFrame(
        {RECORDING_BATCH_FEATURE: parsed.dt.strftime("%Y-%m-%d")},
        index=X.index,
    )

"""Fold-fitted target-free waterpoint spatial-density features."""

from __future__ import annotations

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.neighbors import BallTree as _BallTree
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from feature_engineering import EXPECTED_SOURCE_FEATURES
from feature_engineering import valid_tanzania_coordinates


EARTH_RADIUS_KM = 6371.0088
NEIGHBOUR_RANKS = (1, 10)


class SpatialDensityTransformer(_BaseEstimator, _TransformerMixin):
    """Measure distances to fixed ranks in the fitted waterpoint reference."""

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "SpatialDensityTransformer":
        _validate_source_frame(X)
        coordinates, valid = _radian_coordinates(X)
        if int(valid.sum()) <= max(NEIGHBOUR_RANKS):
            raise ValueError("Spatial density requires more than ten valid sites.")
        self.reference_coordinates_ = coordinates[valid]
        self.tree_ = _BallTree(self.reference_coordinates_, metric="haversine")
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        self.feature_names_out_ = _np.asarray(
            [f"distance_to_waterpoint_{rank}_km" for rank in NEIGHBOUR_RANKS],
            dtype=object,
        )
        training_distances = self._query_raw(X, exclude_one_zero=True)
        medians = _np.nanmedian(training_distances, axis=0)
        if not _np.isfinite(medians).all():
            raise ValueError("Spatial-density medians are not finite.")
        self.imputation_medians_ = medians
        self.training_distances_ = training_distances
        return self

    def fit_transform(
        self,
        X: _pd.DataFrame,
        y: object = None,
        **fit_params: object,
    ) -> _np.ndarray:
        """Exclude each fitting row from its own neighbourhood."""

        self.fit(X, y)
        return self._fill_missing(self.training_distances_)

    def transform(self, X: _pd.DataFrame) -> _np.ndarray:
        _check_is_fitted(self, ("tree_", "imputation_medians_"))
        _validate_source_frame(X)
        return self._fill_missing(self._query_raw(X, exclude_one_zero=False))

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def _query_raw(
        self,
        X: _pd.DataFrame,
        *,
        exclude_one_zero: bool,
    ) -> _np.ndarray:
        coordinates, valid = _radian_coordinates(X)
        result = _np.full((len(X), len(NEIGHBOUR_RANKS)), _np.nan)
        query_count = max(NEIGHBOUR_RANKS) + int(exclude_one_zero)
        if valid.any():
            distances, _ = self.tree_.query(
                coordinates[valid],
                k=query_count,
                return_distance=True,
            )
            if exclude_one_zero:
                distances = distances[:, 1:]
            selected = distances[:, [rank - 1 for rank in NEIGHBOUR_RANKS]]
            result[valid] = selected * EARTH_RADIUS_KM
        return result

    def _fill_missing(self, values: _np.ndarray) -> _np.ndarray:
        result = _np.asarray(values, dtype="float64").copy()
        missing_rows, missing_columns = _np.where(~_np.isfinite(result))
        result[missing_rows, missing_columns] = self.imputation_medians_[
            missing_columns
        ]
        if not _np.isfinite(result).all() or (result < 0).any():
            raise ValueError("Spatial-density features are invalid.")
        return result


def _radian_coordinates(X: _pd.DataFrame) -> tuple[_np.ndarray, _np.ndarray]:
    longitude = _pd.to_numeric(X["longitude"], errors="coerce")
    latitude = _pd.to_numeric(X["latitude"], errors="coerce")
    valid = valid_tanzania_coordinates(longitude, latitude).to_numpy()
    degrees = _np.column_stack(
        (
            latitude.fillna(0).to_numpy(dtype="float64"),
            longitude.fillna(0).to_numpy(dtype="float64"),
        )
    )
    return _np.radians(degrees), valid


def _validate_source_frame(X: _pd.DataFrame) -> None:
    if not isinstance(X, _pd.DataFrame):
        raise TypeError("Spatial density requires a DataFrame.")
    if tuple(X.columns) != EXPECTED_SOURCE_FEATURES:
        raise ValueError("Spatial density received unexpected source columns.")

"""Deterministic geodesic distance from the coordinate origin."""

from __future__ import annotations

import numpy as _np
import pandas as _pd

from feature_engineering import valid_tanzania_coordinates


RADIAL_DISTANCE_FEATURE = "distance_from_origin_km"
EARTH_RADIUS_KM = 6371.0088


def geodesic_origin_distance_km(X: _pd.DataFrame) -> _pd.Series:
    """Return target-free origin distance for valid Tanzania coordinates."""

    longitude = _pd.to_numeric(X["longitude"], errors="coerce")
    latitude = _pd.to_numeric(X["latitude"], errors="coerce")
    valid = valid_tanzania_coordinates(longitude, latitude)
    longitude_radians = _np.radians(longitude.where(valid))
    latitude_radians = _np.radians(latitude.where(valid))
    haversine = (
        _np.sin(latitude_radians / 2.0) ** 2
        + _np.cos(latitude_radians)
        * _np.sin(longitude_radians / 2.0) ** 2
    ).clip(lower=0.0, upper=1.0)
    distance = 2.0 * EARTH_RADIUS_KM * _np.arcsin(_np.sqrt(haversine))
    return _pd.Series(distance, index=X.index, name=RADIAL_DISTANCE_FEATURE)

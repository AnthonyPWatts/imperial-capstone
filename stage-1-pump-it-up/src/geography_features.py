"""Deterministic geography policies for the Pump It Up feature screen."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

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

from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from feature_engineering import engineer_initial_features
from feature_engineering import valid_tanzania_coordinates


RARE_CATEGORY_MINIMUM = 20
MISSING_CATEGORY = "__missing__"
GEOGRAPHY_SELECTION_GATE_ACCURACY = 0.001
GEOGRAPHY_SELECTION_GATE_WORST_FOLD = -0.0025
GEOGRAPHY_SELECTION_GATE_REPAIR_RECALL = -0.02

_COORDINATE_FEATURES = (
    "longitude",
    "latitude",
    "coordinates_missing",
)
_NAMED_GEOGRAPHY_FEATURES = (
    "basin",
    "region",
    "region_code",
    "district_code",
    "lga",
)


@_dataclass(frozen=True)
class GeographyFeaturePolicy:
    """One explicit change to the accepted initial feature policy."""

    key: str
    label: str
    rationale: str
    remove_features: tuple[str, ...] = ()
    add_region_district: bool = False
    add_lga_ward: bool = False
    grid_degrees: tuple[float, ...] = ()
    impute_coordinates: bool = False
    add_coordinate_imputation_level: bool = False


@_dataclass(frozen=True)
class CoordinateCentroidModel:
    """Fold-fitted coordinate medians at decreasing geographic specificity."""

    ward_centroids: _pd.DataFrame
    lga_centroids: _pd.DataFrame
    region_centroids: _pd.DataFrame
    basin_centroids: _pd.DataFrame
    global_longitude: float
    global_latitude: float


GEOGRAPHY_POLICIES = {
    policy.key: policy
    for policy in (
        GeographyFeaturePolicy(
            key="baseline",
            label="Accepted current geography",
            rationale=(
                "Raw coordinate pair plus basin, region, region code, "
                "district code and LGA"
            ),
        ),
        GeographyFeaturePolicy(
            key="no_coordinates",
            label="Named geography without coordinates",
            rationale="Measure the incremental value of the raw coordinate pair",
            remove_features=_COORDINATE_FEATURES,
        ),
        GeographyFeaturePolicy(
            key="coordinates_only",
            label="Coordinates without named geography",
            rationale="Measure dependence on administrative and basin labels",
            remove_features=_NAMED_GEOGRAPHY_FEATURES,
        ),
        GeographyFeaturePolicy(
            key="coarse_region",
            label="Coordinates plus basin and region",
            rationale="Remove district and LGA detail while retaining broad context",
            remove_features=("region_code", "district_code", "lga"),
        ),
        GeographyFeaturePolicy(
            key="lga_backoff",
            label="Coordinates plus basin, region and LGA",
            rationale="Remove ambiguous raw administrative codes but retain LGA",
            remove_features=("region_code", "district_code"),
        ),
        GeographyFeaturePolicy(
            key="district_composite",
            label="Region-district composite",
            rationale=(
                "Replace ambiguous raw region and district codes with their "
                "nationally meaningful composite"
            ),
            remove_features=("region_code", "district_code"),
            add_region_district=True,
        ),
        GeographyFeaturePolicy(
            key="lga_ward_composite",
            label="Current geography plus LGA-ward",
            rationale="Disambiguate repeated ward names with their parent LGA",
            add_lga_ward=True,
        ),
        GeographyFeaturePolicy(
            key="grid_20km",
            label="Current geography plus ~20 km grid",
            rationale="Add a coarse spatial neighbourhood without target encoding",
            grid_degrees=(0.20,),
        ),
        GeographyFeaturePolicy(
            key="grid_10km",
            label="Current geography plus ~10 km grid",
            rationale="Add a mid-resolution spatial neighbourhood",
            grid_degrees=(0.10,),
        ),
        GeographyFeaturePolicy(
            key="multi_resolution_grid",
            label="Current geography plus ~20 km and ~10 km grids",
            rationale="Let the fixed models choose between two spatial scales",
            grid_degrees=(0.20, 0.10),
        ),
        GeographyFeaturePolicy(
            key="coordinate_centroids",
            label="Fold-fitted hierarchical coordinate centroids",
            rationale=(
                "Fill missing coordinates from ward, LGA, region, basin or "
                "global training-fold medians while retaining the missing flag"
            ),
            impute_coordinates=True,
        ),
        GeographyFeaturePolicy(
            key="coordinate_centroids_with_level",
            label="Coordinate centroids plus imputation level",
            rationale=(
                "Add the geographic level used for each fold-fitted coordinate "
                "imputation as an explicit categorical feature"
            ),
            impute_coordinates=True,
            add_coordinate_imputation_level=True,
        ),
        GeographyFeaturePolicy(
            key="combined_context",
            label="Administrative composites plus both grids",
            rationale=(
                "Combine only the independently motivated administrative and "
                "spatial representations"
            ),
            remove_features=("region_code", "district_code"),
            add_region_district=True,
            add_lga_ward=True,
            grid_degrees=(0.20, 0.10),
        ),
    )
}


class GeographyFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Apply one deterministic geography policy to the initial feature frame."""

    def __init__(self, policy: GeographyFeaturePolicy):
        self.policy = policy

    def fit(self, X: _pd.DataFrame, y: object = None) -> "GeographyFeatureEngineer":
        self.coordinate_centroid_model_ = (
            fit_coordinate_centroid_model(X)
            if self.policy.impute_coordinates
            else None
        )
        engineered = engineer_geography_features(
            X,
            self.policy,
            coordinate_centroid_model=self.coordinate_centroid_model_,
        )
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(engineered.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "feature_names_out_")
        engineered = engineer_geography_features(
            X,
            self.policy,
            coordinate_centroid_model=self.coordinate_centroid_model_,
        )
        if tuple(engineered.columns) != tuple(self.feature_names_out_):
            raise ValueError("Geography feature columns changed after fitting.")
        return engineered

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def engineer_geography_features(
    X: _pd.DataFrame,
    policy: GeographyFeaturePolicy,
    *,
    coordinate_centroid_model: CoordinateCentroidModel | None = None,
) -> _pd.DataFrame:
    """Return the initial model frame with one explicit geography treatment."""

    _validate_policy(policy)
    engineered = engineer_initial_features(X)
    if policy.impute_coordinates:
        if coordinate_centroid_model is None:
            raise ValueError(
                "Coordinate-centroid policies require a fold-fitted model."
            )
        engineered = impute_coordinates_from_centroids(
            X,
            engineered,
            coordinate_centroid_model,
            add_level=policy.add_coordinate_imputation_level,
        )
    if policy.remove_features:
        engineered = engineered.drop(columns=list(policy.remove_features))

    if policy.add_region_district:
        engineered["region_district"] = _combine_categories(
            X["region_code"],
            X["district_code"],
        )
    if policy.add_lga_ward:
        engineered["lga_ward"] = _combine_categories(X["lga"], X["ward"])
    for degrees in policy.grid_degrees:
        engineered[_grid_feature_name(degrees)] = _coordinate_grid(X, degrees)

    expected = geography_model_features(policy)
    if tuple(engineered.columns) != expected:
        raise ValueError(
            "Geography feature order changed; "
            f"expected {expected!r}, found {tuple(engineered.columns)!r}."
        )
    return engineered


def geography_model_features(policy: GeographyFeaturePolicy) -> tuple[str, ...]:
    """Return the stable feature order for a geography policy."""

    _validate_policy(policy)
    features = [name for name in MODEL_FEATURES if name not in policy.remove_features]
    if policy.add_region_district:
        features.append("region_district")
    if policy.add_lga_ward:
        features.append("lga_ward")
    if policy.add_coordinate_imputation_level:
        features.append("coordinate_imputation_level")
    features.extend(_grid_feature_name(value) for value in policy.grid_degrees)
    return tuple(features)


def geography_numeric_features(policy: GeographyFeaturePolicy) -> tuple[str, ...]:
    """Return numeric columns retained by a geography policy."""

    features = set(geography_model_features(policy))
    return tuple(name for name in NUMERIC_FEATURES if name in features)


def geography_categorical_features(
    policy: GeographyFeaturePolicy,
) -> tuple[str, ...]:
    """Return categorical columns retained or created by a geography policy."""

    features = geography_model_features(policy)
    numeric = set(geography_numeric_features(policy))
    return tuple(name for name in features if name not in numeric)


def make_geography_preprocessor(
    policy: GeographyFeaturePolicy,
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build fold-fitted preprocessing for one geography policy."""

    numeric_steps: list[tuple[str, object]] = [
        (
            "median_imputation",
            _SimpleImputer(strategy="median", add_indicator=True),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("standardisation", _StandardScaler()))
    numeric_pipeline = _Pipeline(steps=numeric_steps)
    categorical_encoder = _OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=RARE_CATEGORY_MINIMUM,
        sparse_output=sparse_output,
    )
    column_preprocessing = _ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(geography_numeric_features(policy))),
            (
                "categorical",
                categorical_encoder,
                list(geography_categorical_features(policy)),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", GeographyFeatureEngineer(policy)),
            ("column_preprocessing", column_preprocessing),
        ]
    )


def summarise_geography_policies() -> _pd.DataFrame:
    """Return the pre-declared geography screen in execution order."""

    rows = []
    for policy in GEOGRAPHY_POLICIES.values():
        rows.append(
            {
                "policy": policy.key,
                "label": policy.label,
                "engineered_features": len(geography_model_features(policy)),
                "numeric_features": len(geography_numeric_features(policy)),
                "categorical_features": len(
                    geography_categorical_features(policy)
                ),
                "rationale": policy.rationale,
            }
        )
    return _pd.DataFrame(rows).set_index("policy")


def fit_coordinate_centroid_model(X: _pd.DataFrame) -> CoordinateCentroidModel:
    """Learn robust coordinate centres from one training partition only."""

    longitude = _pd.to_numeric(X["longitude"], errors="coerce")
    latitude = _pd.to_numeric(X["latitude"], errors="coerce")
    valid = valid_tanzania_coordinates(longitude, latitude)
    if not valid.any():
        raise ValueError("Coordinate-centroid fitting requires valid coordinates.")

    valid_coordinates = _pd.DataFrame(
        {
            "longitude": longitude.loc[valid],
            "latitude": latitude.loc[valid],
            "ward_key": _combine_categories(X["lga"], X["ward"]).loc[valid],
            "lga_key": _normalise_category(X["lga"]).loc[valid],
            "region_key": _normalise_category(X["region"]).loc[valid],
            "basin_key": _normalise_category(X["basin"]).loc[valid],
        }
    )
    return CoordinateCentroidModel(
        ward_centroids=_group_centroids(
            valid_coordinates,
            "ward_key",
            minimum_rows=5,
        ),
        lga_centroids=_group_centroids(
            valid_coordinates,
            "lga_key",
            minimum_rows=10,
        ),
        region_centroids=_group_centroids(
            valid_coordinates,
            "region_key",
            minimum_rows=1,
        ),
        basin_centroids=_group_centroids(
            valid_coordinates,
            "basin_key",
            minimum_rows=1,
        ),
        global_longitude=float(longitude.loc[valid].median()),
        global_latitude=float(latitude.loc[valid].median()),
    )


def impute_coordinates_from_centroids(
    X: _pd.DataFrame,
    engineered: _pd.DataFrame,
    model: CoordinateCentroidModel,
    *,
    add_level: bool,
) -> _pd.DataFrame:
    """Fill invalid coordinate pairs through the fitted geographic hierarchy."""

    result = engineered.copy()
    missing = result["coordinates_missing"].eq(1)
    levels = _pd.Series("observed", index=result.index, dtype="string")
    hierarchy = (
        (
            "ward",
            _combine_categories(X["lga"], X["ward"]),
            model.ward_centroids,
        ),
        ("lga", _normalise_category(X["lga"]), model.lga_centroids),
        ("region", _normalise_category(X["region"]), model.region_centroids),
        ("basin", _normalise_category(X["basin"]), model.basin_centroids),
    )
    pending = missing.copy()
    for level, keys, centroids in hierarchy:
        mapped_longitude = keys.map(centroids["longitude"])
        mapped_latitude = keys.map(centroids["latitude"])
        available = pending & mapped_longitude.notna() & mapped_latitude.notna()
        result.loc[available, "longitude"] = mapped_longitude.loc[available]
        result.loc[available, "latitude"] = mapped_latitude.loc[available]
        levels.loc[available] = level
        pending &= ~available

    result.loc[pending, "longitude"] = model.global_longitude
    result.loc[pending, "latitude"] = model.global_latitude
    levels.loc[pending] = "global"
    if result.loc[missing, ["longitude", "latitude"]].isna().any().any():
        raise ValueError("Coordinate-centroid imputation left missing values.")
    if add_level:
        result["coordinate_imputation_level"] = levels
    return result


def _combine_categories(left: _pd.Series, right: _pd.Series) -> _pd.Series:
    return _normalise_category(left).str.cat(
        _normalise_category(right),
        sep="::",
    )


def _normalise_category(values: _pd.Series) -> _pd.Series:
    normalised = values.astype("string").str.strip()
    return normalised.mask(normalised.eq("")).fillna(MISSING_CATEGORY)


def _coordinate_grid(X: _pd.DataFrame, degrees: float) -> _pd.Series:
    longitude = _pd.to_numeric(X["longitude"], errors="coerce")
    latitude = _pd.to_numeric(X["latitude"], errors="coerce")
    valid = valid_tanzania_coordinates(longitude, latitude)
    result = _pd.Series(MISSING_CATEGORY, index=X.index, dtype="string")
    longitude_bin = _np.floor(longitude.loc[valid] / degrees).astype("int64")
    latitude_bin = _np.floor(latitude.loc[valid] / degrees).astype("int64")
    result.loc[valid] = longitude_bin.astype("string").str.cat(
        latitude_bin.astype("string"),
        sep="::",
    )
    return result


def _group_centroids(
    values: _pd.DataFrame,
    key: str,
    *,
    minimum_rows: int,
) -> _pd.DataFrame:
    grouped = values.groupby(key, observed=True)[["longitude", "latitude"]]
    medians = grouped.median()
    counts = grouped.size()
    return medians.loc[counts.ge(minimum_rows)].copy()


def _grid_feature_name(degrees: float) -> str:
    if degrees <= 0:
        raise ValueError("Coordinate-grid degrees must be positive.")
    suffix = str(degrees).replace(".", "_")
    return f"coordinate_grid_{suffix}_degrees"


def _validate_policy(policy: GeographyFeaturePolicy) -> None:
    if policy.key not in GEOGRAPHY_POLICIES:
        raise ValueError(f"Unknown geography policy: {policy.key!r}.")
    if GEOGRAPHY_POLICIES[policy.key] != policy:
        raise ValueError(f"Geography policy {policy.key!r} differs from its registry.")
    unknown_removals = set(policy.remove_features) - set(MODEL_FEATURES)
    if unknown_removals:
        raise ValueError(f"Unknown features requested for removal: {unknown_removals!r}.")
    if len(policy.remove_features) != len(set(policy.remove_features)):
        raise ValueError("Geography policy repeats a removed feature.")
    if len(policy.grid_degrees) != len(set(policy.grid_degrees)):
        raise ValueError("Geography policy repeats a grid resolution.")
    if any(value <= 0 for value in policy.grid_degrees):
        raise ValueError("Coordinate-grid degrees must be positive.")
    if policy.add_coordinate_imputation_level and not policy.impute_coordinates:
        raise ValueError(
            "The coordinate-imputation level requires coordinate imputation."
        )

"""Deterministic feature engineering for the first Pump It Up model."""

from __future__ import annotations

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.utils.validation import check_is_fitted as _check_is_fitted


# DrivenData no longer exposes the original competition start date. This is
# the date of the earliest surviving official Pump It Up community post, so it
# provides a fixed competition-era reference without using the current date.
COMPETITION_REFERENCE_DATE = _pd.Timestamp("2015-02-02")

EXPECTED_SOURCE_FEATURES = (
    "amount_tsh",
    "date_recorded",
    "funder",
    "gps_height",
    "installer",
    "longitude",
    "latitude",
    "wpt_name",
    "num_private",
    "basin",
    "subvillage",
    "region",
    "region_code",
    "district_code",
    "lga",
    "ward",
    "population",
    "public_meeting",
    "scheme_management",
    "scheme_name",
    "permit",
    "construction_year",
    "extraction_type",
    "extraction_type_group",
    "extraction_type_class",
    "management",
    "management_group",
    "payment_type",
    "water_quality",
    "quality_group",
    "quantity",
    "source",
    "source_type",
    "source_class",
    "waterpoint_type",
    "waterpoint_type_group",
)

NUMERIC_FEATURES = (
    "amount_tsh",
    "gps_height",
    "longitude",
    "latitude",
    "num_private",
    "population",
    "days_since_recorded",
    "pump_age_at_recording",
    "amount_tsh_recorded",
    "gps_height_missing",
    "coordinates_missing",
    "population_missing",
    "construction_year_missing",
    "construction_year_inconsistent",
)

CATEGORICAL_FEATURES = (
    "basin",
    "region",
    "region_code",
    "district_code",
    "lga",
    "public_meeting",
    "scheme_management",
    "permit",
    "extraction_type",
    "management",
    "payment_type",
    "water_quality",
    "quantity",
    "source",
    "waterpoint_type",
)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DEFERRED_HIGH_CARDINALITY_FEATURES = (
    "funder",
    "installer",
    "wpt_name",
    "subvillage",
    "ward",
    "scheme_name",
)

DEFERRED_HIERARCHY_FEATURES = (
    "extraction_type_group",
    "extraction_type_class",
    "management_group",
    "quality_group",
    "source_type",
    "source_class",
    "waterpoint_type_group",
)

_MISSING_CATEGORY = "__missing__"


class InitialFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Expose deterministic engineering as a scikit-learn pipeline step."""

    def fit(self, X: _pd.DataFrame, y: object = None) -> "InitialFeatureEngineer":
        """Validate the source contract; no statistics are learned."""

        _validate_source_features(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        """Apply the fixed feature policy without mutating ``X``."""

        _check_is_fitted(self, "feature_names_in_")
        return engineer_initial_features(X)

    def get_feature_names_out(
        self,
        input_features: object = None,
    ) -> _np.ndarray:
        """Return the engineered feature names in their stable order."""

        _check_is_fitted(self, "feature_names_in_")
        return _np.asarray(MODEL_FEATURES, dtype=object)


def engineer_initial_features(X: _pd.DataFrame) -> _pd.DataFrame:
    """Return the conservative, deterministic first-model feature frame.

    ``date_recorded`` becomes elapsed days from a fixed competition-era date;
    no calendar components are retained. Sentinel handling creates explicit
    availability flags and leaves learned imputation to the current fold.
    """

    _validate_source_features(X)

    recorded = _pd.to_datetime(X["date_recorded"], errors="raise")
    days_since_recorded = (COMPETITION_REFERENCE_DATE - recorded).dt.days
    if days_since_recorded.isna().any() or days_since_recorded.lt(0).any():
        raise ValueError(
            "date_recorded must be complete and no later than the fixed "
            f"reference date {COMPETITION_REFERENCE_DATE.date()}."
        )

    construction_year = _pd.to_numeric(
        X["construction_year"],
        errors="coerce",
    )
    construction_year_missing = construction_year.isna() | construction_year.eq(0)
    construction_year_inconsistent = (
        ~construction_year_missing
        & construction_year.gt(recorded.dt.year)
    )
    valid_construction_year = (
        ~construction_year_missing
        & ~construction_year_inconsistent
    )
    pump_age_at_recording = (recorded.dt.year - construction_year).where(
        valid_construction_year
    )

    longitude = _pd.to_numeric(X["longitude"], errors="coerce")
    latitude = _pd.to_numeric(X["latitude"], errors="coerce")
    coordinates_missing = (
        longitude.isna()
        | latitude.isna()
        | longitude.abs().le(1e-6)
    )

    gps_height = _pd.to_numeric(X["gps_height"], errors="coerce")
    gps_height_missing = gps_height.isna() | gps_height.eq(0)

    population = _pd.to_numeric(X["population"], errors="coerce")
    population_missing = population.isna() | population.eq(0)

    engineered = _pd.DataFrame(index=X.index)
    engineered["amount_tsh"] = _pd.to_numeric(X["amount_tsh"], errors="coerce")
    engineered["gps_height"] = gps_height.mask(gps_height_missing)
    engineered["longitude"] = longitude.mask(coordinates_missing)
    engineered["latitude"] = latitude.mask(coordinates_missing)
    engineered["num_private"] = _pd.to_numeric(X["num_private"], errors="coerce")
    engineered["population"] = population.mask(population_missing)
    engineered["days_since_recorded"] = days_since_recorded.astype("float64")
    engineered["pump_age_at_recording"] = pump_age_at_recording.astype("float64")
    engineered["amount_tsh_recorded"] = engineered["amount_tsh"].gt(0).astype("int8")
    engineered["gps_height_missing"] = gps_height_missing.astype("int8")
    engineered["coordinates_missing"] = coordinates_missing.astype("int8")
    engineered["population_missing"] = population_missing.astype("int8")
    engineered["construction_year_missing"] = construction_year_missing.astype("int8")
    engineered["construction_year_inconsistent"] = (
        construction_year_inconsistent.astype("int8")
    )

    for column in CATEGORICAL_FEATURES:
        values = X[column].astype("string").str.strip()
        engineered[column] = values.mask(values.eq("")).fillna(
            _MISSING_CATEGORY
        )

    return engineered.loc[:, MODEL_FEATURES]


def summarise_initial_feature_policy() -> _pd.DataFrame:
    """Return the compact feature-policy table shown in the notebook."""

    rows = [
        {
            "feature_group": "Recording date",
            "treatment": (
                "Replace date_recorded with days_since_recorded from "
                f"{COMPETITION_REFERENCE_DATE.date()}"
            ),
        },
        {
            "feature_group": "Construction year",
            "treatment": (
                "Replace with pump_age_at_recording plus missing and "
                "inconsistent flags"
            ),
        },
        {
            "feature_group": "Measurement sentinels",
            "treatment": (
                "Flag unavailable height, coordinates and population; "
                "leave fold-fitted median imputation downstream"
            ),
        },
        {
            "feature_group": "Categorical predictors",
            "treatment": (
                "Use explicit missing values, fold-fitted rare grouping and "
                "one-hot encoding"
            ),
        },
        {
            "feature_group": "Deferred high-cardinality",
            "treatment": ", ".join(DEFERRED_HIGH_CARDINALITY_FEATURES),
        },
        {
            "feature_group": "Deferred hierarchy alternatives",
            "treatment": ", ".join(DEFERRED_HIERARCHY_FEATURES),
        },
    ]
    return _pd.DataFrame(rows).set_index("feature_group")


def _validate_source_features(X: _pd.DataFrame) -> None:
    if not isinstance(X, _pd.DataFrame):
        raise TypeError("Feature engineering requires a pandas DataFrame.")
    if not X.columns.is_unique:
        raise ValueError("Source predictor labels must be unique.")
    actual = tuple(X.columns)
    if actual != EXPECTED_SOURCE_FEATURES:
        missing = [name for name in EXPECTED_SOURCE_FEATURES if name not in actual]
        unexpected = [name for name in actual if name not in EXPECTED_SOURCE_FEATURES]
        raise ValueError(
            "Source predictor contract changed; "
            f"missing={missing!r}, unexpected={unexpected!r}, "
            f"order_matches={set(actual) == set(EXPECTED_SOURCE_FEATURES)}."
        )

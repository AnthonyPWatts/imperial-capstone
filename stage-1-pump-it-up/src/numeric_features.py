"""Fold-fitted numeric state and imputation policies for the feature screen."""

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
from model_preprocessing import RARE_CATEGORY_MINIMUM


SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_WORST_FOLD = -0.0025
SELECTION_GATE_REPAIR_RECALL = -0.02
GEOGRAPHIC_MEDIAN_MINIMUM_LGA_ROWS = 20
NUMERIC_SENTINEL_FEATURES = ("gps_height", "population")
POPULATION_ONE_FEATURE = "population_one"
NUM_PRIVATE_NONZERO_FEATURE = "num_private_nonzero"
MEASUREMENT_BLOCK_MISSING_FEATURE = "measurement_block_missing"
_ADDED_FEATURE_ORDER = (
    POPULATION_ONE_FEATURE,
    NUM_PRIVATE_NONZERO_FEATURE,
    MEASUREMENT_BLOCK_MISSING_FEATURE,
)
_MISSING_CATEGORY = "__missing__"


@_dataclass(frozen=True)
class NumericFeaturePolicy:
    """One explicit treatment of the four audited numeric fields."""

    key: str
    label: str
    rationale: str
    remove_features: tuple[str, ...] = ()
    retain_zero_features: tuple[str, ...] = ()
    geographic_median_features: tuple[str, ...] = ()
    add_population_one: bool = False
    add_num_private_nonzero: bool = False
    add_measurement_block_missing: bool = False


@_dataclass(frozen=True)
class GeographicMedianModel:
    """Training-partition medians for one numeric measurement."""

    feature: str
    lga_medians: _pd.Series
    region_medians: _pd.Series
    global_median: float


NUMERIC_POLICIES = {
    policy.key: policy
    for policy in (
        NumericFeaturePolicy(
            key="baseline",
            label="Accepted numeric policy",
            rationale="Retain the accepted state flags and fold median imputation",
        ),
        NumericFeaturePolicy(
            key="amount_recorded_only",
            label="Amount recorded state only",
            rationale="Remove tariff magnitude while retaining amount_tsh_recorded",
            remove_features=("amount_tsh",),
        ),
        NumericFeaturePolicy(
            key="gps_zero_as_value",
            label="GPS height zero retained",
            rationale="Retain zero height as a value alongside gps_height_missing",
            retain_zero_features=("gps_height",),
        ),
        NumericFeaturePolicy(
            key="gps_geographic_median",
            label="GPS height geographic median",
            rationale=(
                "Fill zero height from fold-fitted LGA, region or global medians"
            ),
            geographic_median_features=("gps_height",),
        ),
        NumericFeaturePolicy(
            key="population_zero_as_value",
            label="Population zero retained",
            rationale="Retain zero population alongside population_missing",
            retain_zero_features=("population",),
        ),
        NumericFeaturePolicy(
            key="population_one_state",
            label="Population-one state",
            rationale="Expose the 7,025-row population-one spike explicitly",
            add_population_one=True,
        ),
        NumericFeaturePolicy(
            key="population_geographic_median",
            label="Population geographic median",
            rationale=(
                "Fill zero population from fold-fitted LGA, region or global medians"
            ),
            geographic_median_features=("population",),
        ),
        NumericFeaturePolicy(
            key="num_private_drop",
            label="Remove num_private",
            rationale="Ablate an undocumented feature that is zero in 98.7% of rows",
            remove_features=("num_private",),
        ),
        NumericFeaturePolicy(
            key="num_private_nonzero_only",
            label="num_private non-zero state only",
            rationale="Replace sparse magnitude with a non-zero state indicator",
            remove_features=("num_private",),
            add_num_private_nonzero=True,
        ),
        NumericFeaturePolicy(
            key="measurement_block_state",
            label="Shared measurement-block state",
            rationale=(
                "Expose simultaneous zero height, population and construction year"
            ),
            add_measurement_block_missing=True,
        ),
        NumericFeaturePolicy(
            key="both_geographic_medians",
            label="Height and population geographic medians",
            rationale="Apply both fold-fitted geographic median treatments",
            geographic_median_features=NUMERIC_SENTINEL_FEATURES,
        ),
        NumericFeaturePolicy(
            key="combined_state_indicators",
            label="Combined numeric state indicators",
            rationale=(
                "Combine population-one, num_private non-zero and shared-block states"
            ),
            remove_features=("num_private",),
            add_population_one=True,
            add_num_private_nonzero=True,
            add_measurement_block_missing=True,
        ),
    )
}


class NumericFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Apply one policy and learn geographic medians from the current fold."""

    def __init__(self, policy: NumericFeaturePolicy):
        self.policy = policy

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "NumericFeatureEngineer":
        _validate_policy(self.policy)
        engineer_initial_features(X)
        self.geographic_median_models_ = {
            feature: fit_geographic_median_model(X, feature)
            for feature in self.policy.geographic_median_features
        }
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(
            numeric_model_features(self.policy),
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, ("geographic_median_models_", "feature_names_out_"))
        return engineer_numeric_features(
            X,
            self.policy,
            geographic_median_models=self.geographic_median_models_,
        )

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def fit_geographic_median_model(
    X: _pd.DataFrame,
    feature: str,
) -> GeographicMedianModel:
    """Fit LGA, region and global medians from non-sentinel measurements."""

    if feature not in NUMERIC_SENTINEL_FEATURES:
        raise ValueError(f"Unsupported geographic-median feature: {feature!r}.")
    values = _pd.to_numeric(X[feature], errors="coerce")
    valid = values.notna() & values.ne(0)
    if not valid.any():
        raise ValueError(f"No valid {feature} values are available for imputation.")
    frame = _pd.DataFrame(
        {
            "value": values.loc[valid],
            "lga": _normalise_category(X.loc[valid, "lga"]),
            "region": _normalise_category(X.loc[valid, "region"]),
        }
    )
    lga_grouped = frame.groupby("lga", observed=True)["value"]
    lga_counts = lga_grouped.size()
    lga_medians = lga_grouped.median().loc[
        lga_counts.ge(GEOGRAPHIC_MEDIAN_MINIMUM_LGA_ROWS)
    ]
    return GeographicMedianModel(
        feature=feature,
        lga_medians=lga_medians,
        region_medians=frame.groupby("region", observed=True)["value"].median(),
        global_median=float(values.loc[valid].median()),
    )


def engineer_numeric_features(
    X: _pd.DataFrame,
    policy: NumericFeaturePolicy,
    *,
    geographic_median_models: dict[str, GeographicMedianModel],
) -> _pd.DataFrame:
    """Return the accepted frame with one explicit numeric treatment."""

    _validate_policy(policy)
    if set(geographic_median_models) != set(policy.geographic_median_features):
        raise ValueError("Geographic median models do not match the selected policy.")

    engineered = engineer_initial_features(X)
    for feature in policy.retain_zero_features:
        engineered[feature] = _pd.to_numeric(X[feature], errors="coerce")
    for feature in policy.geographic_median_features:
        missing = engineered[feature].isna()
        if missing.any():
            model = geographic_median_models[feature]
            lga_values = _normalise_category(X.loc[missing, "lga"]).map(
                model.lga_medians
            )
            region_values = _normalise_category(X.loc[missing, "region"]).map(
                model.region_medians
            )
            engineered.loc[missing, feature] = (
                lga_values.fillna(region_values).fillna(model.global_median)
            )
        if engineered[feature].isna().any():
            raise ValueError(f"Geographic imputation left missing {feature} values.")

    if policy.remove_features:
        engineered = engineered.drop(columns=list(policy.remove_features))
    if policy.add_population_one:
        engineered[POPULATION_ONE_FEATURE] = (
            _pd.to_numeric(X["population"], errors="coerce").eq(1).astype("int8")
        )
    if policy.add_num_private_nonzero:
        engineered[NUM_PRIVATE_NONZERO_FEATURE] = (
            _pd.to_numeric(X["num_private"], errors="coerce").gt(0).astype("int8")
        )
    if policy.add_measurement_block_missing:
        gps_zero = _pd.to_numeric(X["gps_height"], errors="coerce").eq(0)
        population_zero = _pd.to_numeric(
            X["population"], errors="coerce"
        ).eq(0)
        construction_zero = _pd.to_numeric(
            X["construction_year"], errors="coerce"
        ).eq(0)
        engineered[MEASUREMENT_BLOCK_MISSING_FEATURE] = (
            gps_zero & population_zero & construction_zero
        ).astype("int8")

    expected = numeric_model_features(policy)
    if tuple(engineered.columns) != expected:
        raise ValueError(
            "Numeric feature order changed; "
            f"expected {expected!r}, found {tuple(engineered.columns)!r}."
        )
    return engineered


def numeric_model_features(policy: NumericFeaturePolicy) -> tuple[str, ...]:
    """Return the stable feature order for one numeric policy."""

    _validate_policy(policy)
    features = [name for name in MODEL_FEATURES if name not in policy.remove_features]
    selected_additions = {
        POPULATION_ONE_FEATURE: policy.add_population_one,
        NUM_PRIVATE_NONZERO_FEATURE: policy.add_num_private_nonzero,
        MEASUREMENT_BLOCK_MISSING_FEATURE: policy.add_measurement_block_missing,
    }
    features.extend(name for name in _ADDED_FEATURE_ORDER if selected_additions[name])
    return tuple(features)


def numeric_numeric_features(policy: NumericFeaturePolicy) -> tuple[str, ...]:
    """Return numeric columns retained or created by one policy."""

    features = set(numeric_model_features(policy))
    return tuple(
        name
        for name in (*NUMERIC_FEATURES, *_ADDED_FEATURE_ORDER)
        if name in features
    )


def make_numeric_preprocessor(
    policy: NumericFeaturePolicy,
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build fold-fitted preprocessing for one numeric feature policy."""

    numeric_steps: list[tuple[str, object]] = [
        (
            "median_imputation",
            _SimpleImputer(strategy="median", add_indicator=True),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("standardisation", _StandardScaler()))
    column_preprocessing = _ColumnTransformer(
        transformers=[
            (
                "numeric",
                _Pipeline(steps=numeric_steps),
                list(numeric_numeric_features(policy)),
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
            ("feature_engineering", NumericFeatureEngineer(policy)),
            ("column_preprocessing", column_preprocessing),
        ]
    )


def summarise_numeric_policies() -> _pd.DataFrame:
    """Return the predeclared numeric screen in execution order."""

    rows = []
    for policy in NUMERIC_POLICIES.values():
        rows.append(
            {
                "policy": policy.key,
                "label": policy.label,
                "engineered_features": len(numeric_model_features(policy)),
                "removed_features": ", ".join(policy.remove_features),
                "zero_retained": ", ".join(policy.retain_zero_features),
                "geographic_medians": ", ".join(
                    policy.geographic_median_features
                ),
                "rationale": policy.rationale,
            }
        )
    return _pd.DataFrame(rows).set_index("policy")


def _normalise_category(values: _pd.Series) -> _pd.Series:
    return values.astype("string").str.strip().replace("", _pd.NA).fillna(
        _MISSING_CATEGORY
    )


def _validate_policy(policy: NumericFeaturePolicy) -> None:
    if policy.key not in NUMERIC_POLICIES:
        raise ValueError(f"Unknown numeric policy: {policy.key!r}.")
    if NUMERIC_POLICIES[policy.key] != policy:
        raise ValueError(f"Policy {policy.key!r} differs from its registry.")
    if len(policy.remove_features) != len(set(policy.remove_features)):
        raise ValueError("A numeric policy repeats a removed feature.")
    if set(policy.remove_features) - set(NUMERIC_FEATURES):
        raise ValueError("A numeric policy removes a non-numeric feature.")
    if set(policy.retain_zero_features) - set(NUMERIC_SENTINEL_FEATURES):
        raise ValueError("A policy retains zero for an unsupported feature.")
    if set(policy.geographic_median_features) - set(NUMERIC_SENTINEL_FEATURES):
        raise ValueError("A policy imputes an unsupported geographic feature.")
    overlap = set(policy.retain_zero_features) & set(
        policy.geographic_median_features
    )
    if overlap:
        raise ValueError(f"A policy both retains and imputes zero: {overlap!r}.")

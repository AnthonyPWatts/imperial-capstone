"""Leakage-safe multiclass target encoding for deferred identity fields."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.preprocessing import StandardScaler as _StandardScaler
from sklearn.preprocessing import TargetEncoder as _TargetEncoder
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from feature_engineering import engineer_initial_features


TARGET_ENCODING_INNER_FOLDS = 5
TARGET_ENCODING_SEED = 20260822
BASE_RARE_CATEGORY_MINIMUM = 20
SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_WORST_FOLD = -0.0025
SELECTION_GATE_REPAIR_RECALL = -0.02
BLANK_IDENTITY = "__blank__"
MISSING_IDENTITY = "__missing__"

LOCATION_IDENTITY_FEATURES = (
    "lga_ward_identity",
    "lga_subvillage_identity",
    "scheme_name_identity",
)
ORGANISATION_IDENTITY_FEATURES = (
    "funder_identity",
    "installer_identity",
)
ALL_IDENTITY_FEATURES = (
    *LOCATION_IDENTITY_FEATURES,
    *ORGANISATION_IDENTITY_FEATURES,
)


@_dataclass(frozen=True)
class TargetEncodingPolicy:
    """One predeclared set of high-cardinality identities to encode."""

    key: str
    label: str
    rationale: str
    identity_features: tuple[str, ...]


TARGET_ENCODING_POLICIES = {
    policy.key: policy
    for policy in (
        TargetEncodingPolicy(
            key="location_identity",
            label="Location and scheme identities",
            rationale=(
                "Encode LGA-ward, LGA-subvillage and scheme name after their "
                "stronger leakage-safe univariate audit"
            ),
            identity_features=LOCATION_IDENTITY_FEATURES,
        ),
        TargetEncodingPolicy(
            key="organisation_identity",
            label="Funder and installer identities",
            rationale=(
                "Revisit the audited organisation fields through smoothed "
                "class evidence rather than identity one-hot or frequency"
            ),
            identity_features=ORGANISATION_IDENTITY_FEATURES,
        ),
        TargetEncodingPolicy(
            key="combined_identity",
            label="Location, scheme and organisation identities",
            rationale=(
                "Combine the five predeclared identity encodings without "
                "adding waterpoint name or tuning the smoothing strength"
            ),
            identity_features=ALL_IDENTITY_FEATURES,
        ),
    )
}


class TargetEncodingFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Create deterministic identity columns before target encoding."""

    def __init__(self, policy: TargetEncodingPolicy):
        self.policy = policy

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "TargetEncodingFeatureEngineer":
        _validate_policy(self.policy)
        engineered = engineer_target_encoding_features(X, self.policy)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(engineered.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "feature_names_out_")
        engineered = engineer_target_encoding_features(X, self.policy)
        if tuple(engineered.columns) != tuple(self.feature_names_out_):
            raise ValueError("Target-encoding feature columns changed after fitting.")
        return engineered

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def normalise_identity(values: _pd.Series) -> _pd.Series:
    """Normalise case and whitespace while preserving literal sentinel values."""

    missing = values.isna()
    normalised = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )
    normalised = normalised.mask(
        ~missing & normalised.eq(""),
        BLANK_IDENTITY,
    )
    return normalised.mask(missing, MISSING_IDENTITY).fillna(MISSING_IDENTITY)


def identity_feature_frame(X: _pd.DataFrame) -> _pd.DataFrame:
    """Return every audited identity column in a deterministic order."""

    lga = normalise_identity(X["lga"])
    return _pd.DataFrame(
        {
            "lga_ward_identity": lga.str.cat(
                normalise_identity(X["ward"]),
                sep="::",
            ),
            "lga_subvillage_identity": lga.str.cat(
                normalise_identity(X["subvillage"]),
                sep="::",
            ),
            "scheme_name_identity": normalise_identity(X["scheme_name"]),
            "funder_identity": normalise_identity(X["funder"]),
            "installer_identity": normalise_identity(X["installer"]),
        },
        index=X.index,
    )


def engineer_target_encoding_features(
    X: _pd.DataFrame,
    policy: TargetEncodingPolicy,
) -> _pd.DataFrame:
    """Return the accepted frame plus selected high-cardinality identities."""

    _validate_policy(policy)
    engineered = engineer_initial_features(X)
    identities = identity_feature_frame(X)
    for feature in policy.identity_features:
        engineered[feature] = identities[feature]
    expected = target_encoding_model_features(policy)
    if tuple(engineered.columns) != expected:
        raise ValueError("Target-encoding feature order changed.")
    return engineered


def target_encoding_model_features(
    policy: TargetEncodingPolicy,
) -> tuple[str, ...]:
    """Return the stable engineered-frame feature order."""

    _validate_policy(policy)
    return (*MODEL_FEATURES, *policy.identity_features)


def make_target_encoding_preprocessor(
    policy: TargetEncodingPolicy,
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build preprocessing with inner cross-fitted multiclass target encoding."""

    _validate_policy(policy)
    numeric_steps: list[tuple[str, object]] = [
        (
            "median_imputation",
            _SimpleImputer(strategy="median", add_indicator=True),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("standardisation", _StandardScaler()))
    target_cross_validation = _StratifiedKFold(
        n_splits=TARGET_ENCODING_INNER_FOLDS,
        shuffle=True,
        random_state=TARGET_ENCODING_SEED,
    )
    column_preprocessing = _ColumnTransformer(
        transformers=[
            (
                "numeric",
                _Pipeline(steps=numeric_steps),
                list(NUMERIC_FEATURES),
            ),
            (
                "base_categorical",
                _OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=BASE_RARE_CATEGORY_MINIMUM,
                    sparse_output=sparse_output,
                ),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "target_encoded",
                _TargetEncoder(
                    target_type="multiclass",
                    smooth="auto",
                    cv=target_cross_validation,
                ),
                list(policy.identity_features),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", TargetEncodingFeatureEngineer(policy)),
            ("column_preprocessing", column_preprocessing),
        ]
    )


def summarise_target_encoding_policies() -> _pd.DataFrame:
    """Return the fixed target-encoding candidate register."""

    return _pd.DataFrame(
        [
            {
                "policy": policy.key,
                "label": policy.label,
                "identity_features": ", ".join(policy.identity_features),
                "identity_feature_count": len(policy.identity_features),
                "encoded_numeric_features": len(policy.identity_features) * 3,
                "smoothing": "empirical Bayes auto",
                "inner_cross_fitting_folds": TARGET_ENCODING_INNER_FOLDS,
                "rationale": policy.rationale,
            }
            for policy in TARGET_ENCODING_POLICIES.values()
        ]
    ).set_index("policy")


def _validate_policy(policy: TargetEncodingPolicy) -> None:
    if policy.key not in TARGET_ENCODING_POLICIES:
        raise ValueError(f"Unknown target-encoding policy: {policy.key!r}.")
    if TARGET_ENCODING_POLICIES[policy.key] != policy:
        raise ValueError(f"Policy {policy.key!r} differs from its registry.")
    if not policy.identity_features:
        raise ValueError("Target encoding requires at least one identity feature.")
    if len(policy.identity_features) != len(set(policy.identity_features)):
        raise ValueError("A target-encoding policy repeats an identity feature.")
    unknown = set(policy.identity_features) - set(ALL_IDENTITY_FEATURES)
    if unknown:
        raise ValueError(f"Unknown target-encoding identities: {unknown!r}.")

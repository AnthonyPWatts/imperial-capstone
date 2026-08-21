"""Fold-fitted funder and installer treatments for the feature-family screen."""

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


BASE_RARE_CATEGORY_MINIMUM = 20
SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_WORST_FOLD = -0.0025
SELECTION_GATE_REPAIR_RECALL = -0.02
ORGANISATION_FEATURES = ("funder", "installer")
BLANK_CATEGORY = "__blank__"
MISSING_CATEGORY = "__missing__"


@_dataclass(frozen=True)
class HighCardinalityPolicy:
    """One explicit funder/installer representation."""

    key: str
    label: str
    rationale: str
    categorical_features: tuple[str, ...] = ()
    frequency_features: tuple[str, ...] = ()
    rare_category_minimum: int = BASE_RARE_CATEGORY_MINIMUM


HIGH_CARDINALITY_POLICIES = {
    policy.key: policy
    for policy in (
        HighCardinalityPolicy(
            key="baseline",
            label="Accepted feature policy",
            rationale="No funder or installer feature",
        ),
        HighCardinalityPolicy(
            key="funder_rare20",
            label="Funder rare-grouped categorical",
            rationale="Normalised funder with fold-fitted 20-row rare grouping",
            categorical_features=("funder",),
        ),
        HighCardinalityPolicy(
            key="installer_rare20",
            label="Installer rare-grouped categorical",
            rationale="Normalised installer with fold-fitted 20-row rare grouping",
            categorical_features=("installer",),
        ),
        HighCardinalityPolicy(
            key="both_rare20",
            label="Funder and installer rare-grouped categoricals",
            rationale="Add both normalised organisation identities separately",
            categorical_features=ORGANISATION_FEATURES,
        ),
        HighCardinalityPolicy(
            key="funder_frequency",
            label="Funder frequency",
            rationale="Fold-fitted funder prevalence with unseen values mapped to zero",
            frequency_features=("funder",),
        ),
        HighCardinalityPolicy(
            key="installer_frequency",
            label="Installer frequency",
            rationale=(
                "Fold-fitted installer prevalence with unseen values mapped to zero"
            ),
            frequency_features=("installer",),
        ),
        HighCardinalityPolicy(
            key="both_frequency",
            label="Funder and installer frequencies",
            rationale="Add both fold-fitted organisation prevalence features",
            frequency_features=ORGANISATION_FEATURES,
        ),
        HighCardinalityPolicy(
            key="funder_rare20_frequency",
            label="Funder rare-grouped categorical plus frequency",
            rationale="Retain funder identity and expose its training-fold support",
            categorical_features=("funder",),
            frequency_features=("funder",),
        ),
        HighCardinalityPolicy(
            key="installer_rare20_frequency",
            label="Installer rare-grouped categorical plus frequency",
            rationale="Retain installer identity and expose its training-fold support",
            categorical_features=("installer",),
            frequency_features=("installer",),
        ),
        HighCardinalityPolicy(
            key="both_rare20_frequency",
            label="Both rare-grouped categoricals plus frequencies",
            rationale="Retain both identities and their independent support evidence",
            categorical_features=ORGANISATION_FEATURES,
            frequency_features=ORGANISATION_FEATURES,
        ),
        HighCardinalityPolicy(
            key="both_rare50_frequency",
            label="Both 50-row rare groups plus frequencies",
            rationale="Bounded sensitivity to the audit's 50-row support threshold",
            categorical_features=ORGANISATION_FEATURES,
            frequency_features=ORGANISATION_FEATURES,
            rare_category_minimum=50,
        ),
    )
}


class HighCardinalityFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Apply one policy and learn frequency mappings from the current fold."""

    def __init__(self, policy: HighCardinalityPolicy):
        self.policy = policy

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "HighCardinalityFeatureEngineer":
        _validate_policy(self.policy)
        engineer_initial_features(X)
        self.frequency_maps_ = {
            feature: normalise_organisation(X[feature]).value_counts().to_dict()
            for feature in self.policy.frequency_features
        }
        self.fit_rows_ = len(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(
            high_cardinality_model_features(self.policy),
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, ("frequency_maps_", "fit_rows_"))
        return engineer_high_cardinality_features(
            X,
            self.policy,
            frequency_maps=self.frequency_maps_,
            fit_rows=self.fit_rows_,
        )

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def normalise_organisation(values: _pd.Series) -> _pd.Series:
    """Apply conservative case and whitespace normalisation only."""

    missing = values.isna()
    normalised = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
    )
    normalised = normalised.mask(~missing & normalised.eq(""), BLANK_CATEGORY)
    return normalised.mask(missing, MISSING_CATEGORY).fillna(MISSING_CATEGORY)


def engineer_high_cardinality_features(
    X: _pd.DataFrame,
    policy: HighCardinalityPolicy,
    *,
    frequency_maps: dict[str, dict[str, int]],
    fit_rows: int,
) -> _pd.DataFrame:
    """Return the accepted frame plus one fold-fitted organisation treatment."""

    _validate_policy(policy)
    if fit_rows <= 0:
        raise ValueError("Frequency mappings require a positive fitted row count.")
    if set(frequency_maps) != set(policy.frequency_features):
        raise ValueError("Frequency mappings do not match the selected policy.")

    engineered = engineer_initial_features(X)
    normalised = {
        feature: normalise_organisation(X[feature])
        for feature in set(policy.categorical_features)
        | set(policy.frequency_features)
    }
    for feature in policy.categorical_features:
        engineered[f"{feature}_category"] = normalised[feature]
    for feature in policy.frequency_features:
        counts = normalised[feature].map(frequency_maps[feature]).fillna(0)
        engineered[f"{feature}_frequency"] = (
            counts.astype("float64") / fit_rows
        )

    expected = high_cardinality_model_features(policy)
    if tuple(engineered.columns) != expected:
        raise ValueError(
            "High-cardinality feature order changed; "
            f"expected {expected!r}, found {tuple(engineered.columns)!r}."
        )
    return engineered


def high_cardinality_model_features(
    policy: HighCardinalityPolicy,
) -> tuple[str, ...]:
    """Return the stable feature order for one policy."""

    _validate_policy(policy)
    additions = [
        *(f"{feature}_category" for feature in policy.categorical_features),
        *(f"{feature}_frequency" for feature in policy.frequency_features),
    ]
    return (*MODEL_FEATURES, *additions)


def high_cardinality_numeric_features(
    policy: HighCardinalityPolicy,
) -> tuple[str, ...]:
    """Return numeric model columns for one policy."""

    return (
        *NUMERIC_FEATURES,
        *(f"{feature}_frequency" for feature in policy.frequency_features),
    )


def high_cardinality_base_categorical_features(
    policy: HighCardinalityPolicy,
) -> tuple[str, ...]:
    """Return the unchanged initial categorical columns."""

    _validate_policy(policy)
    return CATEGORICAL_FEATURES


def high_cardinality_organisation_categorical_features(
    policy: HighCardinalityPolicy,
) -> tuple[str, ...]:
    """Return new organisation columns requiring their own rare threshold."""

    return tuple(
        f"{feature}_category" for feature in policy.categorical_features
    )


def make_high_cardinality_preprocessor(
    policy: HighCardinalityPolicy,
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build fold-fitted preprocessing for one organisation policy."""

    numeric_steps: list[tuple[str, object]] = [
        (
            "median_imputation",
            _SimpleImputer(strategy="median", add_indicator=True),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("standardisation", _StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = [
        (
            "numeric",
            _Pipeline(steps=numeric_steps),
            list(high_cardinality_numeric_features(policy)),
        ),
        (
            "base_categorical",
            _OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=BASE_RARE_CATEGORY_MINIMUM,
                sparse_output=sparse_output,
            ),
            list(high_cardinality_base_categorical_features(policy)),
        ),
    ]
    organisation_categorical = high_cardinality_organisation_categorical_features(
        policy
    )
    if organisation_categorical:
        transformers.append(
            (
                "organisation_categorical",
                _OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=policy.rare_category_minimum,
                    sparse_output=sparse_output,
                ),
                list(organisation_categorical),
            )
        )
    column_preprocessing = _ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", HighCardinalityFeatureEngineer(policy)),
            ("column_preprocessing", column_preprocessing),
        ]
    )


def summarise_high_cardinality_policies() -> _pd.DataFrame:
    """Return the pre-declared policy register in execution order."""

    rows = []
    for policy in HIGH_CARDINALITY_POLICIES.values():
        rows.append(
            {
                "policy": policy.key,
                "label": policy.label,
                "engineered_features": len(
                    high_cardinality_model_features(policy)
                ),
                "categorical_organisations": ", ".join(
                    policy.categorical_features
                ),
                "frequency_organisations": ", ".join(
                    policy.frequency_features
                ),
                "rare_category_minimum": policy.rare_category_minimum,
                "rationale": policy.rationale,
            }
        )
    return _pd.DataFrame(rows).set_index("policy")


def _validate_policy(policy: HighCardinalityPolicy) -> None:
    if policy.key not in HIGH_CARDINALITY_POLICIES:
        raise ValueError(f"Unknown high-cardinality policy: {policy.key!r}.")
    if HIGH_CARDINALITY_POLICIES[policy.key] != policy:
        raise ValueError(f"Policy {policy.key!r} differs from its registry.")
    for selected in (policy.categorical_features, policy.frequency_features):
        if len(selected) != len(set(selected)):
            raise ValueError("A policy repeats an organisation feature.")
        unknown = set(selected) - set(ORGANISATION_FEATURES)
        if unknown:
            raise ValueError(f"Unknown organisation features: {unknown!r}.")
    if policy.rare_category_minimum < 2:
        raise ValueError("Rare-category minimum must be at least two rows.")

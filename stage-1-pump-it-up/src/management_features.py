"""Management-family hierarchy policies for the fixed-model feature screen."""

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
MANAGEMENT_SOURCE_FEATURES = ("management", "scheme_management")
MANAGEMENT_GROUP_FEATURE = "management_group"
MANAGEMENT_SCHEME_COMPOSITE = "management_scheme_composite"
MANAGEMENT_SCHEME_RELATIONSHIP = "management_scheme_relationship"
_MISSING_CATEGORY = "__missing__"
_ADDED_FEATURE_ORDER = (
    MANAGEMENT_GROUP_FEATURE,
    MANAGEMENT_SCHEME_COMPOSITE,
    MANAGEMENT_SCHEME_RELATIONSHIP,
)


@_dataclass(frozen=True)
class ManagementFeaturePolicy:
    """One explicit representation of the management hierarchy."""

    key: str
    label: str
    rationale: str
    remove_features: tuple[str, ...] = ()
    add_management_group: bool = False
    add_composite: bool = False
    add_relationship: bool = False


MANAGEMENT_POLICIES = {
    policy.key: policy
    for policy in (
        ManagementFeaturePolicy(
            key="baseline",
            label="Management plus scheme management",
            rationale="Retain the accepted two-source management representation",
        ),
        ManagementFeaturePolicy(
            key="management_only",
            label="Management only",
            rationale="Remove the incomplete parallel scheme field",
            remove_features=("scheme_management",),
        ),
        ManagementFeaturePolicy(
            key="scheme_only",
            label="Scheme management only",
            rationale="Measure the main management field's independent value",
            remove_features=("management",),
        ),
        ManagementFeaturePolicy(
            key="management_group_only",
            label="Management group only",
            rationale="Replace both source fields with the deterministic coarse group",
            remove_features=MANAGEMENT_SOURCE_FEATURES,
            add_management_group=True,
        ),
        ManagementFeaturePolicy(
            key="management_plus_group",
            label="Management plus group",
            rationale="Remove scheme management and expose its coarse alternative",
            remove_features=("scheme_management",),
            add_management_group=True,
        ),
        ManagementFeaturePolicy(
            key="scheme_plus_group",
            label="Scheme management plus group",
            rationale="Pair the incomplete scheme field with a complete coarse back-off",
            remove_features=("management",),
            add_management_group=True,
        ),
        ManagementFeaturePolicy(
            key="all_three",
            label="Management, scheme and group",
            rationale="Test whether the deterministic coarse group helps the fixed trees",
            add_management_group=True,
        ),
        ManagementFeaturePolicy(
            key="composite_only",
            label="Management-scheme composite only",
            rationale="Replace separate sources with their normalised joint state",
            remove_features=MANAGEMENT_SOURCE_FEATURES,
            add_composite=True,
        ),
        ManagementFeaturePolicy(
            key="baseline_plus_composite",
            label="Baseline plus management-scheme composite",
            rationale="Expose the source interaction while retaining both main effects",
            add_composite=True,
        ),
        ManagementFeaturePolicy(
            key="baseline_plus_relationship",
            label="Baseline plus relationship state",
            rationale="Distinguish matching, conflicting and missing scheme labels",
            add_relationship=True,
        ),
    )
}


class ManagementFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Apply one deterministic management-family representation."""

    def __init__(self, policy: ManagementFeaturePolicy):
        self.policy = policy

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "ManagementFeatureEngineer":
        _validate_policy(self.policy)
        engineer_initial_features(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(
            management_model_features(self.policy),
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "feature_names_out_")
        return engineer_management_features(X, self.policy)

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def engineer_management_features(
    X: _pd.DataFrame,
    policy: ManagementFeaturePolicy,
) -> _pd.DataFrame:
    """Return the accepted frame with one management hierarchy treatment."""

    _validate_policy(policy)
    engineered = engineer_initial_features(X)
    management = _normalise_category(X["management"])
    scheme = _normalise_category(X["scheme_management"])
    if policy.remove_features:
        engineered = engineered.drop(columns=list(policy.remove_features))
    if policy.add_management_group:
        engineered[MANAGEMENT_GROUP_FEATURE] = _normalise_category(
            X["management_group"]
        )
    if policy.add_composite:
        engineered[MANAGEMENT_SCHEME_COMPOSITE] = management + "::" + scheme
    if policy.add_relationship:
        relationship = _pd.Series("different", index=X.index, dtype="string")
        relationship.loc[scheme.eq(_MISSING_CATEGORY)] = "scheme_missing"
        relationship.loc[management.eq(scheme)] = "same"
        engineered[MANAGEMENT_SCHEME_RELATIONSHIP] = relationship

    expected = management_model_features(policy)
    if tuple(engineered.columns) != expected:
        raise ValueError(
            "Management feature order changed; "
            f"expected {expected!r}, found {tuple(engineered.columns)!r}."
        )
    return engineered


def management_model_features(
    policy: ManagementFeaturePolicy,
) -> tuple[str, ...]:
    """Return the stable feature order for one management policy."""

    _validate_policy(policy)
    features = [name for name in MODEL_FEATURES if name not in policy.remove_features]
    selected_additions = {
        MANAGEMENT_GROUP_FEATURE: policy.add_management_group,
        MANAGEMENT_SCHEME_COMPOSITE: policy.add_composite,
        MANAGEMENT_SCHEME_RELATIONSHIP: policy.add_relationship,
    }
    features.extend(name for name in _ADDED_FEATURE_ORDER if selected_additions[name])
    return tuple(features)


def management_categorical_features(
    policy: ManagementFeaturePolicy,
) -> tuple[str, ...]:
    """Return categorical columns retained or created by one policy."""

    features = set(management_model_features(policy))
    return tuple(
        name
        for name in (*CATEGORICAL_FEATURES, *_ADDED_FEATURE_ORDER)
        if name in features
    )


def make_management_preprocessor(
    policy: ManagementFeaturePolicy,
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build fold-fitted preprocessing for one management policy."""

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
                list(NUMERIC_FEATURES),
            ),
            (
                "categorical",
                _OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=RARE_CATEGORY_MINIMUM,
                    sparse_output=sparse_output,
                ),
                list(management_categorical_features(policy)),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", ManagementFeatureEngineer(policy)),
            ("column_preprocessing", column_preprocessing),
        ]
    )


def summarise_management_policies() -> _pd.DataFrame:
    """Return the predeclared management-family screen in execution order."""

    rows = []
    for policy in MANAGEMENT_POLICIES.values():
        rows.append(
            {
                "policy": policy.key,
                "label": policy.label,
                "engineered_features": len(management_model_features(policy)),
                "removed_features": ", ".join(policy.remove_features),
                "add_management_group": policy.add_management_group,
                "add_composite": policy.add_composite,
                "add_relationship": policy.add_relationship,
                "rationale": policy.rationale,
            }
        )
    return _pd.DataFrame(rows).set_index("policy")


def _normalise_category(values: _pd.Series) -> _pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.casefold()
        .replace("", _pd.NA)
        .fillna(_MISSING_CATEGORY)
    )


def _validate_policy(policy: ManagementFeaturePolicy) -> None:
    if policy.key not in MANAGEMENT_POLICIES:
        raise ValueError(f"Unknown management policy: {policy.key!r}.")
    if MANAGEMENT_POLICIES[policy.key] != policy:
        raise ValueError(f"Policy {policy.key!r} differs from its registry.")
    if len(policy.remove_features) != len(set(policy.remove_features)):
        raise ValueError("A management policy repeats a removed feature.")
    unknown = set(policy.remove_features) - set(MANAGEMENT_SOURCE_FEATURES)
    if unknown:
        raise ValueError(f"A management policy removes unknown features: {unknown!r}.")

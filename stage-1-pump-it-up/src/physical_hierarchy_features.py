"""Deterministic physical-category hierarchy policies for bounded ablation."""

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
PARSIMONY_MAXIMUM_ACCURACY_LOSS = -0.0005
_MISSING_CATEGORY = "__missing__"


@_dataclass(frozen=True)
class HierarchyFamily:
    """One deterministic child-to-parent physical category family."""

    key: str
    label: str
    features: tuple[str, ...]
    current_feature: str


@_dataclass(frozen=True)
class PhysicalHierarchyPolicy:
    """One explicit retained-level policy within a physical hierarchy."""

    family: str
    key: str
    label: str
    rationale: str
    selected_features: tuple[str, ...]


HIERARCHY_FAMILIES = {
    family.key: family
    for family in (
        HierarchyFamily(
            key="extraction",
            label="Extraction type",
            features=(
                "extraction_type",
                "extraction_type_group",
                "extraction_type_class",
            ),
            current_feature="extraction_type",
        ),
        HierarchyFamily(
            key="source",
            label="Source",
            features=("source", "source_type", "source_class"),
            current_feature="source",
        ),
        HierarchyFamily(
            key="quality",
            label="Water quality",
            features=("water_quality", "quality_group"),
            current_feature="water_quality",
        ),
        HierarchyFamily(
            key="waterpoint",
            label="Waterpoint type",
            features=("waterpoint_type", "waterpoint_type_group"),
            current_feature="waterpoint_type",
        ),
    )
}


def _policy(
    family: str,
    key: str,
    label: str,
    rationale: str,
    *selected_features: str,
) -> PhysicalHierarchyPolicy:
    return PhysicalHierarchyPolicy(
        family=family,
        key=key,
        label=label,
        rationale=rationale,
        selected_features=selected_features,
    )


PHYSICAL_HIERARCHY_POLICIES = {
    "extraction": {
        policy.key: policy
        for policy in (
            _policy(
                "extraction",
                "baseline",
                "Granular extraction type",
                "Retain the accepted granular extraction representation",
                "extraction_type",
            ),
            _policy(
                "extraction",
                "none",
                "No extraction hierarchy",
                "Measure the complete hierarchy's independent value",
            ),
            _policy(
                "extraction",
                "group_only",
                "Extraction group only",
                "Replace the granular type with its intermediate parent",
                "extraction_type_group",
            ),
            _policy(
                "extraction",
                "class_only",
                "Extraction class only",
                "Replace the granular type with its broad parent",
                "extraction_type_class",
            ),
            _policy(
                "extraction",
                "type_plus_group",
                "Extraction type plus group",
                "Retain detail and add the intermediate deterministic parent",
                "extraction_type",
                "extraction_type_group",
            ),
            _policy(
                "extraction",
                "type_plus_class",
                "Extraction type plus class",
                "Retain detail and add the broad deterministic parent",
                "extraction_type",
                "extraction_type_class",
            ),
            _policy(
                "extraction",
                "group_plus_class",
                "Extraction group plus class",
                "Remove the child while retaining both parent levels",
                "extraction_type_group",
                "extraction_type_class",
            ),
            _policy(
                "extraction",
                "all_levels",
                "All extraction levels",
                "Measure whether both deterministic parents help the fixed trees",
                "extraction_type",
                "extraction_type_group",
                "extraction_type_class",
            ),
        )
    },
    "source": {
        policy.key: policy
        for policy in (
            _policy(
                "source",
                "baseline",
                "Granular source",
                "Retain the accepted granular source representation",
                "source",
            ),
            _policy(
                "source",
                "none",
                "No source hierarchy",
                "Measure the complete hierarchy's independent value",
            ),
            _policy(
                "source",
                "type_only",
                "Source type only",
                "Replace the granular source with its intermediate parent",
                "source_type",
            ),
            _policy(
                "source",
                "class_only",
                "Source class only",
                "Replace the granular source with its broad parent",
                "source_class",
            ),
            _policy(
                "source",
                "source_plus_type",
                "Source plus type",
                "Retain detail and add the intermediate deterministic parent",
                "source",
                "source_type",
            ),
            _policy(
                "source",
                "source_plus_class",
                "Source plus class",
                "Retain detail and add the broad deterministic parent",
                "source",
                "source_class",
            ),
            _policy(
                "source",
                "type_plus_class",
                "Source type plus class",
                "Remove the child while retaining both parent levels",
                "source_type",
                "source_class",
            ),
            _policy(
                "source",
                "all_levels",
                "All source levels",
                "Measure whether both deterministic parents help the fixed trees",
                "source",
                "source_type",
                "source_class",
            ),
        )
    },
    "quality": {
        policy.key: policy
        for policy in (
            _policy(
                "quality",
                "baseline",
                "Granular water quality",
                "Retain the accepted granular quality representation",
                "water_quality",
            ),
            _policy(
                "quality",
                "none",
                "No quality hierarchy",
                "Measure the complete hierarchy's independent value",
            ),
            _policy(
                "quality",
                "group_only",
                "Quality group only",
                "Replace granular quality with its deterministic parent",
                "quality_group",
            ),
            _policy(
                "quality",
                "both_levels",
                "Water quality plus group",
                "Retain detail and add the deterministic parent",
                "water_quality",
                "quality_group",
            ),
        )
    },
    "waterpoint": {
        policy.key: policy
        for policy in (
            _policy(
                "waterpoint",
                "baseline",
                "Granular waterpoint type",
                "Retain the accepted granular waterpoint representation",
                "waterpoint_type",
            ),
            _policy(
                "waterpoint",
                "none",
                "No waterpoint hierarchy",
                "Measure the complete hierarchy's independent value",
            ),
            _policy(
                "waterpoint",
                "group_only",
                "Waterpoint group only",
                "Replace granular waterpoint type with its deterministic parent",
                "waterpoint_type_group",
            ),
            _policy(
                "waterpoint",
                "both_levels",
                "Waterpoint type plus group",
                "Retain detail and add the deterministic parent",
                "waterpoint_type",
                "waterpoint_type_group",
            ),
        )
    },
}


class PhysicalHierarchyFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Apply one deterministic physical-hierarchy representation."""

    def __init__(self, policy: PhysicalHierarchyPolicy):
        self.policy = policy

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "PhysicalHierarchyFeatureEngineer":
        _validate_policy(self.policy)
        engineer_initial_features(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(
            hierarchy_model_features(self.policy),
            dtype=object,
        )
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "feature_names_out_")
        return engineer_physical_hierarchy_features(X, self.policy)

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def engineer_physical_hierarchy_features(
    X: _pd.DataFrame,
    policy: PhysicalHierarchyPolicy,
) -> _pd.DataFrame:
    """Return the accepted frame with one hierarchy-level treatment."""

    _validate_policy(policy)
    family = HIERARCHY_FAMILIES[policy.family]
    engineered = engineer_initial_features(X)
    if family.current_feature not in policy.selected_features:
        engineered = engineered.drop(columns=family.current_feature)
    for feature in policy.selected_features:
        if feature not in engineered:
            engineered[feature] = _normalise_category(X[feature])

    expected = hierarchy_model_features(policy)
    if tuple(engineered.columns) != expected:
        raise ValueError(
            "Physical hierarchy feature order changed; "
            f"expected {expected!r}, found {tuple(engineered.columns)!r}."
        )
    return engineered


def hierarchy_model_features(
    policy: PhysicalHierarchyPolicy,
) -> tuple[str, ...]:
    """Return the stable feature order for one hierarchy policy."""

    _validate_policy(policy)
    family = HIERARCHY_FAMILIES[policy.family]
    features = list(MODEL_FEATURES)
    if family.current_feature not in policy.selected_features:
        features.remove(family.current_feature)
    features.extend(
        feature
        for feature in policy.selected_features
        if feature != family.current_feature
    )
    return tuple(features)


def hierarchy_categorical_features(
    policy: PhysicalHierarchyPolicy,
) -> tuple[str, ...]:
    """Return categorical columns retained or introduced by one policy."""

    features = set(hierarchy_model_features(policy))
    family = HIERARCHY_FAMILIES[policy.family]
    ordered_candidates = dict.fromkeys((*CATEGORICAL_FEATURES, *family.features))
    return tuple(
        name
        for name in ordered_candidates
        if name in features
    )


def make_physical_hierarchy_preprocessor(
    policy: PhysicalHierarchyPolicy,
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build fold-fitted preprocessing for one hierarchy policy."""

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
                list(hierarchy_categorical_features(policy)),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", PhysicalHierarchyFeatureEngineer(policy)),
            ("column_preprocessing", column_preprocessing),
        ]
    )


def summarise_physical_hierarchy_policies(family_key: str) -> _pd.DataFrame:
    """Return one family's predeclared ablation register."""

    if family_key not in PHYSICAL_HIERARCHY_POLICIES:
        raise ValueError(f"Unknown hierarchy family: {family_key!r}.")
    rows = []
    for policy in PHYSICAL_HIERARCHY_POLICIES[family_key].values():
        rows.append(
            {
                "policy": policy.key,
                "label": policy.label,
                "engineered_features": len(hierarchy_model_features(policy)),
                "selected_features": ", ".join(policy.selected_features),
                "rationale": policy.rationale,
            }
        )
    return _pd.DataFrame(rows).set_index("policy")


def _normalise_category(values: _pd.Series) -> _pd.Series:
    normalised = values.astype("string").str.strip()
    return normalised.mask(normalised.eq("")).fillna(_MISSING_CATEGORY)


def _validate_policy(policy: PhysicalHierarchyPolicy) -> None:
    if policy.family not in PHYSICAL_HIERARCHY_POLICIES:
        raise ValueError(f"Unknown hierarchy family: {policy.family!r}.")
    registered = PHYSICAL_HIERARCHY_POLICIES[policy.family]
    if policy.key not in registered or registered[policy.key] != policy:
        raise ValueError(
            f"Unknown or modified {policy.family} policy: {policy.key!r}."
        )
    if len(policy.selected_features) != len(set(policy.selected_features)):
        raise ValueError("A hierarchy policy repeats a selected feature.")
    family_features = set(HIERARCHY_FAMILIES[policy.family].features)
    unknown = set(policy.selected_features) - family_features
    if unknown:
        raise ValueError(f"A hierarchy policy selects unknown features: {unknown!r}.")

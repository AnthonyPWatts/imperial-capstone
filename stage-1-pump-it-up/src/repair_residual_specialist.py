"""Conservative repair-class corrections behind the selected ensemble."""

from __future__ import annotations

from collections.abc import Mapping as _Mapping

import numpy as _np
import pandas as _pd
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.pipeline import make_pipeline as _make_pipeline
from sklearn.preprocessing import StandardScaler as _StandardScaler

from final_model import CLASS_LABELS
from final_model import validate_probabilities
from target_encoding_features import normalise_identity


REPAIR_META_COMPONENTS = (
    "deep_xgboost",
    "spatial_xgboost",
    "random_forest",
    "frequency_random_forest",
    "spatial_random_forest",
    "identity_catboost",
)
REPAIR_META_THRESHOLD = 0.70
REPAIR_META_C = 0.10
REPAIR_HISTORY_COLUMN = "subvillage"
REPAIR_HISTORY_MINIMUM_SUPPORT = 20
REPAIR_HISTORY_MINIMUM_RATE = 0.40

_FUNCTIONAL_POSITION = CLASS_LABELS.index("functional")
_REPAIR_POSITION = CLASS_LABELS.index("functional needs repair")


def build_repair_meta_features(
    components: _Mapping[str, _np.ndarray],
    blended_probabilities: _np.ndarray,
) -> _np.ndarray:
    """Create a fixed residual feature matrix from aligned model probabilities."""

    missing = set(REPAIR_META_COMPONENTS).difference(components)
    if missing:
        raise KeyError(f"Missing repair-meta components: {sorted(missing)!r}.")
    blend = _np.asarray(blended_probabilities, dtype="float64")
    validate_probabilities(blend, len(blend))
    ordered_components = []
    for name in REPAIR_META_COMPONENTS:
        values = _np.asarray(components[name], dtype="float64")
        validate_probabilities(values, len(blend))
        ordered_components.append(values)

    repair_votes = _np.column_stack(
        [values.argmax(axis=1) == _REPAIR_POSITION for values in ordered_components]
    ).sum(axis=1)
    repair_minus_functional = _np.column_stack(
        [
            values[:, _REPAIR_POSITION] - values[:, _FUNCTIONAL_POSITION]
            for values in ordered_components
        ]
    )
    repair_memberships = _np.column_stack(
        [values[:, _REPAIR_POSITION] for values in ordered_components]
    )
    denominator = _np.maximum(blend[:, _FUNCTIONAL_POSITION], 1e-6)
    derived = _np.column_stack(
        [
            blend,
            blend[:, _REPAIR_POSITION] - blend[:, _FUNCTIONAL_POSITION],
            blend[:, _REPAIR_POSITION] / denominator,
            repair_votes,
            repair_minus_functional.max(axis=1),
            repair_memberships.std(axis=1),
        ]
    )
    result = _np.column_stack([*ordered_components, derived])
    if result.shape != (len(blend), 26):
        raise ValueError(f"Unexpected repair-meta feature shape: {result.shape!r}.")
    if not _np.isfinite(result).all():
        raise ValueError("Repair-meta features must all be finite.")
    return result


def make_repair_meta_classifier():
    """Return the fixed, deliberately simple residual classifier."""

    return _make_pipeline(
        _StandardScaler(),
        _LogisticRegression(C=REPAIR_META_C, max_iter=2_000),
    )


def operational_training_mask(target: _pd.Series | _np.ndarray) -> _np.ndarray:
    """Select only functional and repair labels for the residual boundary."""

    values = _np.asarray(target, dtype=object)
    return _np.isin(
        values,
        ("functional", "functional needs repair"),
    )


def binary_repair_target(target: _pd.Series | _np.ndarray) -> _np.ndarray:
    """Map the operational labels to zero for functional and one for repair."""

    values = _np.asarray(target, dtype=object)
    if not operational_training_mask(values).all():
        raise ValueError("Binary repair target contains a non-operational label.")
    return (values == "functional needs repair").astype("int8")


def overlay_repair_probabilities(
    base_probabilities: _np.ndarray,
    repair_probabilities: _np.ndarray,
    *,
    threshold: float = REPAIR_META_THRESHOLD,
    additional_mask: _np.ndarray | None = None,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Swap functional and repair memberships for selected functional rows."""

    base = _np.asarray(base_probabilities, dtype="float64")
    validate_probabilities(base, len(base))
    repair = _np.asarray(repair_probabilities, dtype="float64")
    if repair.shape != (len(base),):
        raise ValueError("Repair-specialist probabilities have the wrong shape.")
    if not _np.isfinite(repair).all() or (repair < 0).any() or (repair > 1).any():
        raise ValueError("Repair-specialist probabilities must lie in [0, 1].")
    if not 0 < threshold <= 1:
        raise ValueError("Repair-specialist threshold must lie in (0, 1].")

    selected = (base.argmax(axis=1) == _FUNCTIONAL_POSITION) & (
        repair >= threshold
    )
    if additional_mask is not None:
        additional = _np.asarray(additional_mask, dtype=bool)
        if additional.shape != (len(base),):
            raise ValueError("Additional repair mask has the wrong shape.")
        selected |= (
            base.argmax(axis=1) == _FUNCTIONAL_POSITION
        ) & additional

    overlaid = base.copy()
    functional_values = overlaid[selected, _FUNCTIONAL_POSITION].copy()
    overlaid[selected, _FUNCTIONAL_POSITION] = overlaid[
        selected,
        _REPAIR_POSITION,
    ]
    overlaid[selected, _REPAIR_POSITION] = functional_values
    validate_probabilities(overlaid, len(base))
    if selected.any() and not _np.all(
        overlaid[selected].argmax(axis=1) == _REPAIR_POSITION
    ):
        raise ValueError("Repair overlay did not change every selected decision.")
    return overlaid, selected


def repair_history_mask(
    X_training: _pd.DataFrame,
    y_training: _pd.Series | _np.ndarray,
    X_prediction: _pd.DataFrame,
    *,
    column: str = REPAIR_HISTORY_COLUMN,
    minimum_support: int = REPAIR_HISTORY_MINIMUM_SUPPORT,
    minimum_rate: float = REPAIR_HISTORY_MINIMUM_RATE,
) -> _np.ndarray:
    """Flag groups with strong, supported repair-over-functional history."""

    if column not in X_training or column not in X_prediction:
        raise KeyError(f"Repair-history column is missing: {column!r}.")
    if minimum_support <= 0:
        raise ValueError("Repair-history support must be positive.")
    if not 0 < minimum_rate < 1:
        raise ValueError("Repair-history rate must lie in (0, 1).")
    target = _np.asarray(y_training, dtype=object)
    if len(target) != len(X_training):
        raise ValueError("Repair-history training rows and labels are misaligned.")

    groups = normalise_identity(X_training[column]).reset_index(drop=True)
    counts = _pd.DataFrame(
        {
            "group": groups,
            "repair": target == "functional needs repair",
            "functional": target == "functional",
        }
    ).groupby("group", observed=True)[["repair", "functional"]].sum()
    counts["support"] = counts["repair"] + counts["functional"]
    counts["repair_rate"] = (counts["repair"] + 1) / (
        counts["support"] + 2
    )
    eligible = counts.index[
        counts["support"].ge(minimum_support)
        & counts["repair_rate"].ge(minimum_rate)
    ]
    prediction_groups = normalise_identity(X_prediction[column])
    return prediction_groups.isin(eligible).to_numpy(dtype=bool)

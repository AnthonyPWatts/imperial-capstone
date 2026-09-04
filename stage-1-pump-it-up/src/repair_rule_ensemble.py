"""Fold-safe, high-precision repair overrides for the selected ensemble."""

from __future__ import annotations

from collections.abc import Sequence as _Sequence
from dataclasses import dataclass as _dataclass

import numpy as _np
import pandas as _pd

from final_model import CLASS_LABELS
from final_model import validate_probabilities


FUNCTIONAL_LABEL = "functional"
REPAIR_LABEL = "functional needs repair"
NON_FUNCTIONAL_LABEL = "non functional"

FUNCTIONAL_POSITION = CLASS_LABELS.index(FUNCTIONAL_LABEL)
REPAIR_POSITION = CLASS_LABELS.index(REPAIR_LABEL)

IDENTITY_CONFLICT_RULE = "identity_catboost_conflict"
IDENTITY_REPAIR_TO_FUNCTIONAL_RATIO = 2.0


@_dataclass(frozen=True)
class RepairHistoryRule:
    """One fixed target-history rule and its probability guardrail."""

    name: str
    key: str
    minimum_repair_rate: float
    minimum_support: int
    minimum_repair_to_functional_ratio: float


REPAIR_HISTORY_RULES = (
    RepairHistoryRule(
        name="scheme_name_history",
        key="scheme_name",
        minimum_repair_rate=0.35,
        minimum_support=50,
        minimum_repair_to_functional_ratio=0.60,
    ),
    RepairHistoryRule(
        name="grid_005_history",
        key="grid_005",
        minimum_repair_rate=0.75,
        minimum_support=3,
        minimum_repair_to_functional_ratio=0.90,
    ),
    RepairHistoryRule(
        name="lga_subvillage_history",
        key="lga_subvillage",
        minimum_repair_rate=1.00,
        minimum_support=3,
        minimum_repair_to_functional_ratio=0.50,
    ),
    RepairHistoryRule(
        name="exact_coordinate_name_history",
        key="exact_coordinate_name",
        minimum_repair_rate=0.45,
        minimum_support=1,
        minimum_repair_to_functional_ratio=0.90,
    ),
    RepairHistoryRule(
        name="subvillage_history",
        key="subvillage",
        minimum_repair_rate=1.00,
        minimum_support=3,
        minimum_repair_to_functional_ratio=0.50,
    ),
)

HISTORY_RULE_NAMES = tuple(rule.name for rule in REPAIR_HISTORY_RULES)
STRICT_HISTORY_RULE_NAMES = tuple(
    name for name in HISTORY_RULE_NAMES if name != "scheme_name_history"
)
FULL_RULE_NAMES = (*HISTORY_RULE_NAMES, IDENTITY_CONFLICT_RULE)


def build_repair_rule_masks(
    X_training: _pd.DataFrame,
    y_training: _pd.Series | _np.ndarray,
    X_prediction: _pd.DataFrame,
    base_probabilities: _np.ndarray,
    identity_probabilities: _np.ndarray,
) -> _pd.DataFrame:
    """Fit the fixed history mappings and flag guarded prediction rows."""

    target = _validate_training_data(X_training, y_training)
    base = _np.asarray(base_probabilities, dtype="float64")
    identity = _np.asarray(identity_probabilities, dtype="float64")
    validate_probabilities(base, len(X_prediction))
    validate_probabilities(identity, len(X_prediction))

    training_keys = engineer_repair_rule_keys(X_training)
    prediction_keys = engineer_repair_rule_keys(X_prediction)
    functional_prediction = base.argmax(axis=1) == FUNCTIONAL_POSITION
    base_ratio = _safe_ratio(
        base[:, REPAIR_POSITION],
        base[:, FUNCTIONAL_POSITION],
    )
    masks: dict[str, _np.ndarray] = {}
    for rule in REPAIR_HISTORY_RULES:
        mapping = fit_repair_history_mapping(
            training_keys[rule.key],
            target,
        )
        repair_rate = (
            prediction_keys[rule.key]
            .map(mapping["repair_rate"])
            .fillna(0.0)
            .to_numpy(dtype="float64")
        )
        support = (
            prediction_keys[rule.key]
            .map(mapping["support"])
            .fillna(0)
            .to_numpy(dtype="int64")
        )
        masks[rule.name] = (
            functional_prediction
            & (repair_rate >= rule.minimum_repair_rate)
            & (support >= rule.minimum_support)
            & (
                base_ratio
                >= rule.minimum_repair_to_functional_ratio
            )
        )

    identity_ratio = _safe_ratio(
        identity[:, REPAIR_POSITION],
        identity[:, FUNCTIONAL_POSITION],
    )
    masks[IDENTITY_CONFLICT_RULE] = (
        functional_prediction
        & (identity_ratio >= IDENTITY_REPAIR_TO_FUNCTIONAL_RATIO)
    )
    result = _pd.DataFrame(masks, index=_pd.RangeIndex(len(X_prediction)))
    _validate_rule_masks(result, expected_rows=len(X_prediction))
    return result


def cross_fit_repair_rule_masks(
    X: _pd.DataFrame,
    y: _pd.Series | _np.ndarray,
    validation_folds: _pd.Series | _np.ndarray,
    base_probabilities: _np.ndarray,
    identity_probabilities: _np.ndarray,
) -> _pd.DataFrame:
    """Create OOF rule masks with mappings fitted outside each validation fold."""

    target = _validate_training_data(X, y)
    folds = _np.asarray(validation_folds)
    if folds.shape != (len(X),) or _pd.isna(folds).any():
        raise ValueError("Validation-fold assignments are incomplete or misaligned.")
    unique_folds = _np.unique(folds)
    if len(unique_folds) < 2:
        raise ValueError("Cross-fitting requires at least two validation folds.")

    base = _np.asarray(base_probabilities, dtype="float64")
    identity = _np.asarray(identity_probabilities, dtype="float64")
    validate_probabilities(base, len(X))
    validate_probabilities(identity, len(X))
    output = _pd.DataFrame(False, index=_pd.RangeIndex(len(X)), columns=FULL_RULE_NAMES)
    for fold in unique_folds:
        training = folds != fold
        validation = folds == fold
        fold_masks = build_repair_rule_masks(
            X.iloc[training],
            target[training],
            X.iloc[validation],
            base[validation],
            identity[validation],
        )
        output.loc[validation, :] = fold_masks.to_numpy(dtype=bool)
    _validate_rule_masks(output, expected_rows=len(X))
    return output


def engineer_repair_rule_keys(X: _pd.DataFrame) -> _pd.DataFrame:
    """Return the five fixed identity and geographic lookup keys."""

    required = {
        "longitude",
        "latitude",
        "scheme_name",
        "lga",
        "subvillage",
        "wpt_name",
    }
    missing = required.difference(X.columns)
    if missing:
        raise KeyError(f"Missing repair-rule source columns: {sorted(missing)!r}.")

    longitude = _numeric_key(X["longitude"], multiplier=1.0, decimals=6)
    latitude = _numeric_key(X["latitude"], multiplier=1.0, decimals=6)
    grid_longitude = _integer_grid_key(X["longitude"], multiplier=20.0)
    grid_latitude = _integer_grid_key(X["latitude"], multiplier=20.0)
    scheme = _string_key(X["scheme_name"])
    lga = _string_key(X["lga"])
    subvillage = _string_key(X["subvillage"])
    waterpoint_name = _string_key(X["wpt_name"])
    return _pd.DataFrame(
        {
            "scheme_name": scheme,
            "grid_005": grid_longitude.str.cat(grid_latitude, sep="|"),
            "lga_subvillage": lga.str.cat(subvillage, sep="|"),
            "exact_coordinate_name": (
                longitude.str.cat(latitude, sep="|")
                .str.cat(waterpoint_name, sep="|")
            ),
            "subvillage": subvillage,
        }
    ).reset_index(drop=True)


def fit_repair_history_mapping(
    keys: _pd.Series,
    target: _pd.Series | _np.ndarray,
) -> _pd.DataFrame:
    """Count repair and functional labels; non-functional rows add no support."""

    values = _np.asarray(target, dtype=object)
    if values.shape != (len(keys),):
        raise ValueError("Repair-history keys and labels are misaligned.")
    unexpected = set(_pd.unique(values)).difference(CLASS_LABELS)
    if unexpected:
        raise ValueError(f"Repair-history labels are invalid: {sorted(unexpected)!r}.")
    counts = (
        _pd.DataFrame(
            {
                "key": _string_key(keys).reset_index(drop=True),
                "repair": values == REPAIR_LABEL,
                "functional": values == FUNCTIONAL_LABEL,
            }
        )
        .groupby("key", observed=True)[["repair", "functional"]]
        .sum()
        .astype("int64")
    )
    counts["support"] = counts["repair"] + counts["functional"]
    counts["repair_rate"] = counts["repair"].div(
        counts["support"].replace(0, _np.nan)
    )
    return counts


def overlay_repair_rule_union(
    base_probabilities: _np.ndarray,
    rule_masks: _pd.DataFrame,
    *,
    rule_names: _Sequence[str] = FULL_RULE_NAMES,
    additional_mask: _np.ndarray | _pd.Series | None = None,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Swap memberships for a rule union and an optional external mask."""

    base = _np.asarray(base_probabilities, dtype="float64")
    validate_probabilities(base, len(base))
    _validate_rule_masks(rule_masks, expected_rows=len(base))
    selected_names = tuple(rule_names)
    if not selected_names:
        raise ValueError("At least one repair rule must be selected.")
    unknown = set(selected_names).difference(rule_masks.columns)
    if unknown:
        raise KeyError(f"Unknown repair rules: {sorted(unknown)!r}.")
    selected = (
        rule_masks.loc[:, list(selected_names)]
        .any(axis=1)
        .to_numpy(copy=True)
    )
    functional_prediction = base.argmax(axis=1) == FUNCTIONAL_POSITION
    if additional_mask is not None:
        additional = _np.asarray(additional_mask, dtype=bool)
        if additional.shape != (len(base),):
            raise ValueError("The additional repair mask has the wrong shape.")
        selected |= functional_prediction & additional
    if (selected & ~functional_prediction).any():
        raise ValueError("Repair rules may only select functional predictions.")

    overlaid = base.copy()
    functional_membership = overlaid[selected, FUNCTIONAL_POSITION].copy()
    overlaid[selected, FUNCTIONAL_POSITION] = overlaid[selected, REPAIR_POSITION]
    overlaid[selected, REPAIR_POSITION] = functional_membership
    validate_probabilities(overlaid, len(base))
    if selected.any() and not _np.all(
        overlaid[selected].argmax(axis=1) == REPAIR_POSITION
    ):
        raise ValueError("A selected repair override did not change its decision.")
    return overlaid, selected


def summarise_rule_contributions(
    rule_masks: _pd.DataFrame,
    *,
    target: _pd.Series | _np.ndarray | None = None,
    rule_order: _Sequence[str] = FULL_RULE_NAMES,
) -> _pd.DataFrame:
    """Report standalone, unique and fixed-order incremental rule coverage."""

    _validate_rule_masks(rule_masks, expected_rows=len(rule_masks))
    names = tuple(rule_order)
    unknown = set(names).difference(rule_masks.columns)
    if unknown:
        raise KeyError(f"Unknown repair rules: {sorted(unknown)!r}.")
    labels = None if target is None else _np.asarray(target, dtype=object)
    if labels is not None and labels.shape != (len(rule_masks),):
        raise ValueError("Rule masks and contribution labels are misaligned.")

    selected_so_far = _np.zeros(len(rule_masks), dtype=bool)
    rows = []
    all_values = rule_masks.loc[:, list(names)].to_numpy(dtype=bool)
    for position, name in enumerate(names):
        standalone = rule_masks[name].to_numpy(dtype=bool)
        unique = standalone & (all_values.sum(axis=1) == 1)
        incremental = standalone & ~selected_so_far
        row: dict[str, object] = {
            "rule": name,
            "standalone_triggers": int(standalone.sum()),
            "unique_triggers": int(unique.sum()),
            "incremental_triggers": int(incremental.sum()),
            "overlaps_prior_rules": int((standalone & selected_so_far).sum()),
            "rule_order": position + 1,
        }
        if labels is not None:
            row.update(_label_change_counts("standalone", standalone, labels))
            row.update(_label_change_counts("unique", unique, labels))
            row.update(_label_change_counts("incremental", incremental, labels))
        rows.append(row)
        selected_so_far |= standalone
    return _pd.DataFrame(rows).set_index("rule")


def rule_overlap_counts(
    rule_masks: _pd.DataFrame,
    *,
    rule_names: _Sequence[str] = FULL_RULE_NAMES,
) -> _pd.DataFrame:
    """Return symmetric pairwise trigger-overlap counts."""

    _validate_rule_masks(rule_masks, expected_rows=len(rule_masks))
    names = tuple(rule_names)
    unknown = set(names).difference(rule_masks.columns)
    if unknown:
        raise KeyError(f"Unknown repair rules: {sorted(unknown)!r}.")
    values = rule_masks.loc[:, list(names)].to_numpy(dtype="int64")
    return _pd.DataFrame(values.T @ values, index=names, columns=names)


def selected_rule_names(mask_row: _pd.Series) -> str:
    """Format the triggered rules for one row in declared order."""

    return ";".join(name for name in FULL_RULE_NAMES if bool(mask_row[name]))


def _validate_training_data(
    X: _pd.DataFrame,
    y: _pd.Series | _np.ndarray,
) -> _np.ndarray:
    if not isinstance(X, _pd.DataFrame):
        raise TypeError("Repair-rule features must be supplied as a DataFrame.")
    target = _np.asarray(y, dtype=object)
    if target.shape != (len(X),):
        raise ValueError("Repair-rule training rows and labels are misaligned.")
    unexpected = set(_pd.unique(target)).difference(CLASS_LABELS)
    if unexpected:
        raise ValueError(f"Repair-rule labels are invalid: {sorted(unexpected)!r}.")
    return target


def _validate_rule_masks(rule_masks: _pd.DataFrame, *, expected_rows: int) -> None:
    if not isinstance(rule_masks, _pd.DataFrame):
        raise TypeError("Repair-rule masks must be supplied as a DataFrame.")
    if len(rule_masks) != expected_rows:
        raise ValueError("Repair-rule masks have the wrong row count.")
    missing = set(FULL_RULE_NAMES).difference(rule_masks.columns)
    if missing:
        raise KeyError(f"Repair-rule masks are incomplete: {sorted(missing)!r}.")
    if rule_masks.loc[:, list(FULL_RULE_NAMES)].isna().any().any():
        raise ValueError("Repair-rule masks contain missing values.")


def _label_change_counts(
    prefix: str,
    mask: _np.ndarray,
    labels: _np.ndarray,
) -> dict[str, int]:
    selected = labels[mask]
    repair = int((selected == REPAIR_LABEL).sum())
    functional = int((selected == FUNCTIONAL_LABEL).sum())
    return {
        f"{prefix}_actual_repair": repair,
        f"{prefix}_actual_functional": functional,
        f"{prefix}_actual_non_functional": int(
            (selected == NON_FUNCTIONAL_LABEL).sum()
        ),
        f"{prefix}_net_correct": repair - functional,
    }


def _safe_ratio(numerator: _np.ndarray, denominator: _np.ndarray) -> _np.ndarray:
    return _np.asarray(numerator, dtype="float64") / _np.maximum(
        _np.asarray(denominator, dtype="float64"),
        1e-12,
    )


def _string_key(values: _pd.Series) -> _pd.Series:
    return values.fillna("_").astype(str)


def _numeric_key(
    values: _pd.Series,
    *,
    multiplier: float,
    decimals: int,
) -> _pd.Series:
    numeric = _pd.to_numeric(values, errors="coerce") * multiplier
    return numeric.round(decimals).astype(str)


def _integer_grid_key(values: _pd.Series, *, multiplier: float) -> _pd.Series:
    numeric = _pd.to_numeric(values, errors="coerce") * multiplier
    finite = _np.isfinite(numeric.to_numpy(dtype="float64"))
    output = _pd.Series("_", index=values.index, dtype="object")
    output.loc[finite] = _np.floor(numeric.loc[finite]).astype("int64").astype(str)
    return output.astype(str)

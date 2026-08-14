"""Validation for the raw Pump It Up feature and label DataFrames."""

from __future__ import annotations

import json as _json
from pathlib import Path as _Path

import pandas as _pd


_POLICY_FILE = _Path(__file__).with_name("raw_feature_column_policy.json")
_POLICY = _json.loads(_POLICY_FILE.read_text(encoding="utf-8"))
_EXPECTED_FEATURE_COLUMNS = set(_POLICY["expected_columns"])


def validate_raw_feature_schema(raw_features: _pd.DataFrame) -> None:
    """Raise if a raw feature DataFrame does not have the agreed schema."""

    _require_dataframe(raw_features, "raw_features")
    if raw_features.empty:
        raise ValueError("raw_features must contain at least one row.")

    _validate_unique_columns(raw_features, "raw_features")

    actual_columns = set(raw_features.columns)
    missing = sorted(_EXPECTED_FEATURE_COLUMNS - actual_columns)
    unexpected = sorted(actual_columns - _EXPECTED_FEATURE_COLUMNS)
    if missing or unexpected:
        raise ValueError(
            "raw_features does not match the expected schema; "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )


def validate_label_frame(labels: _pd.DataFrame) -> None:
    """Raise if a label DataFrame lacks a usable ID and label column."""

    _require_dataframe(labels, "labels")
    if labels.empty:
        raise ValueError("labels must contain at least one row.")

    _validate_unique_columns(labels, "labels")
    _validate_id_column(labels, "labels")

    if len(labels.columns) < 2:
        raise ValueError(
            "labels must contain at least one label column in addition to 'id'."
        )


def validate_aligned_ids(
    features: _pd.DataFrame,
    labels: _pd.DataFrame,
) -> None:
    """Raise unless feature and label IDs match in the same row order."""

    _require_dataframe(features, "features")
    _require_dataframe(labels, "labels")
    if features.empty:
        raise ValueError("features must contain at least one row.")
    if labels.empty:
        raise ValueError("labels must contain at least one row.")

    _validate_unique_columns(features, "features")
    _validate_unique_columns(labels, "labels")
    _validate_id_column(features, "features")
    _validate_id_column(labels, "labels")

    if len(features) != len(labels):
        raise ValueError(
            f"Row count mismatch: features has {len(features)} rows, "
            f"but labels has {len(labels)} rows."
        )

    feature_ids = set(features["id"])
    label_ids = set(labels["id"])
    if feature_ids != label_ids:
        features_only = sorted(feature_ids - label_ids)
        labels_only = sorted(label_ids - feature_ids)
        raise ValueError(
            "ID sets do not match: "
            f"features-only IDs are {features_only!r}, "
            f"labels-only IDs are {labels_only!r}."
        )

    # Reset the indexes so this compares row position rather than pandas index
    # labels. To my C# head, these are now two plain, zero-based ID sequences.
    feature_ids_in_order = features["id"].reset_index(drop=True)
    label_ids_in_order = labels["id"].reset_index(drop=True)
    mismatched_rows = feature_ids_in_order.ne(label_ids_in_order)
    if bool(mismatched_rows.any()):
        first_mismatch = mismatched_rows.idxmax()
        raise ValueError(
            f"ID mismatch at row {first_mismatch}: "
            f"features ID is {feature_ids_in_order[first_mismatch]!r}, "
            f"but labels ID is {label_ids_in_order[first_mismatch]!r}."
        )


def _require_dataframe(value: object, name: str) -> None:
    if not isinstance(value, _pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame, not {type(value).__name__}."
        )


def _validate_unique_columns(frame: _pd.DataFrame, name: str) -> None:
    duplicated = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"{name} contains duplicate columns: {duplicated!r}.")


def _validate_id_column(frame: _pd.DataFrame, name: str) -> None:
    if "id" not in frame.columns:
        raise KeyError(f"{name} must contain an 'id' column.")

    null_count = int(frame["id"].isna().sum())
    if null_count:
        raise ValueError(f"{name} contains {null_count} null ID value(s).")

    blank_ids = frame["id"].map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    if bool(blank_ids.any()):
        raise ValueError(
            f"{name} contains {int(blank_ids.sum())} blank ID value(s)."
        )

    duplicate_ids = (
        frame.loc[frame["id"].duplicated(keep=False), "id"]
        .unique()
        .tolist()
    )
    if duplicate_ids:
        raise ValueError(f"{name} contains duplicate IDs: {duplicate_ids!r}.")

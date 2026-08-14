"""Fixed, non-statistical column removal for Pump It Up feature data."""

# C#-brain note to future me: this keeps type annotations from being resolved
# while the module is imported. They remain guidance for me and editor tooling;
# Python still does not enforce them like C# parameter and return types.
from __future__ import annotations

import json as _json
from pathlib import Path as _Path

import pandas as _pd


# This is policy, not configuration that should vary by environment. Keeping the
# long schema and fixed decisions in JSON makes them easy to inspect while this
# file stays focused on behaviour. `__file__` means the path of this Python file,
# so the policy is found reliably regardless of the current working directory.
_POLICY_FILE = _Path(__file__).with_name("raw_feature_column_policy.json")
_POLICY = _json.loads(_POLICY_FILE.read_text(encoding="utf-8"))

_EXPECTED_COLUMNS = tuple(_POLICY["expected_columns"])
_COLUMNS_TO_REMOVE = _POLICY["columns_to_remove"]
_PAYMENT_MAPPING = _POLICY["payment_mapping"]
_RECORDED_BY_VALUE = _POLICY["recorded_by_value"]


def remove_known_redundant_columns(
    raw_features: _pd.DataFrame,
) -> _pd.DataFrame:
    """Return raw features without the three agreed redundant columns.

    This is deliberately not learned feature selection or dimensionality
    reduction. The function verifies the fixed evidence for each removal, then
    removes ``quantity_group``, ``payment`` and ``recorded_by``. It retains the
    ``id`` column and does not mutate the supplied DataFrame.
    """

    _validate_raw_schema(raw_features)
    _validate_quantity_duplicate(raw_features)
    _validate_payment_mapping(raw_features)
    _validate_recorded_by(raw_features)

    # `drop` returns a new DataFrame because `inplace` is false by default. The
    # explicit copy makes the ownership boundary unambiguous to my C# head: the
    # caller can edit the result without editing the DataFrame passed in here.
    return raw_features.drop(columns=_COLUMNS_TO_REMOVE).copy(deep=True)


def _validate_raw_schema(raw_features: _pd.DataFrame) -> None:
    if not isinstance(raw_features, _pd.DataFrame):
        raise TypeError("raw_features must be a pandas DataFrame.")

    # Pandas permits duplicate column labels, unlike a normal C# object. Reject
    # them before set comparison could hide the problem.
    duplicated = raw_features.columns[
        raw_features.columns.duplicated()
    ].tolist()
    if duplicated:
        raise ValueError(f"raw_features has duplicate columns: {duplicated!r}.")

    actual_columns = set(raw_features.columns)
    expected_columns = set(_EXPECTED_COLUMNS)
    missing = sorted(expected_columns - actual_columns)
    unexpected = sorted(actual_columns - expected_columns)
    if missing or unexpected:
        raise ValueError(
            "raw_features does not match the expected schema; "
            f"missing={missing!r}, unexpected={unexpected!r}."
        )


def _validate_quantity_duplicate(raw_features: _pd.DataFrame) -> None:
    quantity = raw_features["quantity"]
    quantity_group = raw_features["quantity_group"]

    # These are element-by-element operations. `|` combines the two boolean
    # Series; ordinary Python `or` cannot combine an entire pandas column. The
    # missing-value clause matters because, like SQL NULL, NaN is not equal to
    # itself.
    matches = quantity.eq(quantity_group) | (
        quantity.isna() & quantity_group.isna()
    )
    if not bool(matches.all()):
        mismatch_count = int((~matches).sum())
        raise ValueError(
            "quantity_group no longer duplicates quantity; "
            f"{mismatch_count} row(s) differ."
        )


def _validate_payment_mapping(raw_features: _pd.DataFrame) -> None:
    payment = raw_features["payment"]

    # `isin` is a vectorised `Contains`, and `~` negates each boolean result.
    unknown_payment = ~payment.isin(_PAYMENT_MAPPING)
    if bool(unknown_payment.any()):
        unexpected = payment.loc[unknown_payment].drop_duplicates().tolist()
        raise ValueError(
            "payment contains values outside the agreed mapping: "
            f"{unexpected!r}."
        )

    # `map(dictionary)` is roughly
    # `payment.Select(value => mapping[value])`, while retaining row alignment.
    expected_payment_type = payment.map(_PAYMENT_MAPPING)
    matches = expected_payment_type.eq(raw_features["payment_type"])
    if not bool(matches.all()):
        mismatch_count = int((~matches).sum())
        raise ValueError(
            "payment no longer maps one-to-one to payment_type; "
            f"{mismatch_count} row(s) differ."
        )


def _validate_recorded_by(raw_features: _pd.DataFrame) -> None:
    matches = raw_features["recorded_by"].eq(_RECORDED_BY_VALUE)
    if not bool(matches.all()):
        mismatch_count = int((~matches).sum())
        raise ValueError(
            "recorded_by must contain only "
            f"{_RECORDED_BY_VALUE!r}; {mismatch_count} row(s) differ."
        )

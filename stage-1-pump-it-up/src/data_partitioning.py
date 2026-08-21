"""Freeze the local test partition and development cross-validation folds."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import hashlib as _hashlib

import numpy as _np
import pandas as _pd
from sklearn.model_selection import PredefinedSplit as _PredefinedSplit
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
from sklearn.model_selection import train_test_split as _train_test_split

from modelling_data import ModellingData


LOCAL_TEST_FRACTION = 0.20
LOCAL_TEST_SEED = 20260820
CROSS_VALIDATION_FOLDS = 5
CROSS_VALIDATION_SEED = 20260820

_EXPECTED_DEVELOPMENT_FINGERPRINT = (
    "c8a9e27264e4932140ffe9f19c897229230f9da00126f7daf200b4e37e79f6e6"
)
_EXPECTED_LOCAL_TEST_FINGERPRINT = (
    "6b5aff2b3b7d8dc3bc6ee1b36defabcfef312b03fef940d3fd0ebde4551b1552"
)
_EXPECTED_CROSS_VALIDATION_FINGERPRINT = (
    "bd7da743e9b4498888fa9a7b4166f64266a5f86c5175b7a9805bf3031b554469"
)

_CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


@_dataclass(frozen=True)
class PartitionedData:
    """Development data, untouched local test data and fixed CV membership."""

    development_ids: _pd.Series
    X_development: _pd.DataFrame
    y_development: _pd.Series
    local_test_ids: _pd.Series
    X_local_test: _pd.DataFrame
    y_local_test: _pd.Series
    validation_folds: _pd.Series
    development_fingerprint: str
    local_test_fingerprint: str
    cross_validation_fingerprint: str


def partition_modelling_data(modelling_data: ModellingData) -> PartitionedData:
    """Reserve the local test set and freeze validation-fold membership."""

    original_positions = _np.arange(len(modelling_data.y_original))
    development_positions, local_test_positions = _train_test_split(
        original_positions,
        test_size=LOCAL_TEST_FRACTION,
        random_state=LOCAL_TEST_SEED,
        stratify=modelling_data.y_original,
    )

    # Restore source order within each partition. Membership is randomised; row
    # order remains stable and easier to trace back to the original data.
    development_positions = _np.sort(development_positions)
    local_test_positions = _np.sort(local_test_positions)

    development_ids = modelling_data.original_ids.iloc[
        development_positions
    ].reset_index(drop=True)
    X_development = modelling_data.X_original.iloc[
        development_positions
    ].reset_index(drop=True)
    y_development = modelling_data.y_original.iloc[
        development_positions
    ].reset_index(drop=True)

    local_test_ids = modelling_data.original_ids.iloc[
        local_test_positions
    ].reset_index(drop=True)
    X_local_test = modelling_data.X_original.iloc[
        local_test_positions
    ].reset_index(drop=True)
    y_local_test = modelling_data.y_original.iloc[
        local_test_positions
    ].reset_index(drop=True)

    validation_folds = _assign_validation_folds(X_development, y_development)
    partitioned_data = PartitionedData(
        development_ids=development_ids,
        X_development=X_development,
        y_development=y_development,
        local_test_ids=local_test_ids,
        X_local_test=X_local_test,
        y_local_test=y_local_test,
        validation_folds=validation_folds,
        development_fingerprint=_fingerprint_ids(development_ids),
        local_test_fingerprint=_fingerprint_ids(local_test_ids),
        cross_validation_fingerprint=_fingerprint_folds(
            development_ids,
            validation_folds,
        ),
    )
    _validate_partition_contract(modelling_data, partitioned_data)
    return partitioned_data


def make_cross_validation(
    partitioned_data: PartitionedData,
) -> _PredefinedSplit:
    """Return the frozen development-fold plan for scikit-learn evaluation."""

    zero_based_folds = partitioned_data.validation_folds.to_numpy() - 1
    return _PredefinedSplit(test_fold=zero_based_folds)


def summarise_partitioned_data(
    partitioned_data: PartitionedData,
) -> tuple[_pd.DataFrame, _pd.DataFrame, _pd.DataFrame]:
    """Return settings, partition balance and validation-fold summaries."""

    settings = _pd.DataFrame(
        [
            {"setting": "Local test fraction", "value": LOCAL_TEST_FRACTION},
            {"setting": "Local test seed", "value": LOCAL_TEST_SEED},
            {"setting": "Cross-validation folds", "value": CROSS_VALIDATION_FOLDS},
            {"setting": "Cross-validation seed", "value": CROSS_VALIDATION_SEED},
            {
                "setting": "Development ID fingerprint",
                "value": partitioned_data.development_fingerprint,
            },
            {
                "setting": "Local test ID fingerprint",
                "value": partitioned_data.local_test_fingerprint,
            },
            {
                "setting": "Cross-validation fingerprint",
                "value": partitioned_data.cross_validation_fingerprint,
            },
        ]
    ).set_index("setting")

    partition_summary = _pd.DataFrame(
        [
            _class_summary(
                "Development",
                partitioned_data.y_development,
                denominator=(
                    len(partitioned_data.y_development)
                    + len(partitioned_data.y_local_test)
                ),
            ),
            _class_summary(
                "Local test",
                partitioned_data.y_local_test,
                denominator=(
                    len(partitioned_data.y_development)
                    + len(partitioned_data.y_local_test)
                ),
            ),
        ]
    ).set_index("partition")

    fold_rows = []
    for fold_number in range(1, CROSS_VALIDATION_FOLDS + 1):
        validation_target = partitioned_data.y_development.loc[
            partitioned_data.validation_folds.eq(fold_number)
        ]
        fold_rows.append(
            _class_summary(
                f"Fold {fold_number}",
                validation_target,
                denominator=len(partitioned_data.y_development),
            )
        )
    fold_summary = _pd.DataFrame(fold_rows).set_index("partition")
    fold_summary.index.name = "validation_fold"
    return settings, partition_summary, fold_summary


def _assign_validation_folds(
    X_development: _pd.DataFrame,
    y_development: _pd.Series,
) -> _pd.Series:
    cross_validation = _StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=CROSS_VALIDATION_SEED,
    )
    assignments = _np.zeros(len(y_development), dtype=_np.int8)
    for fold_number, (_, validation_positions) in enumerate(
        cross_validation.split(X_development, y_development),
        start=1,
    ):
        assignments[validation_positions] = fold_number
    return _pd.Series(assignments, name="validation_fold")


def _class_summary(
    name: str,
    target: _pd.Series,
    *,
    denominator: int,
) -> dict[str, object]:
    class_shares = target.value_counts(normalize=True).reindex(
        _CLASS_LABELS,
        fill_value=0,
    )
    return {
        "partition": name,
        "rows": len(target),
        "share_of_parent": len(target) / denominator,
        **{label: class_shares[label] for label in _CLASS_LABELS},
    }


def _fingerprint_ids(ids: _pd.Series) -> str:
    digest = _hashlib.sha256()
    for value in sorted(str(value) for value in ids):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint_folds(ids: _pd.Series, folds: _pd.Series) -> str:
    digest = _hashlib.sha256()
    memberships = sorted(
        zip((str(value) for value in ids), folds, strict=True),
        key=lambda item: item[0],
    )
    for identifier, fold_number in memberships:
        digest.update(f"{identifier}:{fold_number}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_partition_contract(
    modelling_data: ModellingData,
    partitioned_data: PartitionedData,
) -> None:
    expected_shapes = {
        "X_development": (47_520, 36),
        "y_development": (47_520,),
        "development_ids": (47_520,),
        "X_local_test": (11_880, 36),
        "y_local_test": (11_880,),
        "local_test_ids": (11_880,),
        "validation_folds": (47_520,),
    }
    actual_shapes = {
        "X_development": partitioned_data.X_development.shape,
        "y_development": partitioned_data.y_development.shape,
        "development_ids": partitioned_data.development_ids.shape,
        "X_local_test": partitioned_data.X_local_test.shape,
        "y_local_test": partitioned_data.y_local_test.shape,
        "local_test_ids": partitioned_data.local_test_ids.shape,
        "validation_folds": partitioned_data.validation_folds.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(
            "Partition shapes changed; "
            f"expected {expected_shapes!r}, found {actual_shapes!r}."
        )

    development_ids = set(partitioned_data.development_ids)
    local_test_ids = set(partitioned_data.local_test_ids)
    original_ids = set(modelling_data.original_ids)
    competition_ids = set(modelling_data.competition_ids)
    if development_ids & local_test_ids:
        raise ValueError("Development and local test identifiers overlap.")
    if development_ids | local_test_ids != original_ids:
        raise ValueError("Development and local test identifiers do not cover original data.")
    if original_ids & competition_ids:
        raise ValueError("Competition identifiers overlap the locally partitioned data.")

    expected_fold_numbers = set(range(1, CROSS_VALIDATION_FOLDS + 1))
    actual_fold_numbers = set(partitioned_data.validation_folds)
    if actual_fold_numbers != expected_fold_numbers:
        raise ValueError(
            "Cross-validation fold membership is incomplete; "
            f"expected {expected_fold_numbers!r}, found {actual_fold_numbers!r}."
        )

    expected_fingerprints = {
        "development": _EXPECTED_DEVELOPMENT_FINGERPRINT,
        "local_test": _EXPECTED_LOCAL_TEST_FINGERPRINT,
        "cross_validation": _EXPECTED_CROSS_VALIDATION_FINGERPRINT,
    }
    actual_fingerprints = {
        "development": partitioned_data.development_fingerprint,
        "local_test": partitioned_data.local_test_fingerprint,
        "cross_validation": partitioned_data.cross_validation_fingerprint,
    }
    if actual_fingerprints != expected_fingerprints:
        raise ValueError(
            "Frozen partition membership changed; "
            f"expected {expected_fingerprints!r}, found {actual_fingerprints!r}."
        )

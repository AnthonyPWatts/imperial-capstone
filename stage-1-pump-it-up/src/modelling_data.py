"""Prepare the validated Pump It Up modelling-data handoff."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

import pandas as _pd

from data_preparation import remove_known_redundant_columns as _remove_redundant
from source_data_validation import validate_aligned_ids as _validate_aligned_ids


_CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)

_EXPECTED_SHAPES = {
    "X_original": (59_400, 36),
    "y_original": (59_400,),
    "original_ids": (59_400,),
    "X_competition": (14_850, 36),
    "competition_ids": (14_850,),
}


@_dataclass(frozen=True)
class ModellingData:
    """Pre-split labelled data and separate competition-scoring data."""

    original_ids: _pd.Series
    X_original: _pd.DataFrame
    y_original: _pd.Series
    competition_ids: _pd.Series
    X_competition: _pd.DataFrame


def prepare_modelling_data(
    raw_original: _pd.DataFrame,
    labels_original: _pd.DataFrame,
    raw_competition: _pd.DataFrame,
) -> ModellingData:
    """Return the validated modelling frames built from immutable sources.

    ``original`` means the complete labelled data before local partitioning.
    ``competition`` means the unlabelled rows supplied for DrivenData scoring.
    Identifiers are retained as metadata but removed from both predictor frames.
    """

    _validate_aligned_ids(raw_original, labels_original)
    _validate_target_contract(labels_original)

    prepared_original = _remove_redundant(raw_original)
    prepared_competition = _remove_redundant(raw_competition)

    original_ids = prepared_original.pop("id").copy()
    competition_ids = prepared_competition.pop("id").copy()
    X_original = prepared_original.copy(deep=True)
    X_competition = prepared_competition.copy(deep=True)

    labels_by_id = labels_original.set_index("id")
    y_original = labels_by_id.loc[
        original_ids,
        "status_group",
    ].reset_index(drop=True)

    if list(X_original.columns) != list(X_competition.columns):
        raise ValueError(
            "Prepared original and competition predictor columns do not align."
        )

    modelling_data = ModellingData(
        original_ids=original_ids,
        X_original=X_original,
        y_original=y_original,
        competition_ids=competition_ids,
        X_competition=X_competition,
    )
    _validate_modelling_data_contract(modelling_data)
    return modelling_data


def summarise_modelling_data(
    modelling_data: ModellingData,
) -> tuple[_pd.DataFrame, _pd.DataFrame]:
    """Return modelling-frame and target-distribution summaries."""

    checkpoint = _pd.DataFrame(
        [
            {
                "object": "X_original",
                "rows": modelling_data.X_original.shape[0],
                "columns": modelling_data.X_original.shape[1],
                "role": "Predictors before local split",
            },
            {
                "object": "y_original",
                "rows": modelling_data.y_original.shape[0],
                "columns": 1,
                "role": "Target before local split",
            },
            {
                "object": "original_ids",
                "rows": modelling_data.original_ids.shape[0],
                "columns": 1,
                "role": "IDs before local split",
            },
            {
                "object": "X_competition",
                "rows": modelling_data.X_competition.shape[0],
                "columns": modelling_data.X_competition.shape[1],
                "role": "Unlabelled competition predictors",
            },
            {
                "object": "competition_ids",
                "rows": modelling_data.competition_ids.shape[0],
                "columns": 1,
                "role": "Competition submission IDs",
            },
        ]
    ).set_index("object")

    target_summary = (
        modelling_data.y_original.value_counts()
        .reindex(_CLASS_LABELS)
        .rename("rows")
        .to_frame()
    )
    target_summary["share"] = target_summary["rows"] / len(
        modelling_data.y_original
    )
    return checkpoint, target_summary


def _validate_target_contract(labels_original: _pd.DataFrame) -> None:
    if list(labels_original.columns) != ["id", "status_group"]:
        raise ValueError(
            "TrainingSetLabels.csv must contain exactly "
            "['id', 'status_group'] in that order."
        )

    actual_labels = set(labels_original["status_group"].dropna())
    if (
        actual_labels != set(_CLASS_LABELS)
        or labels_original["status_group"].isna().any()
    ):
        raise ValueError(
            "Original labels must be complete and contain exactly the three "
            f"documented classes; found {sorted(actual_labels)!r}."
        )


def _validate_modelling_data_contract(modelling_data: ModellingData) -> None:
    actual_shapes = {
        "X_original": modelling_data.X_original.shape,
        "y_original": modelling_data.y_original.shape,
        "original_ids": modelling_data.original_ids.shape,
        "X_competition": modelling_data.X_competition.shape,
        "competition_ids": modelling_data.competition_ids.shape,
    }
    if actual_shapes != _EXPECTED_SHAPES:
        raise ValueError(
            "Modelling-data shapes changed; "
            f"expected {_EXPECTED_SHAPES!r}, found {actual_shapes!r}."
        )

    forbidden_predictors = {"id", "status_group"}.intersection(
        modelling_data.X_original.columns
    )
    if forbidden_predictors:
        raise ValueError(
            "Metadata or target columns reached the predictors: "
            f"{forbidden_predictors!r}."
        )

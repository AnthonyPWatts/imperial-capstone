"""Fold-fitted preprocessing for the first Pump It Up feature model."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass

import numpy as _np
import pandas as _pd
from scipy import sparse as _sparse
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder

from data_partitioning import PartitionedData
from feature_engineering import (
    CATEGORICAL_FEATURES,
    COMPETITION_REFERENCE_DATE,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    InitialFeatureEngineer,
    engineer_initial_features,
)


RARE_CATEGORY_MINIMUM = 20


@_dataclass(frozen=True)
class PreprocessingSmokeTest:
    """Inspectable result of fitting preprocessing on one development fold."""

    preprocessor: _Pipeline
    summary: _pd.DataFrame
    categorical_coverage: _pd.DataFrame


def make_initial_preprocessor(*, sparse_output: bool = True) -> _Pipeline:
    """Build fold-fitted transforms with sparse or dense equivalent values."""

    numeric_pipeline = _Pipeline(
        steps=[
            (
                "median_imputation",
                _SimpleImputer(strategy="median", add_indicator=True),
            ),
        ]
    )
    categorical_encoder = _OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=RARE_CATEGORY_MINIMUM,
        sparse_output=sparse_output,
    )
    column_preprocessing = _ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_encoder, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", InitialFeatureEngineer()),
            ("column_preprocessing", column_preprocessing),
        ]
    )


def smoke_test_first_fold(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> PreprocessingSmokeTest:
    """Fit only on fold 1's training rows, then transform its validation rows."""

    training_positions, validation_positions = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[training_positions]
    X_validation = partitioned_data.X_development.iloc[validation_positions]

    preprocessor = make_initial_preprocessor()
    transformed_training = preprocessor.fit_transform(X_training)
    transformed_validation = preprocessor.transform(X_validation)
    feature_names = preprocessor.get_feature_names_out()

    unseen_category_probe = X_validation.iloc[[0]].copy()
    unseen_category_probe.loc[:, "basin"] = "__unseen_smoke_test__"
    transformed_probe = preprocessor.transform(unseen_category_probe)

    _validate_transformed_matrix(
        transformed_training,
        expected_rows=len(training_positions),
        expected_columns=len(feature_names),
        name="training",
    )
    _validate_transformed_matrix(
        transformed_validation,
        expected_rows=len(validation_positions),
        expected_columns=len(feature_names),
        name="validation",
    )
    _validate_transformed_matrix(
        transformed_probe,
        expected_rows=1,
        expected_columns=len(feature_names),
        name="unseen-category probe",
    )

    engineered_training = engineer_initial_features(X_training)
    engineered_validation = engineer_initial_features(X_validation)
    categorical_coverage = _summarise_categorical_coverage(
        engineered_training,
        engineered_validation,
    )

    validation_fold = partitioned_data.validation_folds.iloc[
        validation_positions
    ].unique()
    if len(validation_fold) != 1:
        raise ValueError(
            "The preprocessing smoke test spans multiple validation folds."
        )

    summary = _pd.DataFrame(
        [
            {"measurement": "Validation fold", "value": int(validation_fold[0])},
            {"measurement": "Training rows", "value": len(training_positions)},
            {"measurement": "Validation rows", "value": len(validation_positions)},
            {
                "measurement": "Raw predictors received",
                "value": partitioned_data.X_development.shape[1],
            },
            {
                "measurement": "Engineered predictors selected",
                "value": len(MODEL_FEATURES),
            },
            {
                "measurement": "Transformed predictors",
                "value": len(feature_names),
            },
            {
                "measurement": "Sparse output",
                "value": bool(_sparse.issparse(transformed_training)),
            },
            {
                "measurement": "Synthetic unseen category handled",
                "value": True,
            },
            {
                "measurement": "Competition-era reference date",
                "value": str(COMPETITION_REFERENCE_DATE.date()),
            },
            {
                "measurement": "Training days_since_recorded range",
                "value": _format_range(engineered_training["days_since_recorded"]),
            },
            {
                "measurement": "Validation days_since_recorded range",
                "value": _format_range(engineered_validation["days_since_recorded"]),
            },
            {
                "measurement": "Non-finite transformed values",
                "value": 0,
            },
        ]
    ).set_index("measurement")

    return PreprocessingSmokeTest(
        preprocessor=preprocessor,
        summary=summary,
        categorical_coverage=categorical_coverage,
    )


def _summarise_categorical_coverage(
    training: _pd.DataFrame,
    validation: _pd.DataFrame,
) -> _pd.DataFrame:
    rows = []
    for column in CATEGORICAL_FEATURES:
        training_levels = set(training[column])
        validation_levels = set(validation[column])
        unseen_levels = validation_levels - training_levels
        rows.append(
            {
                "feature": column,
                "training_levels": len(training_levels),
                "validation_levels": len(validation_levels),
                "unseen_validation_levels": len(unseen_levels),
                "validation_rows_with_unseen_level": int(
                    validation[column].isin(unseen_levels).sum()
                ),
            }
        )
    return _pd.DataFrame(rows).set_index("feature")


def _validate_transformed_matrix(
    matrix: object,
    *,
    expected_rows: int,
    expected_columns: int,
    name: str,
) -> None:
    if matrix.shape != (expected_rows, expected_columns):
        raise ValueError(
            f"Unexpected {name} matrix shape: {matrix.shape!r}; expected "
            f"{(expected_rows, expected_columns)!r}."
        )
    values = matrix.data if _sparse.issparse(matrix) else _np.asarray(matrix)
    if not _np.isfinite(values).all():
        raise ValueError(f"The transformed {name} matrix contains non-finite values.")


def _format_range(values: _pd.Series) -> str:
    return f"{int(values.min())} to {int(values.max())} days"

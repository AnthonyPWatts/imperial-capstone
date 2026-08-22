"""Evaluate a fixed pump-age cohort beside the accepted continuous age."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace

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

from data_partitioning import PartitionedData
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from feature_engineering import engineer_initial_features
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_preprocessing import RARE_CATEGORY_MINIMUM


AGE_COHORT_FEATURE = "pump_age_cohort"
AGE_COHORT_MODEL_FEATURES = (*MODEL_FEATURES, AGE_COHORT_FEATURE)
AGE_COHORT_CATEGORICAL_FEATURES = (*CATEGORICAL_FEATURES, AGE_COHORT_FEATURE)
AGE_COHORT_BINS = (-1, 2, 5, 10, 20, 30, _np.inf)
AGE_COHORT_LABELS = ("0-2", "3-5", "6-10", "11-20", "21-30", "31+")
UNKNOWN_COHORT = "__unknown__"
INCONSISTENT_COHORT = "__inconsistent__"


@_dataclass(frozen=True)
class AgeCohortTrial:
    """XGBoost and Random Forest evidence for the fixed age cohorts."""

    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    engineered_features: int
    transformed_features_fold_1: int


class AgeCohortFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Add deterministic age cohorts without learning cut points."""

    def fit(self, X: _pd.DataFrame, y: object = None) -> "AgeCohortFeatureEngineer":
        engineered = engineer_age_cohort_features(X)
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(engineered.columns, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(self, "feature_names_out_")
        engineered = engineer_age_cohort_features(X)
        if tuple(engineered.columns) != tuple(self.feature_names_out_):
            raise ValueError("Age-cohort feature columns changed after fitting.")
        return engineered

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def engineer_age_cohort_features(X: _pd.DataFrame) -> _pd.DataFrame:
    """Return accepted features plus one fixed categorical pump-age cohort."""

    engineered = engineer_initial_features(X)
    recorded_year = _pd.to_datetime(X["date_recorded"], errors="raise").dt.year
    construction_year = _pd.to_numeric(X["construction_year"], errors="coerce")
    missing = construction_year.isna() | construction_year.eq(0)
    inconsistent = ~missing & construction_year.gt(recorded_year)
    age = (recorded_year - construction_year).where(~missing & ~inconsistent)
    cohort = _pd.cut(
        age,
        bins=AGE_COHORT_BINS,
        labels=AGE_COHORT_LABELS,
        include_lowest=True,
        right=True,
    ).astype("string")
    cohort = cohort.mask(missing, UNKNOWN_COHORT)
    cohort = cohort.mask(inconsistent, INCONSISTENT_COHORT)
    if cohort.isna().any():
        raise ValueError("Age cohort policy did not classify every row.")
    engineered[AGE_COHORT_FEATURE] = cohort
    if tuple(engineered.columns) != AGE_COHORT_MODEL_FEATURES:
        raise ValueError("Age-cohort feature order changed.")
    return engineered


def make_age_cohort_preprocessor(
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build accepted preprocessing plus the fixed categorical cohort."""

    numeric_steps: list[tuple[str, object]] = [
        (
            "median_imputation",
            _SimpleImputer(strategy="median", add_indicator=True),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("standardisation", _StandardScaler()))
    columns = _ColumnTransformer(
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
                list(AGE_COHORT_CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", AgeCohortFeatureEngineer()),
            ("column_preprocessing", columns),
        ]
    )


def evaluate_age_cohort_trial(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> AgeCohortTrial:
    """Evaluate accepted tree specifications with the fixed age cohort."""

    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [pump-age cohort]",
        feature_policy="accepted plus one fixed pump-age cohort",
    )
    xgboost = evaluate_gpu_candidate(
        xgboost_spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_age_cohort_preprocessor,
    )
    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_age_cohort_preprocessor,
        model_name="Random Forest [pump-age cohort]",
    )
    training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[training_positions]
    preprocessor = make_age_cohort_preprocessor()
    transformed = preprocessor.fit_transform(
        X_training,
        partitioned_data.y_development.iloc[training_positions],
    )
    return AgeCohortTrial(
        xgboost=xgboost,
        random_forest=random_forest,
        engineered_features=len(AGE_COHORT_MODEL_FEATURES),
        transformed_features_fold_1=transformed.shape[1],
    )

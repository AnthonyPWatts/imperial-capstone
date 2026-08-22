"""Evaluate two-stage target-free height and amount reconstruction."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace

import numpy as _np
import pandas as _pd
from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor as _HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer as _SimpleImputer
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.preprocessing import StandardScaler as _StandardScaler
from sklearn.utils.validation import check_is_fitted as _check_is_fitted

from categorical_frequency_forest_evaluation import CategoricalFrequencyFeatureEngineer
from categorical_frequency_forest_evaluation import FREQUENCY_MODEL_FEATURES
from data_partitioning import PartitionedData
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_preprocessing import RARE_CATEGORY_MINIMUM
from spatial_height_imputation_evaluation import engineer_spatial_height_features
from spatial_height_imputation_evaluation import fit_spatial_height_model


AMOUNT_REGRESSION_FEATURES = tuple(
    feature
    for feature in FREQUENCY_MODEL_FEATURES
    if feature not in {"amount_tsh", "amount_tsh_recorded"}
)
AMOUNT_REGRESSION_ITERATIONS = 200
AMOUNT_REGRESSION_LEARNING_RATE = 0.05
AMOUNT_REGRESSION_MAX_LEAVES = 31
AMOUNT_REGRESSION_MIN_LEAF = 20
AMOUNT_REGRESSION_L2 = 1.0
AMOUNT_REGRESSION_SEED = 20260822


@_dataclass(frozen=True)
class LearnedNumericImputationTrial:
    """Accepted components refitted with two-stage numeric reconstruction."""

    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    engineered_features: int
    transformed_features_fold_1: int


class LearnedNumericFeatureEngineer(_BaseEstimator, _TransformerMixin):
    """Fit spatial height then log-amount regression without class labels."""

    def fit(
        self,
        X: _pd.DataFrame,
        y: object = None,
    ) -> "LearnedNumericFeatureEngineer":
        self.spatial_height_model_ = fit_spatial_height_model(X)
        self.frequency_engineer_ = CategoricalFrequencyFeatureEngineer().fit(X)
        predictors = self._amount_predictors(X)
        raw_amount = _pd.to_numeric(X["amount_tsh"], errors="coerce")
        observed = raw_amount.notna() & raw_amount.gt(0)
        if observed.sum() < AMOUNT_REGRESSION_MIN_LEAF * 2:
            raise ValueError("Too few positive amount_tsh rows for regression.")
        self.amount_regressor_ = _HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=AMOUNT_REGRESSION_LEARNING_RATE,
            max_iter=AMOUNT_REGRESSION_ITERATIONS,
            max_leaf_nodes=AMOUNT_REGRESSION_MAX_LEAVES,
            min_samples_leaf=AMOUNT_REGRESSION_MIN_LEAF,
            l2_regularization=AMOUNT_REGRESSION_L2,
            early_stopping=False,
            random_state=AMOUNT_REGRESSION_SEED,
        ).fit(
            predictors.loc[observed, list(AMOUNT_REGRESSION_FEATURES)],
            _np.log1p(raw_amount.loc[observed].to_numpy(dtype="float64")),
        )
        self.feature_names_in_ = _np.asarray(X.columns, dtype=object)
        self.feature_names_out_ = _np.asarray(MODEL_FEATURES, dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: _pd.DataFrame) -> _pd.DataFrame:
        _check_is_fitted(
            self,
            (
                "spatial_height_model_",
                "frequency_engineer_",
                "amount_regressor_",
            ),
        )
        engineered = engineer_spatial_height_features(
            X,
            self.spatial_height_model_,
        )
        raw_amount = _pd.to_numeric(X["amount_tsh"], errors="coerce")
        missing_amount = raw_amount.isna() | raw_amount.le(0)
        if missing_amount.any():
            predictors = self._amount_predictors(X)
            log_predictions = self.amount_regressor_.predict(
                predictors.loc[
                    missing_amount,
                    list(AMOUNT_REGRESSION_FEATURES),
                ]
            )
            engineered.loc[missing_amount, "amount_tsh"] = _np.maximum(
                _np.expm1(log_predictions),
                0.0,
            )
        if engineered["amount_tsh"].isna().any():
            raise ValueError("Learned amount imputation left missing values.")
        return engineered.loc[:, MODEL_FEATURES]

    def get_feature_names_out(self, input_features: object = None) -> _np.ndarray:
        _check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()

    def _amount_predictors(self, X: _pd.DataFrame) -> _pd.DataFrame:
        predictors = self.frequency_engineer_.transform(X)
        spatial = engineer_spatial_height_features(X, self.spatial_height_model_)
        predictors["gps_height"] = spatial["gps_height"].to_numpy()
        return predictors


def make_learned_numeric_preprocessor(
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
) -> _Pipeline:
    """Build accepted preprocessing after two-stage numeric reconstruction."""

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
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
        verbose_feature_names_out=False,
    )
    return _Pipeline(
        steps=[
            ("feature_engineering", LearnedNumericFeatureEngineer()),
            ("column_preprocessing", columns),
        ]
    )


def evaluate_learned_numeric_imputation_trial(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> LearnedNumericImputationTrial:
    """Evaluate accepted trees with spatial height and learned amount."""

    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [learned numeric imputation]",
        feature_policy="spatial height plus learned log amount_tsh",
    )
    xgboost = evaluate_gpu_candidate(
        xgboost_spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_learned_numeric_preprocessor,
    )
    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_learned_numeric_preprocessor,
        model_name="Random Forest [learned numeric imputation]",
    )
    training_positions, _ = next(cross_validation.split())
    X_training = partitioned_data.X_development.iloc[training_positions]
    preprocessor = make_learned_numeric_preprocessor()
    transformed = preprocessor.fit_transform(
        X_training,
        partitioned_data.y_development.iloc[training_positions],
    )
    return LearnedNumericImputationTrial(
        xgboost=xgboost,
        random_forest=random_forest,
        engineered_features=len(MODEL_FEATURES),
        transformed_features_fold_1=transformed.shape[1],
    )

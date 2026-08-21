"""Leakage-safe GPU candidate evaluation on the frozen development folds."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
import time as _time
from typing import Mapping as _Mapping

from catboost import CatBoostClassifier as _CatBoostClassifier
from lightgbm import early_stopping as _lgb_early_stopping
from lightgbm import LGBMClassifier as _LGBMClassifier
from lightgbm import log_evaluation as _lgb_log_evaluation
import numpy as _np
import pandas as _pd
from sklearn.model_selection import train_test_split as _train_test_split
from xgboost import XGBClassifier as _XGBClassifier

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import engineer_initial_features
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_preprocessing import make_initial_preprocessor


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
CLASS_TO_INTEGER = {
    label: position for position, label in enumerate(CLASS_LABELS)
}

CATBOOST_FAMILY = "CatBoost GPU"
XGBOOST_FAMILY = "XGBoost GPU"
LIGHTGBM_FAMILY = "LightGBM CPU"
FROZEN_FEATURE_POLICY = "initial 29-feature policy"

INNER_STOP_FRACTION = 0.10
INNER_STOP_SEED = 20260821
CATBOOST_ITERATION_CAP = 4000
CATBOOST_STOPPING_ROUNDS = 200
XGBOOST_ITERATION_CAP = 3000
XGBOOST_STOPPING_ROUNDS = 100
LIGHTGBM_ITERATION_CAP = 4000
LIGHTGBM_STOPPING_ROUNDS = 100

CATBOOST_VARIANTS: dict[str, dict[str, int | float]] = {
    "d7": {
        "depth": 7,
        "learning_rate": 0.06,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    },
    "d8": {
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    },
    "d9": {
        "depth": 9,
        "learning_rate": 0.04,
        "l2_leaf_reg": 8.0,
        "random_strength": 1.0,
        "bagging_temperature": 1.0,
    },
    "d8 regularised": {
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 12.0,
        "random_strength": 1.5,
        "bagging_temperature": 1.5,
    },
}

XGBOOST_VARIANTS: dict[str, dict[str, int | float | str]] = {
    "depth 8": {
        "learning_rate": 0.04,
        "max_depth": 8,
        "min_child_weight": 3,
        "gamma": 0.0,
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
    },
    "depth 11 conservative": {
        "learning_rate": 0.03,
        "max_depth": 11,
        "min_child_weight": 5,
        "gamma": 0.02,
        "reg_lambda": 4.0,
        "reg_alpha": 0.05,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
    },
    "lossguide 64": {
        "learning_rate": 0.04,
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": 64,
        "min_child_weight": 3,
        "gamma": 0.0,
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
    },
}

LIGHTGBM_VARIANTS: dict[str, dict[str, int | float]] = {
    "leaves 63": {
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_lambda": 2.0,
        "reg_alpha": 0.0,
    },
    "leaves 127 conservative": {
        "learning_rate": 0.025,
        "num_leaves": 127,
        "min_child_samples": 30,
        "subsample": 0.90,
        "colsample_bytree": 0.85,
        "reg_lambda": 4.0,
        "reg_alpha": 0.05,
    },
}


@_dataclass(frozen=True)
class GpuCandidateSpec:
    """One explicit GPU model, feature policy and random seed."""

    name: str
    family: str
    variant: str
    feature_policy: str
    seed: int = INNER_STOP_SEED


def make_catboost_spec(
    *,
    variant: str,
    seed: int = INNER_STOP_SEED,
) -> GpuCandidateSpec:
    """Return a named CatBoost specification from the bounded screen."""

    if variant not in CATBOOST_VARIANTS:
        raise ValueError(f"Unknown CatBoost variant: {variant!r}.")
    return GpuCandidateSpec(
        name=f"CatBoost {variant} [current features]",
        family=CATBOOST_FAMILY,
        variant=variant,
        feature_policy=FROZEN_FEATURE_POLICY,
        seed=seed,
    )


def make_xgboost_spec(
    *,
    variant: str,
    seed: int = INNER_STOP_SEED,
) -> GpuCandidateSpec:
    """Return a current-policy XGBoost specification."""

    if variant not in XGBOOST_VARIANTS:
        raise ValueError(f"Unknown XGBoost variant: {variant!r}.")
    return GpuCandidateSpec(
        name=f"XGBoost {variant} [current one-hot]",
        family=XGBOOST_FAMILY,
        variant=variant,
        feature_policy=FROZEN_FEATURE_POLICY,
        seed=seed,
    )


def make_lightgbm_spec(
    *,
    variant: str,
    seed: int = INNER_STOP_SEED,
) -> GpuCandidateSpec:
    """Return a current-policy CPU LightGBM specification."""

    if variant not in LIGHTGBM_VARIANTS:
        raise ValueError(f"Unknown LightGBM variant: {variant!r}.")
    return GpuCandidateSpec(
        name=f"LightGBM {variant} [current one-hot]",
        family=LIGHTGBM_FAMILY,
        variant=variant,
        feature_policy=FROZEN_FEATURE_POLICY,
        seed=seed,
    )


def with_seed(
    spec: GpuCandidateSpec,
    *,
    seed: int,
) -> GpuCandidateSpec:
    """Return the same candidate with another explicit training seed."""

    return _replace(spec, name=f"{spec.name} seed {seed}", seed=seed)


def evaluate_gpu_candidate(
    spec: GpuCandidateSpec,
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> CandidateEvaluation:
    """Select tree counts inside outer training, then score each outer fold."""

    probability_values = _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostics = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        fold_started = _time.perf_counter()
        probabilities, fold_diagnostics = _fit_outer_fold(
            spec,
            X_training,
            y_training,
            X_validation,
            fold_number=fold_number,
        )
        probability_values[validation_positions] = probabilities
        fold_diagnostics["total_seconds"] = _time.perf_counter() - fold_started
        diagnostics.append(
            {"validation_fold": fold_number, **fold_diagnostics}
        )
        print(
            f"Completed {spec.name} fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS}: "
            f"{fold_diagnostics['selected_iterations']} trees in "
            f"{fold_diagnostics['total_seconds']:.1f} seconds."
        )

    return build_candidate_evaluation(
        model_name=spec.name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )


def fit_gpu_candidate_probabilities(
    spec: GpuCandidateSpec,
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_prediction: _pd.DataFrame,
    *,
    iterations: int,
) -> tuple[_np.ndarray, float]:
    """Fit one fixed-iteration GPU candidate and return ordered probabilities."""

    if iterations <= 0:
        raise ValueError("GPU candidate iterations must be positive.")
    started = _time.perf_counter()
    y_encoded = _encode_target(y_training)
    if spec.family == CATBOOST_FAMILY:
        X_fit, categorical = _engineer_catboost_features(X_training)
        X_predict, prediction_categorical = _engineer_catboost_features(
            X_prediction
        )
        if categorical != prediction_categorical:
            raise ValueError("CatBoost category columns changed at prediction time.")
        model = _make_catboost_model(
            spec,
            iterations=iterations,
            early_stopping=False,
        )
        model.fit(X_fit, y_encoded, cat_features=list(categorical), verbose=False)
    elif spec.family == XGBOOST_FAMILY:
        preprocessor = make_initial_preprocessor(sparse_output=True)
        X_fit = preprocessor.fit_transform(X_training)
        X_predict = preprocessor.transform(X_prediction)
        model = _make_xgboost_model(
            spec,
            iterations=iterations,
            early_stopping=False,
        )
        model.fit(X_fit, y_encoded, verbose=False)
    elif spec.family == LIGHTGBM_FAMILY:
        preprocessor = make_initial_preprocessor(sparse_output=True)
        X_fit = preprocessor.fit_transform(X_training)
        X_predict = preprocessor.transform(X_prediction)
        model = _make_lightgbm_model(
            spec,
            iterations=iterations,
        )
        model.fit(
            X_fit,
            y_encoded,
            callbacks=[_lgb_log_evaluation(period=0)],
        )
    else:
        raise ValueError(f"Unknown GPU family: {spec.family!r}.")

    probabilities = _ordered_encoded_probabilities(model, X_predict)
    return probabilities, _time.perf_counter() - started


def median_selected_iterations(evaluation: CandidateEvaluation) -> int:
    """Return the median fold-selected tree count for a full-data refit."""

    values = evaluation.diagnostics["selected_iterations"].to_numpy()
    if len(values) != CROSS_VALIDATION_FOLDS:
        raise ValueError("Expected one selected iteration count per frozen fold.")
    return int(_np.median(values))


def _fit_outer_fold(
    spec: GpuCandidateSpec,
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_validation: _pd.DataFrame,
    *,
    fold_number: int,
) -> tuple[_np.ndarray, dict[str, int | float]]:
    positions = _np.arange(len(y_training))
    fit_positions, stop_positions = _train_test_split(
        positions,
        test_size=INNER_STOP_FRACTION,
        random_state=INNER_STOP_SEED + fold_number,
        stratify=y_training,
    )
    X_inner_fit = X_training.iloc[fit_positions]
    y_inner_fit = _encode_target(y_training.iloc[fit_positions])
    X_inner_stop = X_training.iloc[stop_positions]
    y_inner_stop = _encode_target(y_training.iloc[stop_positions])
    y_outer_training = _encode_target(y_training)

    stopping_started = _time.perf_counter()
    if spec.family == CATBOOST_FAMILY:
        X_inner_fit, categorical = _engineer_catboost_features(X_inner_fit)
        X_inner_stop, stop_categorical = _engineer_catboost_features(
            X_inner_stop
        )
        if categorical != stop_categorical:
            raise ValueError("CatBoost category columns changed inside a fold.")
        stopping_model = _make_catboost_model(
            spec,
            iterations=CATBOOST_ITERATION_CAP,
            early_stopping=True,
        )
        stopping_model.fit(
            X_inner_fit,
            y_inner_fit,
            cat_features=list(categorical),
            eval_set=(X_inner_stop, y_inner_stop),
            use_best_model=True,
            verbose=False,
        )
        selected_iterations = int(stopping_model.get_best_iteration()) + 1
    elif spec.family == XGBOOST_FAMILY:
        stopping_preprocessor = make_initial_preprocessor(sparse_output=True)
        X_inner_fit = stopping_preprocessor.fit_transform(X_inner_fit)
        X_inner_stop = stopping_preprocessor.transform(X_inner_stop)
        stopping_model = _make_xgboost_model(
            spec,
            iterations=XGBOOST_ITERATION_CAP,
            early_stopping=True,
        )
        stopping_model.fit(
            X_inner_fit,
            y_inner_fit,
            eval_set=[(X_inner_stop, y_inner_stop)],
            verbose=False,
        )
        selected_iterations = int(stopping_model.best_iteration) + 1
    elif spec.family == LIGHTGBM_FAMILY:
        stopping_preprocessor = make_initial_preprocessor(sparse_output=True)
        X_inner_fit = stopping_preprocessor.fit_transform(X_inner_fit)
        X_inner_stop = stopping_preprocessor.transform(X_inner_stop)
        stopping_model = _make_lightgbm_model(
            spec,
            iterations=LIGHTGBM_ITERATION_CAP,
        )
        stopping_model.fit(
            X_inner_fit,
            y_inner_fit,
            eval_X=X_inner_stop,
            eval_y=y_inner_stop,
            eval_metric="multi_logloss",
            callbacks=[
                _lgb_early_stopping(
                    LIGHTGBM_STOPPING_ROUNDS,
                    verbose=False,
                ),
                _lgb_log_evaluation(period=0),
            ],
        )
        selected_iterations = int(stopping_model.best_iteration_)
    else:
        raise ValueError(f"Unknown GPU family: {spec.family!r}.")
    stopping_seconds = _time.perf_counter() - stopping_started
    if selected_iterations <= 0:
        raise ValueError("Early stopping selected no trees.")

    refit_started = _time.perf_counter()
    if spec.family == CATBOOST_FAMILY:
        X_outer_training, categorical = _engineer_catboost_features(X_training)
        X_validation, validation_categorical = _engineer_catboost_features(
            X_validation
        )
        if categorical != validation_categorical:
            raise ValueError("CatBoost category columns changed across outer data.")
        model = _make_catboost_model(
            spec,
            iterations=selected_iterations,
            early_stopping=False,
        )
        model.fit(
            X_outer_training,
            y_outer_training,
            cat_features=list(categorical),
            verbose=False,
        )
    elif spec.family == XGBOOST_FAMILY:
        refit_preprocessor = make_initial_preprocessor(sparse_output=True)
        X_outer_training = refit_preprocessor.fit_transform(X_training)
        X_validation = refit_preprocessor.transform(X_validation)
        model = _make_xgboost_model(
            spec,
            iterations=selected_iterations,
            early_stopping=False,
        )
        model.fit(X_outer_training, y_outer_training, verbose=False)
    else:
        refit_preprocessor = make_initial_preprocessor(sparse_output=True)
        X_outer_training = refit_preprocessor.fit_transform(X_training)
        X_validation = refit_preprocessor.transform(X_validation)
        model = _make_lightgbm_model(
            spec,
            iterations=selected_iterations,
        )
        model.fit(
            X_outer_training,
            y_outer_training,
            callbacks=[_lgb_log_evaluation(period=0)],
        )
    probabilities = _ordered_encoded_probabilities(model, X_validation)
    refit_predict_seconds = _time.perf_counter() - refit_started
    return probabilities, {
        "selected_iterations": selected_iterations,
        "inner_fit_rows": len(fit_positions),
        "inner_stop_rows": len(stop_positions),
        "stopping_seconds": stopping_seconds,
        "refit_predict_seconds": refit_predict_seconds,
    }


def _make_catboost_model(
    spec: GpuCandidateSpec,
    *,
    iterations: int,
    early_stopping: bool,
) -> _CatBoostClassifier:
    parameters = {
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass",
        "custom_metric": ["Accuracy"],
        "task_type": "GPU",
        "devices": "0",
        "boosting_type": "Plain",
        "grow_policy": "SymmetricTree",
        "iterations": iterations,
        "border_count": 128,
        "one_hot_max_size": 10,
        "max_ctr_complexity": 2,
        "bootstrap_type": "Bayesian",
        "random_seed": spec.seed,
        "leaf_estimation_method": "Newton",
        "leaf_estimation_iterations": 1,
        "gpu_ram_part": 0.85,
        "gpu_cat_features_storage": "GpuRam",
        "allow_writing_files": False,
        "verbose": False,
        **CATBOOST_VARIANTS[spec.variant],
    }
    if early_stopping:
        parameters.update(
            {
                "od_type": "Iter",
                "od_wait": CATBOOST_STOPPING_ROUNDS,
            }
        )
    return _CatBoostClassifier(**parameters)


def _make_xgboost_model(
    spec: GpuCandidateSpec,
    *,
    iterations: int,
    early_stopping: bool,
) -> _XGBClassifier:
    parameters = {
        "objective": "multi:softprob",
        "num_class": len(CLASS_LABELS),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": iterations,
        "max_bin": 256,
        "random_state": spec.seed,
        "n_jobs": 6,
        "validate_parameters": True,
        **XGBOOST_VARIANTS[spec.variant],
    }
    if early_stopping:
        parameters["early_stopping_rounds"] = XGBOOST_STOPPING_ROUNDS
    return _XGBClassifier(**parameters)


def _make_lightgbm_model(
    spec: GpuCandidateSpec,
    *,
    iterations: int,
) -> _LGBMClassifier:
    return _LGBMClassifier(
        objective="multiclass",
        num_class=len(CLASS_LABELS),
        n_estimators=iterations,
        max_depth=-1,
        random_state=spec.seed,
        n_jobs=6,
        verbosity=-1,
        **LIGHTGBM_VARIANTS[spec.variant],
    )


def _encode_target(target: _pd.Series) -> _np.ndarray:
    encoded = target.map(CLASS_TO_INTEGER)
    if encoded.isna().any():
        unexpected = sorted(set(target) - set(CLASS_LABELS))
        raise ValueError(f"Unexpected target classes: {unexpected!r}.")
    return encoded.to_numpy(dtype="int64")


def _engineer_catboost_features(
    X: _pd.DataFrame,
) -> tuple[_pd.DataFrame, tuple[str, ...]]:
    engineered = engineer_initial_features(X).copy()
    for column in CATEGORICAL_FEATURES:
        engineered[column] = engineered[column].astype(str)
    return engineered, tuple(CATEGORICAL_FEATURES)


def _ordered_encoded_probabilities(
    model: object,
    X_prediction: object,
) -> _np.ndarray:
    classes = _np.asarray(model.classes_, dtype="int64")
    probabilities = _pd.DataFrame(
        model.predict_proba(X_prediction),
        columns=classes,
    ).loc[:, list(range(len(CLASS_LABELS)))]
    values = probabilities.to_numpy()
    if not _np.isfinite(values).all() or not _np.allclose(
        values.sum(axis=1),
        1.0,
    ):
        raise ValueError("GPU candidate produced invalid probabilities.")
    return values

"""Evaluate one-vs-rest CatBoost with complete native identities."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import time as _time

from catboost import CatBoostClassifier as _CatBoostClassifier
import numpy as _np
import pandas as _pd
from sklearn.model_selection import train_test_split as _train_test_split

from catboost_identity_evaluation import (
    engineer_complete_identity_catboost_features,
)
from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from gpu_model_evaluation import CATBOOST_ITERATION_CAP
from gpu_model_evaluation import CATBOOST_STOPPING_ROUNDS
from gpu_model_evaluation import CATBOOST_VARIANTS
from gpu_model_evaluation import CLASS_LABELS
from gpu_model_evaluation import INNER_STOP_FRACTION
from gpu_model_evaluation import INNER_STOP_SEED
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from one_vs_rest_evaluation import normalise_ovr_memberships


OVR_CATBOOST_VARIANT = "d8"


@_dataclass(frozen=True)
class OneVsRestCatBoostTrial:
    """OOF evidence for the complete-identity CatBoost OvR policy."""

    evaluation: CandidateEvaluation
    class_labels: tuple[str, ...]
    engineered_features: int
    categorical_features: int


def make_ovr_identity_catboost_model(
    *,
    iterations: int,
    early_stopping: bool,
) -> _CatBoostClassifier:
    """Build the fixed binary identity CatBoost boundary."""

    if iterations <= 0:
        raise ValueError("OvR CatBoost iterations must be positive.")
    parameters = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
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
        "random_seed": INNER_STOP_SEED,
        "leaf_estimation_method": "Newton",
        "leaf_estimation_iterations": 1,
        "gpu_ram_part": 0.85,
        "gpu_cat_features_storage": "GpuRam",
        "allow_writing_files": False,
        "verbose": False,
        **CATBOOST_VARIANTS[OVR_CATBOOST_VARIANT],
    }
    if early_stopping:
        parameters.update(
            {
                "od_type": "Iter",
                "od_wait": CATBOOST_STOPPING_ROUNDS,
            }
        )
    return _CatBoostClassifier(**parameters)


def evaluate_one_vs_rest_identity_catboost(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> OneVsRestCatBoostTrial:
    """Fit one complete-identity binary CatBoost boundary per class and fold."""

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
        raw_probabilities = []
        fold_diagnostics: dict[str, int | float] = {
            "inner_training_rows": len(training_positions),
            "validation_rows": len(validation_positions),
        }
        for class_position, class_label in enumerate(CLASS_LABELS):
            positive, selected_iterations = _fit_binary_identity_class_fold(
                X_training,
                y_training.eq(class_label).astype("int8"),
                X_validation,
                split_seed=INNER_STOP_SEED + fold_number + class_position,
            )
            raw_probabilities.append(positive)
            fold_diagnostics[f"{class_label} selected_iterations"] = (
                selected_iterations
            )
        probability_values[validation_positions] = normalise_ovr_memberships(
            _np.column_stack(raw_probabilities)
        )
        fold_diagnostics["total_seconds"] = _time.perf_counter() - fold_started
        diagnostics.append(
            {"validation_fold": fold_number, **fold_diagnostics}
        )
        print(
            f"Completed one-vs-rest identity CatBoost fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS} in "
            f"{fold_diagnostics['total_seconds']:.1f} seconds."
        )

    evaluation = build_candidate_evaluation(
        model_name="One-vs-rest CatBoost d8 [complete deferred identities]",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )
    first_training_positions, _ = next(cross_validation.split())
    engineered, categorical = engineer_complete_identity_catboost_features(
        partitioned_data.X_development.iloc[first_training_positions]
    )
    return OneVsRestCatBoostTrial(
        evaluation=evaluation,
        class_labels=CLASS_LABELS,
        engineered_features=engineered.shape[1],
        categorical_features=len(categorical),
    )


def _fit_binary_identity_class_fold(
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_validation: _pd.DataFrame,
    *,
    split_seed: int,
) -> tuple[_np.ndarray, int]:
    if set(y_training.unique()) != {0, 1}:
        raise ValueError("Every identity OvR boundary needs both binary classes.")
    positions = _np.arange(len(y_training))
    fit_positions, stop_positions = _train_test_split(
        positions,
        test_size=INNER_STOP_FRACTION,
        random_state=split_seed,
        stratify=y_training,
    )
    X_inner_fit, categorical = engineer_complete_identity_catboost_features(
        X_training.iloc[fit_positions]
    )
    X_inner_stop, stop_categorical = (
        engineer_complete_identity_catboost_features(
            X_training.iloc[stop_positions]
        )
    )
    if categorical != stop_categorical:
        raise ValueError("Identity OvR category columns changed inside a fold.")
    stopping_model = make_ovr_identity_catboost_model(
        iterations=CATBOOST_ITERATION_CAP,
        early_stopping=True,
    )
    stopping_model.fit(
        X_inner_fit,
        y_training.iloc[fit_positions],
        cat_features=list(categorical),
        eval_set=(X_inner_stop, y_training.iloc[stop_positions]),
        use_best_model=True,
        verbose=False,
    )
    selected_iterations = int(stopping_model.get_best_iteration()) + 1

    X_fit, categorical = engineer_complete_identity_catboost_features(X_training)
    X_predict, prediction_categorical = (
        engineer_complete_identity_catboost_features(X_validation)
    )
    if categorical != prediction_categorical:
        raise ValueError("Identity OvR category columns changed at prediction.")
    model = make_ovr_identity_catboost_model(
        iterations=selected_iterations,
        early_stopping=False,
    )
    model.fit(
        X_fit,
        y_training,
        cat_features=list(categorical),
        verbose=False,
    )
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Unexpected binary CatBoost classes: {classes!r}.")
    positive = _np.asarray(model.predict_proba(X_predict), dtype="float64")[:, 1]
    return positive, selected_iterations

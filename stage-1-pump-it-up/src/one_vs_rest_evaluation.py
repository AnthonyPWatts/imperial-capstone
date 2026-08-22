"""Evaluate three independent one-vs-rest XGBoost class boundaries."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import time as _time

import numpy as _np
import pandas as _pd
from sklearn.model_selection import train_test_split as _train_test_split
from xgboost import XGBClassifier as _XGBClassifier

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from gpu_model_evaluation import CLASS_LABELS
from gpu_model_evaluation import INNER_STOP_FRACTION
from gpu_model_evaluation import INNER_STOP_SEED
from gpu_model_evaluation import XGBOOST_ITERATION_CAP
from gpu_model_evaluation import XGBOOST_STOPPING_ROUNDS
from gpu_model_evaluation import XGBOOST_VARIANTS
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_preprocessing import make_initial_preprocessor


OVR_VARIANT = "depth 8 child 1"


@_dataclass(frozen=True)
class OneVsRestTrial:
    """OOF evidence for the fixed three-boundary XGBoost policy."""

    evaluation: CandidateEvaluation
    class_labels: tuple[str, ...]


def normalise_ovr_memberships(raw_probabilities: _np.ndarray) -> _np.ndarray:
    """Normalise independent positive probabilities into row memberships."""

    values = _np.asarray(raw_probabilities, dtype="float64")
    if values.ndim != 2 or values.shape[1] != len(CLASS_LABELS):
        raise ValueError("OvR probabilities must have one column per class.")
    if not _np.isfinite(values).all() or (values < 0).any():
        raise ValueError("OvR probabilities must be finite and non-negative.")
    row_sums = values.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("OvR probability rows must have positive mass.")
    return values / row_sums


def evaluate_one_vs_rest_xgboost(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> OneVsRestTrial:
    """Fit one binary XGBoost boundary per class inside every outer fold."""

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
            positive, selected_iterations = fit_binary_xgboost_fold(
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
            f"Completed one-vs-rest XGBoost fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS} in "
            f"{fold_diagnostics['total_seconds']:.1f} seconds."
        )

    evaluation = build_candidate_evaluation(
        model_name="One-vs-rest XGBoost depth 8 child 1",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )
    return OneVsRestTrial(evaluation=evaluation, class_labels=CLASS_LABELS)


def fit_binary_xgboost_fold(
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_validation: _pd.DataFrame,
    *,
    split_seed: int,
) -> tuple[_np.ndarray, int]:
    if set(y_training.unique()) != {0, 1}:
        raise ValueError("Every binary boundary requires positive and negative rows.")
    positions = _np.arange(len(y_training))
    fit_positions, stop_positions = _train_test_split(
        positions,
        test_size=INNER_STOP_FRACTION,
        random_state=split_seed,
        stratify=y_training,
    )
    stopping_preprocessor = make_initial_preprocessor()
    X_inner_fit = stopping_preprocessor.fit_transform(
        X_training.iloc[fit_positions],
        y_training.iloc[fit_positions],
    )
    X_inner_stop = stopping_preprocessor.transform(
        X_training.iloc[stop_positions]
    )
    stopping_model = _make_binary_xgboost(
        iterations=XGBOOST_ITERATION_CAP,
        early_stopping=True,
    )
    stopping_model.fit(
        X_inner_fit,
        y_training.iloc[fit_positions],
        eval_set=[(X_inner_stop, y_training.iloc[stop_positions])],
        verbose=False,
    )
    selected_iterations = int(stopping_model.best_iteration) + 1

    refit_preprocessor = make_initial_preprocessor()
    X_fit = refit_preprocessor.fit_transform(X_training, y_training)
    X_predict = refit_preprocessor.transform(X_validation)
    model = _make_binary_xgboost(
        iterations=selected_iterations,
        early_stopping=False,
    )
    model.fit(X_fit, y_training, verbose=False)
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Unexpected binary XGBoost classes: {classes!r}.")
    positive = _np.asarray(model.predict_proba(X_predict), dtype="float64")[:, 1]
    return positive, selected_iterations


def _make_binary_xgboost(
    *,
    iterations: int,
    early_stopping: bool,
) -> _XGBClassifier:
    if iterations <= 0:
        raise ValueError("OvR XGBoost iterations must be positive.")
    parameters = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": iterations,
        "max_bin": 256,
        "random_state": INNER_STOP_SEED,
        "n_jobs": 6,
        "validate_parameters": True,
        **XGBOOST_VARIANTS[OVR_VARIANT],
    }
    if early_stopping:
        parameters["early_stopping_rounds"] = XGBOOST_STOPPING_ROUNDS
    return _XGBClassifier(**parameters)

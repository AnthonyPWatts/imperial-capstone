"""Fold-safe RealMLP-TD feature preparation and probability utilities."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import gc as _gc
from pathlib import Path as _Path
import time as _time
from typing import Any as _Any

import numpy as _np
import pandas as _pd

from catboost_identity_evaluation import DEFERRED_IDENTITY_FEATURES
from catboost_identity_evaluation import engineer_complete_identity_catboost_features
from feature_engineering import CATEGORICAL_FEATURES


REALMLP_SEED = 20260904
REALMLP_BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20)
REALMLP_FIT_TIME_LIMIT_SECONDS = 600
REALMLP_MODEL_NAME = (
    "RealMLP-TD [accepted features plus six normalised identities]"
)
REALMLP_CATEGORICAL_FEATURES = (
    *CATEGORICAL_FEATURES,
    *DEFERRED_IDENTITY_FEATURES,
)
CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


@_dataclass(frozen=True)
class RealMLPFeatureState:
    """Training-fold statistics required by the RealMLP input contract."""

    columns: tuple[str, ...]
    numeric_features: tuple[str, ...]
    numeric_medians: _pd.Series


@_dataclass(frozen=True)
class RealMLPFitPrediction:
    """Probabilities and elapsed times from one independently fitted model."""

    probabilities: _np.ndarray
    fit_seconds: float
    predict_seconds: float


def fit_realmlp_feature_state(X_training: _pd.DataFrame) -> RealMLPFeatureState:
    """Fit numeric imputation using training rows only."""

    engineered, categorical = engineer_complete_identity_catboost_features(
        X_training
    )
    if categorical != REALMLP_CATEGORICAL_FEATURES:
        raise ValueError("RealMLP categorical feature order changed.")
    numeric_features = tuple(
        column
        for column in engineered.columns
        if column not in REALMLP_CATEGORICAL_FEATURES
    )
    numeric = engineered.loc[:, numeric_features].apply(
        _pd.to_numeric,
        errors="coerce",
    )
    finite_or_missing = _np.isfinite(numeric.to_numpy(dtype="float64")) | (
        numeric.isna().to_numpy()
    )
    if not finite_or_missing.all():
        raise ValueError("RealMLP numeric training features contain infinity.")
    medians = numeric.median(axis="index")
    if medians.isna().any() or not _np.isfinite(
        medians.to_numpy(dtype="float64")
    ).all():
        raise ValueError("RealMLP numeric medians are incomplete or non-finite.")
    return RealMLPFeatureState(
        columns=tuple(engineered.columns),
        numeric_features=numeric_features,
        numeric_medians=medians,
    )


def transform_realmlp_features(
    X: _pd.DataFrame,
    state: RealMLPFeatureState,
) -> _pd.DataFrame:
    """Apply deterministic features and training-fold numeric imputation."""

    engineered, categorical = engineer_complete_identity_catboost_features(X)
    if tuple(engineered.columns) != state.columns:
        raise ValueError("RealMLP engineered feature columns changed.")
    if categorical != REALMLP_CATEGORICAL_FEATURES:
        raise ValueError("RealMLP categorical feature order changed.")

    transformed = engineered.copy()
    numeric = transformed.loc[:, state.numeric_features].apply(
        _pd.to_numeric,
        errors="coerce",
    )
    numeric = numeric.fillna(state.numeric_medians).astype("float32")
    if not _np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise ValueError("RealMLP transformed numeric features are not finite.")
    for column in state.numeric_features:
        transformed[column] = numeric[column]
    for column in REALMLP_CATEGORICAL_FEATURES:
        transformed[column] = (
            transformed[column]
            .astype("string")
            .fillna("__missing__")
            .astype(str)
        )
    return transformed.loc[:, state.columns]


def prepare_realmlp_features(
    X_training: _pd.DataFrame,
    X_prediction: _pd.DataFrame,
) -> tuple[_pd.DataFrame, _pd.DataFrame, RealMLPFeatureState]:
    """Fit one training-only state and transform train and prediction rows."""

    state = fit_realmlp_feature_state(X_training)
    return (
        transform_realmlp_features(X_training, state),
        transform_realmlp_features(X_prediction, state),
        state,
    )


def make_realmlp_classifier(
    temporary_folder: _Path,
    *,
    device: str = "cuda",
) -> _Any:
    """Create exactly one seeded tuned-default RealMLP classifier."""

    from pytabkit import RealMLP_TD_Classifier

    temporary_folder = _Path(temporary_folder)
    temporary_folder.mkdir(parents=True, exist_ok=True)
    return RealMLP_TD_Classifier(
        device=device,
        random_state=REALMLP_SEED,
        n_cv=1,
        n_refit=0,
        n_repeats=1,
        val_fraction=0.20,
        tmp_folder=temporary_folder,
        verbosity=1,
        val_metric_name="class_error",
    )


def fit_predict_realmlp(
    X_training: _pd.DataFrame,
    y_training: _pd.Series | _np.ndarray,
    X_prediction: _pd.DataFrame,
    temporary_folder: _Path,
    *,
    device: str = "cuda",
) -> RealMLPFitPrediction:
    """Fit one tuned-default model with an internal validation split."""

    training, prediction, _ = prepare_realmlp_features(
        X_training,
        X_prediction,
    )
    classifier = make_realmlp_classifier(temporary_folder, device=device)
    try:
        started = _time.perf_counter()
        classifier.fit(
            training,
            _np.asarray(y_training),
            cat_col_names=list(REALMLP_CATEGORICAL_FEATURES),
            time_to_fit_in_seconds=REALMLP_FIT_TIME_LIMIT_SECONDS,
        )
        fit_seconds = _time.perf_counter() - started
        predict_started = _time.perf_counter()
        probabilities = ordered_probabilities(classifier, prediction)
        predict_seconds = _time.perf_counter() - predict_started
        return RealMLPFitPrediction(
            probabilities=probabilities,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
        )
    finally:
        try:
            classifier.to("cpu")
        except (AttributeError, RuntimeError):
            pass
        del classifier
        _gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def ordered_probabilities(classifier: _Any, X: _pd.DataFrame) -> _np.ndarray:
    """Predict probabilities in the fixed competition class order."""

    probabilities = (
        _pd.DataFrame(
            classifier.predict_proba(X),
            columns=classifier.classes_,
        )
        .loc[:, list(CLASS_LABELS)]
        .to_numpy(dtype="float64")
    )
    validate_probabilities(probabilities, len(X))
    return probabilities


def blend_probabilities(
    incumbent: _np.ndarray,
    realmlp: _np.ndarray,
    realmlp_weight: float,
) -> _np.ndarray:
    """Blend fixed aligned probabilities without fitting a combiner."""

    if realmlp_weight <= 0.0 or realmlp_weight >= 1.0:
        raise ValueError("RealMLP blend weight must be strictly between zero and one.")
    if incumbent.shape != realmlp.shape:
        raise ValueError("Incumbent and RealMLP probability shapes differ.")
    validate_probabilities(incumbent, len(incumbent))
    validate_probabilities(realmlp, len(realmlp))
    blended = (1.0 - realmlp_weight) * incumbent + realmlp_weight * realmlp
    validate_probabilities(blended, len(blended))
    return blended


def hard_predictions(probabilities: _np.ndarray) -> _np.ndarray:
    """Return labels from an aligned three-class probability matrix."""

    validate_probabilities(probabilities, len(probabilities))
    return _np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]


def validate_probabilities(probabilities: _np.ndarray, rows: int) -> None:
    """Reject incomplete or misaligned probability matrices."""

    values = _np.asarray(probabilities)
    if values.shape != (rows, len(CLASS_LABELS)):
        raise ValueError(
            "Expected probability shape "
            f"{(rows, len(CLASS_LABELS))}, found {values.shape}."
        )
    if not _np.isfinite(values).all():
        raise ValueError("Probabilities are not all finite.")
    if (values < 0.0).any() or (values > 1.0).any():
        raise ValueError("Probabilities fall outside zero to one.")
    if not _np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Probabilities do not sum to one.")

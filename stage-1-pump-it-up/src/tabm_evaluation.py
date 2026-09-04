"""Fold-safe preprocessing and fitting for the bounded TabM screen."""

from __future__ import annotations

from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
import math as _math
import random as _random
import time as _time
from typing import Any as _Any

import numpy as _np
import pandas as _pd
from sklearn.model_selection import train_test_split as _train_test_split
from sklearn.preprocessing import QuantileTransformer as _QuantileTransformer

from catboost_identity_evaluation import DEFERRED_IDENTITY_FEATURES
from catboost_identity_evaluation import engineer_complete_identity_catboost_features
from feature_engineering import CATEGORICAL_FEATURES


TABM_SEED = 20260904
TABM_TOP_IDENTITY_VALUES = 50
TABM_BLEND_WEIGHTS = (0.02, 0.05, 0.10, 0.15)
TABM_INNER_VALIDATION_FRACTION = 0.20
TABM_NUMERIC_BINS = 48
TABM_NUMERIC_EMBEDDING_DIMENSION = 16
TABM_BATCH_SIZE = 256
TABM_MAX_EPOCHS = 256
TABM_PATIENCE = 16
CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


@_dataclass(frozen=True)
class TabMPreprocessorState:
    """Statistics and category dictionaries learned from training rows only."""

    numeric_features: tuple[str, ...]
    numeric_medians: _pd.Series
    numeric_transformer: _QuantileTransformer
    categorical_features: tuple[str, ...]
    category_values: tuple[tuple[str, ...], ...]

    @property
    def category_cardinalities(self) -> tuple[int, ...]:
        """Include index zero, reserved for rare or unseen values."""

        return tuple(len(values) + 1 for values in self.category_values)


@_dataclass(frozen=True)
class TabMPreparedFeatures:
    """Dense arrays in the exact contract expected by ``tabm.TabM``."""

    numeric: _np.ndarray
    categorical: _np.ndarray


@_dataclass(frozen=True)
class TabMFitPrediction:
    """One fold's outer-validation predictions and fit diagnostics."""

    probabilities: _np.ndarray
    fit_seconds: float
    best_epoch: int
    best_inner_accuracy: float
    parameter_count: int
    category_cardinalities: tuple[int, ...]


def fit_tabm_preprocessor(X_training: _pd.DataFrame) -> TabMPreprocessorState:
    """Fit numeric scaling and compact categorical dictionaries fold-safely."""

    engineered, categorical_features = engineer_complete_identity_catboost_features(
        X_training
    )
    expected_categorical = (*CATEGORICAL_FEATURES, *DEFERRED_IDENTITY_FEATURES)
    if categorical_features != expected_categorical:
        raise ValueError("TabM categorical feature order changed.")

    numeric_features = tuple(
        column for column in engineered if column not in categorical_features
    )
    numeric = engineered.loc[:, numeric_features].apply(
        _pd.to_numeric,
        errors="coerce",
    )
    medians = numeric.median(axis="index")
    if medians.isna().any() or not _np.isfinite(
        medians.to_numpy(dtype="float64")
    ).all():
        raise ValueError("TabM numeric medians are incomplete or non-finite.")
    numeric_values = numeric.fillna(medians).to_numpy(dtype="float32")
    noise = _np.random.default_rng(0).normal(
        0.0,
        1e-5,
        numeric_values.shape,
    ).astype("float32")
    transformer = _QuantileTransformer(
        n_quantiles=max(min(len(X_training) // 30, 1000), 10),
        output_distribution="normal",
        subsample=10**9,
        random_state=TABM_SEED,
    ).fit(numeric_values + noise)

    category_values = []
    for feature in categorical_features:
        limit = (
            TABM_TOP_IDENTITY_VALUES
            if feature in DEFERRED_IDENTITY_FEATURES
            else None
        )
        category_values.append(
            select_category_values(engineered[feature], limit=limit)
        )
    return TabMPreprocessorState(
        numeric_features=numeric_features,
        numeric_medians=medians,
        numeric_transformer=transformer,
        categorical_features=categorical_features,
        category_values=tuple(category_values),
    )


def transform_tabm_features(
    X: _pd.DataFrame,
    state: TabMPreprocessorState,
) -> TabMPreparedFeatures:
    """Apply one training-only preprocessing state to arbitrary predictor rows."""

    engineered, categorical_features = engineer_complete_identity_catboost_features(X)
    if categorical_features != state.categorical_features:
        raise ValueError("TabM categorical feature order changed.")
    numeric = (
        engineered.loc[:, state.numeric_features]
        .apply(_pd.to_numeric, errors="coerce")
        .fillna(state.numeric_medians)
        .to_numpy(dtype="float32")
    )
    numeric = state.numeric_transformer.transform(numeric).astype(
        "float32",
        copy=False,
    )
    if not _np.isfinite(numeric).all():
        raise ValueError("TabM transformed numeric features are not finite.")

    encoded_columns = []
    for feature, values in zip(
        state.categorical_features,
        state.category_values,
        strict=True,
    ):
        encoded_columns.append(encode_categories(engineered[feature], values))
    categorical = _np.column_stack(encoded_columns).astype("int64", copy=False)
    return TabMPreparedFeatures(numeric=numeric, categorical=categorical)


def select_category_values(
    values: _pd.Series,
    *,
    limit: int | None,
) -> tuple[str, ...]:
    """Select frequent values deterministically, using lexical tie-breaking."""

    normalised = values.astype("string").fillna("__missing__").astype(str)
    counts = (
        normalised.value_counts(dropna=False)
        .rename_axis("value")
        .reset_index(name="count")
    )
    counts["sort_value"] = counts["value"].astype(str)
    counts = counts.sort_values(
        ["count", "sort_value"],
        ascending=[False, True],
        kind="stable",
    )
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("Category limit must be a positive integer or None.")
        counts = counts.iloc[:limit]
    return tuple(counts["value"].astype(str))


def encode_categories(
    values: _pd.Series,
    selected_values: tuple[str, ...],
) -> _np.ndarray:
    """Map selected categories to 1..N and rare/unseen categories to zero."""

    mapping = {value: position for position, value in enumerate(selected_values, 1)}
    normalised = values.astype("string").fillna("__missing__").astype(str)
    return normalised.map(mapping).fillna(0).to_numpy(dtype="int64")


def fit_predict_tabm(
    X_outer_training: _pd.DataFrame,
    y_outer_training: _pd.Series,
    X_outer_validation: _pd.DataFrame,
    *,
    device: str = "cuda",
    seed: int = TABM_SEED,
) -> TabMFitPrediction:
    """Fit official TabM defaults with an inner validation early-stop split."""

    import rtdl_num_embeddings
    import torch
    import torch.nn.functional as functional
    from tabm import TabM

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA TabM requested, but PyTorch cannot see the GPU.")
    torch_device = torch.device(device)
    _random.seed(seed)
    _np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 2)

    positions = _np.arange(len(y_outer_training))
    inner_training, inner_validation = _train_test_split(
        positions,
        test_size=TABM_INNER_VALIDATION_FRACTION,
        random_state=seed,
        stratify=y_outer_training,
    )
    state = fit_tabm_preprocessor(X_outer_training.iloc[inner_training])
    prepared_training = transform_tabm_features(
        X_outer_training.iloc[inner_training], state
    )
    prepared_inner_validation = transform_tabm_features(
        X_outer_training.iloc[inner_validation], state
    )
    prepared_outer_validation = transform_tabm_features(X_outer_validation, state)
    label_positions = {label: position for position, label in enumerate(CLASS_LABELS)}
    y_values = y_outer_training.map(label_positions).to_numpy(dtype="int64")

    data = {
        "train_num": torch.as_tensor(prepared_training.numeric, device=torch_device),
        "train_cat": torch.as_tensor(
            prepared_training.categorical, device=torch_device
        ),
        "train_y": torch.as_tensor(y_values[inner_training], device=torch_device),
        "inner_num": torch.as_tensor(
            prepared_inner_validation.numeric, device=torch_device
        ),
        "inner_cat": torch.as_tensor(
            prepared_inner_validation.categorical, device=torch_device
        ),
        "inner_y": torch.as_tensor(y_values[inner_validation], device=torch_device),
        "outer_num": torch.as_tensor(
            prepared_outer_validation.numeric, device=torch_device
        ),
        "outer_cat": torch.as_tensor(
            prepared_outer_validation.categorical, device=torch_device
        ),
    }
    numeric_embeddings = rtdl_num_embeddings.PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(
            data["train_num"],
            n_bins=TABM_NUMERIC_BINS,
        ),
        d_embedding=TABM_NUMERIC_EMBEDDING_DIMENSION,
        activation=False,
        version="B",
    )
    model = TabM.make(
        n_num_features=prepared_training.numeric.shape[1],
        cat_cardinalities=list(state.category_cardinalities),
        d_out=len(CLASS_LABELS),
        num_embeddings=numeric_embeddings,
    ).to(torch_device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=3e-4,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    best_state = _deepcopy(model.state_dict())
    best_inner_accuracy = -_math.inf
    best_epoch = -1
    remaining_patience = TABM_PATIENCE
    started = _time.perf_counter()

    for epoch in range(TABM_MAX_EPOCHS):
        model.train()
        batches = torch.randperm(
            len(inner_training),
            device=torch_device,
        ).split(TABM_BATCH_SIZE)
        for batch in batches:
            optimiser.zero_grad()
            logits = model(
                data["train_num"][batch],
                data["train_cat"][batch],
            )
            loss = functional.cross_entropy(
                logits.flatten(0, 1),
                data["train_y"][batch].repeat_interleave(model.k),
            )
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        inner_probabilities = _predict_probabilities(
            model,
            data["inner_num"],
            data["inner_cat"],
        )
        inner_predictions = inner_probabilities.argmax(axis=1)
        inner_accuracy = float(
            (inner_predictions == y_values[inner_validation]).mean()
        )
        if inner_accuracy > best_inner_accuracy:
            best_inner_accuracy = inner_accuracy
            best_epoch = epoch
            best_state = _deepcopy(model.state_dict())
            remaining_patience = TABM_PATIENCE
        else:
            remaining_patience -= 1
        if remaining_patience < 0:
            break

    model.load_state_dict(best_state)
    probabilities = _predict_probabilities(
        model,
        data["outer_num"],
        data["outer_cat"],
    )
    fit_seconds = _time.perf_counter() - started
    _validate_probabilities(probabilities, len(X_outer_validation))
    model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return TabMFitPrediction(
        probabilities=probabilities,
        fit_seconds=fit_seconds,
        best_epoch=best_epoch,
        best_inner_accuracy=best_inner_accuracy,
        parameter_count=parameter_count,
        category_cardinalities=state.category_cardinalities,
    )


def _predict_probabilities(
    model: _Any,
    numeric: _Any,
    categorical: _Any,
) -> _np.ndarray:
    import torch

    model.eval()
    chunks = []
    with torch.inference_mode():
        for positions in torch.arange(
            len(numeric),
            device=numeric.device,
        ).split(2048):
            logits = model(numeric[positions], categorical[positions])
            chunks.append(logits.softmax(dim=-1).mean(dim=1).cpu())
    return torch.cat(chunks).numpy().astype("float64", copy=False)


def _validate_probabilities(probabilities: _np.ndarray, rows: int) -> None:
    if probabilities.shape != (rows, len(CLASS_LABELS)):
        raise ValueError("TabM probability shape changed.")
    if not _np.isfinite(probabilities).all():
        raise ValueError("TabM probabilities are not finite.")
    if not _np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("TabM probabilities do not sum to one.")

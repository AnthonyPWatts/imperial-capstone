"""Two predeclared category-model challengers to the 0.8304 incumbent."""

from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split

from final_model import CLASS_LABELS, validate_probabilities
from gpu_model_evaluation import (
    INNER_STOP_SEED, INNER_STOP_FRACTION,
    _make_catboost_model, _make_lightgbm_model,
    make_catboost_spec, make_lightgbm_spec,
)
from spatial_grid_catboost_evaluation import engineer_spatial_grid_catboost_features


CHALLENGERS = ("catboost_grid_depth10", "lightgbm_native_identity_grid")
BLENDS = ("depth10_grid_half_slot", "native_lightgbm_at_0_10")
RECIPES = {
    CHALLENGERS[0]: {
        "base": "CatBoost d8 GPU", "depth": 10, "learning_rate": 0.05,
        "l2_leaf_reg": 5.0, "features": "complete identities and four spatial grids",
        "seed": INNER_STOP_SEED, "iteration_cap": 4000, "stopping_rounds": 200,
    },
    CHALLENGERS[1]: {
        "base": "LightGBM leaves 63 bagged CPU", "learning_rate": 0.03,
        "num_leaves": 63, "min_child_samples": 20, "subsample": 0.8,
        "subsample_freq": 1, "colsample_bytree": 0.9, "reg_lambda": 2.0,
        "reg_alpha": 0.0, "cat_smooth": 20.0, "cat_l2": 10.0,
        "min_data_per_group": 100, "max_cat_threshold": 32,
        "category_vocabulary": "fit partition only; unseen category becomes missing",
        "features": "complete identities and four spatial grids",
        "seed": INNER_STOP_SEED, "iteration_cap": 4000, "stopping_rounds": 100,
    },
}


def native_frames(training, validation):
    """Use training-only category vocabularies, with consistent ordered columns."""
    training, categories = engineer_spatial_grid_catboost_features(training)
    validation, validation_categories = engineer_spatial_grid_catboost_features(validation)
    if categories != validation_categories or list(training) != list(validation):
        raise ValueError("Feature contracts differ across partitions.")
    for column in training:
        if column in categories:
            vocabulary = sorted(training[column].unique().tolist())
            dtype = pd.CategoricalDtype(vocabulary, ordered=False)
            training[column] = training[column].astype(dtype)
            validation[column] = validation[column].astype(dtype)
        else:
            training[column] = training[column].astype("float64")
            validation[column] = validation[column].astype("float64")
    return training, validation, categories


def make_model(candidate, iterations, *, stopping=False):
    if candidate == CHALLENGERS[0]:
        model = _make_catboost_model(
            make_catboost_spec(variant="d8"), iterations=iterations,
            early_stopping=stopping,
        )
        return model.set_params(depth=10)
    if candidate == CHALLENGERS[1]:
        model = _make_lightgbm_model(
            make_lightgbm_spec(variant="leaves 63 bagged"), iterations=iterations,
        )
        return model.set_params(cat_smooth=20.0, cat_l2=10.0,
                                min_data_per_group=100, max_cat_threshold=32)
    raise ValueError(f"Unknown locked candidate: {candidate}.")


def encode_target(target):
    encoded = pd.Series(target).map(dict(zip(CLASS_LABELS, range(3))))
    if encoded.isna().any():
        raise ValueError("Unknown target class.")
    return encoded.to_numpy(dtype="int64")


def fit_probabilities(candidate, X_train, y_train, X_predict, *, iterations):
    """Refit on all supplied training rows, never on prediction labels."""
    model = make_model(candidate, iterations)
    if candidate == CHALLENGERS[0]:
        training, categorical = engineer_spatial_grid_catboost_features(X_train)
        prediction, _ = engineer_spatial_grid_catboost_features(X_predict)
        model.fit(training, encode_target(y_train), cat_features=list(categorical),
                  verbose=False)
    else:
        training, prediction, categorical = native_frames(X_train, X_predict)
        model.fit(training, encode_target(y_train), categorical_feature=list(categorical),
                  callbacks=[log_evaluation(period=0)])
    if not np.array_equal(model.classes_, np.arange(3)):
        raise ValueError("Estimator class order changed.")
    probabilities = np.asarray(model.predict_proba(prediction), dtype="float64")
    validate_probabilities(probabilities, len(X_predict))
    return probabilities


def fit_fold(candidate, X_train, y_train, X_predict, fold):
    """Select iterations inside outer training, then refit all outer training."""
    started = time.perf_counter()
    fit, stop = train_test_split(
        np.arange(len(y_train)), test_size=INNER_STOP_FRACTION,
        random_state=INNER_STOP_SEED + fold, stratify=y_train,
    )
    model = make_model(candidate, 4000, stopping=True)
    if candidate == CHALLENGERS[0]:
        training, categorical = engineer_spatial_grid_catboost_features(X_train.iloc[fit])
        stopping, _ = engineer_spatial_grid_catboost_features(X_train.iloc[stop])
        model.fit(training, encode_target(y_train.iloc[fit]),
                  cat_features=list(categorical),
                  eval_set=(stopping, encode_target(y_train.iloc[stop])),
                  use_best_model=True, verbose=False)
        iterations = int(model.get_best_iteration()) + 1
    else:
        training, stopping, categorical = native_frames(X_train.iloc[fit], X_train.iloc[stop])
        model.fit(training, encode_target(y_train.iloc[fit]),
                  categorical_feature=list(categorical),
                  eval_X=stopping, eval_y=encode_target(y_train.iloc[stop]),
                  eval_metric="multi_logloss",
                  callbacks=[early_stopping(100, verbose=False), log_evaluation(period=0)])
        iterations = int(model.best_iteration_)
    if not 1 <= iterations <= 4000:
        raise ValueError("Invalid selected iteration count.")
    probabilities = fit_probabilities(candidate, X_train, y_train, X_predict,
                                      iterations=iterations)
    return probabilities, {"validation_fold": fold, "selected_iterations": iterations,
                           "inner_fit_rows": len(fit), "inner_stop_rows": len(stop),
                           "total_seconds": time.perf_counter() - started}


def blend_candidates(incumbent, grid, challengers):
    validate_probabilities(incumbent, len(incumbent))
    validate_probabilities(grid, len(incumbent))
    output = {}
    for candidate, probabilities in challengers.items():
        validate_probabilities(probabilities, len(incumbent))
        if candidate == CHALLENGERS[0]:
            output[BLENDS[0]] = incumbent + 0.05 * (probabilities - grid)
        elif candidate == CHALLENGERS[1]:
            output[BLENDS[1]] = 0.9 * incumbent + 0.1 * probabilities
        else:
            raise ValueError("Unregistered challenger.")
    if set(output) == set(BLENDS):
        output["combined_locked_changes"] = 0.9 * output[BLENDS[0]] + 0.1 * challengers[CHALLENGERS[1]]
    for probabilities in output.values():
        validate_probabilities(probabilities, len(incumbent))
    return output


def compare_probabilities(target, folds, incumbent, candidate):
    y = encode_target(target)
    baseline = incumbent.argmax(axis=1)
    prediction = candidate.argmax(axis=1)
    paired = (prediction == y).astype(int) - (baseline == y).astype(int)
    fold_deltas = [float(paired[np.asarray(folds) == fold].mean()) for fold in range(1, 6)]
    repair_delta = float(((prediction[y == 1] == 1).mean() - (baseline[y == 1] == 1).mean()))
    return {"accuracy": float((prediction == y).mean()),
            "incumbent_accuracy": float((baseline == y).mean()),
            "net_correct_rows": int(paired.sum()), "accuracy_delta": float(paired.mean()),
            "fold_deltas": fold_deltas, "fold_wins": sum(d > 0 for d in fold_deltas),
            "worst_fold_delta": min(fold_deltas), "repair_recall_delta": repair_delta,
            "changed_labels": int((baseline != prediction).sum()),
            "passes_guard": bool(paired.mean() > 0 and sum(d > 0 for d in fold_deltas) >= 3
                                 and min(fold_deltas) >= -0.001 and repair_delta >= -0.01)}


def cache_digest(metadata, ids, probabilities, diagnostics):
    digest = hashlib.sha256(json.dumps({"metadata": metadata, "diagnostics": diagnostics},
                                      sort_keys=True, allow_nan=False).encode())
    digest.update(np.asarray(ids, dtype="<i8").tobytes())
    digest.update(np.asarray(probabilities, dtype="<f8").tobytes())
    return digest.hexdigest()


def validate_cache(payload, metadata, ids):
    if set(payload) != {"metadata", "ids", "probabilities", "diagnostics", "digest"}:
        raise ValueError("Cache schema changed.")
    if payload["metadata"] != metadata or not np.array_equal(payload["ids"], ids):
        raise ValueError("Cache metadata or ID order changed.")
    validate_probabilities(payload["probabilities"], len(ids))
    if payload["digest"] != cache_digest(metadata, ids, payload["probabilities"], payload["diagnostics"]):
        raise ValueError("Cache content digest changed.")
    return payload["probabilities"], payload["diagnostics"]

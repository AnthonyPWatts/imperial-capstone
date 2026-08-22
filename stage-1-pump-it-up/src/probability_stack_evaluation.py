"""Evaluate a constrained meta-model over aligned class memberships."""

from __future__ import annotations

import time as _time

import numpy as _np
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.pipeline import make_pipeline as _make_pipeline
from sklearn.preprocessing import StandardScaler as _StandardScaler

from data_partitioning import PartitionedData
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation


MEMBERSHIP_CLIP = 1e-6
STACK_C = 1.0
STACK_MAX_ITERATIONS = 500
STACK_SEED = 20260821


def centred_log_membership_features(
    component_probabilities: list[_np.ndarray],
) -> _np.ndarray:
    """Return finite centred log-memberships for each component and row."""

    if len(component_probabilities) < 2:
        raise ValueError("Membership stacking requires at least two components.")
    arrays = [_np.asarray(values, dtype="float64") for values in component_probabilities]
    expected_shape = arrays[0].shape
    if len(expected_shape) != 2 or expected_shape[1] < 2:
        raise ValueError("Component memberships require rows by classes.")
    for values in arrays:
        if values.shape != expected_shape:
            raise ValueError("Component membership shapes differ.")
        if not _np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Component memberships must be finite and non-negative.")
        if not _np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Component memberships must sum to one by row.")
    stacked = _np.stack(arrays, axis=1)
    logged = _np.log(_np.clip(stacked, MEMBERSHIP_CLIP, 1.0))
    centred = logged - logged.mean(axis=2, keepdims=True)
    features = centred.reshape(expected_shape[0], -1)
    if not _np.isfinite(features).all():
        raise ValueError("Centred log-memberships contain non-finite values.")
    return features


def evaluate_cross_fitted_probability_stack(
    partitioned_data: PartitionedData,
    cross_validation: object,
    *components: CandidateEvaluation,
) -> CandidateEvaluation:
    """Fit one fixed multinomial combiner across the frozen OOF folds."""

    if len(components) < 2:
        raise ValueError("Probability stacking requires at least two components.")
    reference = components[0].out_of_fold_probabilities
    for component in components:
        if (
            component.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Probability-stack components use different folds.")
        if not component.out_of_fold_probabilities.index.equals(reference.index):
            raise ValueError("Probability-stack component rows are misaligned.")
        if tuple(component.out_of_fold_probabilities.columns) != tuple(
            reference.columns
        ):
            raise ValueError("Probability-stack component classes are misaligned.")
    features = centred_log_membership_features(
        [component.out_of_fold_probabilities.to_numpy() for component in components]
    )
    probabilities = _np.empty((len(features), reference.shape[1]), dtype="float64")
    diagnostic_rows = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        started = _time.perf_counter()
        pipeline = _make_pipeline(
            _StandardScaler(),
            _LogisticRegression(
                C=STACK_C,
                max_iter=STACK_MAX_ITERATIONS,
                solver="lbfgs",
                random_state=STACK_SEED,
            ),
        )
        pipeline.fit(
            features[training_positions],
            partitioned_data.y_development.iloc[training_positions],
        )
        classifier = pipeline.named_steps["logisticregression"]
        class_positions = [
            list(classifier.classes_).index(label) for label in reference.columns
        ]
        probabilities[validation_positions] = pipeline.predict_proba(
            features[validation_positions]
        )[:, class_positions]
        diagnostic_rows.append(
            {
                "validation_fold": fold_number,
                "total_seconds": _time.perf_counter() - started,
                "meta_features": features.shape[1],
                "iterations": int(classifier.n_iter_.max()),
                "coefficient_l2_norm": float(_np.linalg.norm(classifier.coef_)),
            }
        )
    return build_candidate_evaluation(
        model_name="Centred-log six-component logistic stack",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=diagnostic_rows,
    )

"""Cumulative-threshold helpers for the tentative three-class target order."""

from __future__ import annotations

from collections.abc import Iterable as _Iterable

import numpy as _np
import pandas as _pd
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import balanced_accuracy_score as _balanced_accuracy_score


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
ORDINAL_VALUES = {label: position for position, label in enumerate(CLASS_LABELS)}


def project_cumulative_probabilities(
    repair_or_worse: _np.ndarray,
    non_functional: _np.ndarray,
) -> _np.ndarray:
    """Project two cumulative probabilities onto P(Y >= 1) >= P(Y >= 2).

    Independent binary classifiers can cross.  For a crossing row, the closest
    pair under equal squared-error weight sets both probabilities to their mean.
    """

    lower = _validated_probability_vector(repair_or_worse, "repair_or_worse")
    upper = _validated_probability_vector(non_functional, "non_functional")
    if lower.shape != upper.shape:
        raise ValueError("Cumulative probability vectors must have the same shape.")
    projected = _np.column_stack([lower, upper]).copy()
    crossings = projected[:, 1] > projected[:, 0]
    crossing_mean = projected[crossings].mean(axis=1)
    projected[crossings, 0] = crossing_mean
    projected[crossings, 1] = crossing_mean
    return projected


def cumulative_class_probabilities(
    repair_or_worse: _np.ndarray,
    non_functional: _np.ndarray,
) -> _np.ndarray:
    """Convert coherent cumulative probabilities to three class probabilities."""

    cumulative = project_cumulative_probabilities(
        repair_or_worse,
        non_functional,
    )
    probabilities = _np.column_stack(
        [
            1.0 - cumulative[:, 0],
            cumulative[:, 0] - cumulative[:, 1],
            cumulative[:, 1],
        ]
    )
    if not _np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Ordinal class probabilities do not sum to one.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Ordinal class probabilities fall outside zero to one.")
    return probabilities


def cumulative_threshold_predictions(
    repair_or_worse: _np.ndarray,
    non_functional: _np.ndarray,
    *,
    repair_cutoff: float = 0.5,
    non_functional_cutoff: float = 0.5,
) -> _np.ndarray:
    """Apply two explicit cut-offs to coherent cumulative probabilities."""

    _validate_cutoff(repair_cutoff, "repair_cutoff")
    _validate_cutoff(non_functional_cutoff, "non_functional_cutoff")
    cumulative = project_cumulative_probabilities(
        repair_or_worse,
        non_functional,
    )
    predictions = _np.full(len(cumulative), CLASS_LABELS[1], dtype=object)
    predictions[cumulative[:, 0] < repair_cutoff] = CLASS_LABELS[0]
    predictions[cumulative[:, 1] >= non_functional_cutoff] = CLASS_LABELS[2]
    return predictions


def tune_cumulative_cutoffs(
    y_true: _pd.Series | _np.ndarray,
    repair_or_worse: _np.ndarray,
    non_functional: _np.ndarray,
    *,
    candidate_cutoffs: _Iterable[float] | None = None,
) -> dict[str, float]:
    """Select two cut-offs by nominal accuracy on a calibration partition.

    Ties prefer balanced accuracy, then the cut-offs closest to 0.5.  The caller
    is responsible for supplying a calibration partition disjoint from the rows
    on which the selected cut-offs will be evaluated.
    """

    target = _np.asarray(y_true, dtype=object)
    if len(target) == 0 or set(target) != set(CLASS_LABELS):
        raise ValueError("Calibration target must contain all three classes.")
    cumulative = project_cumulative_probabilities(
        repair_or_worse,
        non_functional,
    )
    if len(cumulative) != len(target):
        raise ValueError("Calibration probabilities and target have different rows.")
    if candidate_cutoffs is None:
        candidate_cutoffs = _np.linspace(0.20, 0.80, 13)
    candidates = tuple(float(value) for value in candidate_cutoffs)
    if not candidates:
        raise ValueError("At least one candidate cut-off is required.")
    for value in candidates:
        _validate_cutoff(value, "candidate_cutoff")

    best: tuple[tuple[float, float, float], dict[str, float]] | None = None
    for repair_cutoff in candidates:
        for non_functional_cutoff in candidates:
            predictions = cumulative_threshold_predictions(
                cumulative[:, 0],
                cumulative[:, 1],
                repair_cutoff=repair_cutoff,
                non_functional_cutoff=non_functional_cutoff,
            )
            accuracy = float(_accuracy_score(target, predictions))
            balanced_accuracy = float(
                _balanced_accuracy_score(target, predictions)
            )
            proximity = -(
                abs(repair_cutoff - 0.5) + abs(non_functional_cutoff - 0.5)
            )
            ranking = (accuracy, balanced_accuracy, proximity)
            record = {
                "repair_cutoff": repair_cutoff,
                "non_functional_cutoff": non_functional_cutoff,
                "calibration_accuracy": accuracy,
                "calibration_balanced_accuracy": balanced_accuracy,
            }
            if best is None or ranking > best[0]:
                best = (ranking, record)
    assert best is not None
    return best[1]


def ordinal_error_metrics(
    y_true: _pd.Series | _np.ndarray,
    predictions: _np.ndarray,
) -> dict[str, float]:
    """Return order-aware error diagnostics without replacing nominal accuracy."""

    actual = _np.asarray([ORDINAL_VALUES[value] for value in y_true], dtype="int64")
    predicted = _np.asarray(
        [ORDINAL_VALUES[value] for value in predictions],
        dtype="int64",
    )
    distance = _np.abs(actual - predicted)
    return {
        "mean_absolute_ordinal_error": float(distance.mean()),
        "two_step_error_rate": float(_np.mean(distance == 2)),
    }


def _validated_probability_vector(values: _np.ndarray, name: str) -> _np.ndarray:
    vector = _np.asarray(values, dtype="float64")
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional probability vector.")
    if not _np.isfinite(vector).all() or (vector < 0).any() or (vector > 1).any():
        raise ValueError(f"{name} contains invalid probabilities.")
    return vector


def _validate_cutoff(value: float, name: str) -> None:
    if not _np.isfinite(value) or value <= 0 or value >= 1:
        raise ValueError(f"{name} must fall strictly between zero and one.")

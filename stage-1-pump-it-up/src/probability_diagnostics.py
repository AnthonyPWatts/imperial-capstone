"""Probability reliability, disagreement and edge-case diagnostics."""

from __future__ import annotations

from itertools import combinations as _combinations

import numpy as _np
import pandas as _pd
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import log_loss as _log_loss


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
REPAIR_LABEL = "functional needs repair"


def probability_quality_summary(
    y_true: _pd.Series | _np.ndarray,
    probabilities: _np.ndarray,
    *,
    bins: int = 10,
) -> dict[str, float | int]:
    """Summarise accuracy, calibration, confidence and ambiguous cases."""

    target, values = _validated_inputs(y_true, probabilities)
    predicted_positions = values.argmax(axis=1)
    predictions = _np.asarray(CLASS_LABELS)[predicted_positions]
    confidence = values.max(axis=1)
    sorted_probabilities = _np.sort(values, axis=1)
    margin = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    correct = predictions == target
    one_hot = _np.column_stack([target == label for label in CLASS_LABELS])
    reliability = top_label_reliability(target, values, bins=bins)
    high_confidence = confidence >= 0.80
    low_margin = margin < 0.10
    actual_repair = target == REPAIR_LABEL
    missed_repair = actual_repair & (predictions != REPAIR_LABEL)
    confident_missed_repair = missed_repair & (confidence >= 0.70)

    return {
        "accuracy": float(_accuracy_score(target, predictions)),
        "log_loss": float(_log_loss(target, values, labels=list(CLASS_LABELS))),
        "multiclass_brier": float(_np.mean(_np.sum((values - one_hot) ** 2, axis=1))),
        "top_label_ece": float(
            _np.sum(
                reliability["rows"]
                * reliability["absolute_gap"].fillna(0.0)
            )
            / len(target)
        ),
        "mean_confidence": float(confidence.mean()),
        "mean_winning_margin": float(margin.mean()),
        "high_confidence_rows": int(high_confidence.sum()),
        "high_confidence_error_rows": int((high_confidence & ~correct).sum()),
        "high_confidence_error_rate": float(
            0.0 if not high_confidence.any() else (~correct[high_confidence]).mean()
        ),
        "low_margin_rows": int(low_margin.sum()),
        "low_margin_share": float(low_margin.mean()),
        "low_margin_accuracy": float(
            _np.nan if not low_margin.any() else correct[low_margin].mean()
        ),
        "missed_repair_rows": int(missed_repair.sum()),
        "confident_missed_repair_rows": int(confident_missed_repair.sum()),
    }


def top_label_reliability(
    y_true: _pd.Series | _np.ndarray,
    probabilities: _np.ndarray,
    *,
    bins: int = 10,
) -> _pd.DataFrame:
    """Bin winning confidence and compare it with observed correctness."""

    target, values = _validated_inputs(y_true, probabilities)
    positions = values.argmax(axis=1)
    predictions = _np.asarray(CLASS_LABELS)[positions]
    return _reliability_frame(
        values.max(axis=1),
        predictions == target,
        bins=bins,
        observed_name="accuracy",
    )


def classwise_reliability(
    y_true: _pd.Series | _np.ndarray,
    probabilities: _np.ndarray,
    *,
    label: str,
    bins: int = 10,
) -> _pd.DataFrame:
    """Compare one class's predicted probability with its observed frequency."""

    if label not in CLASS_LABELS:
        raise ValueError(f"Unknown class label: {label!r}.")
    target, values = _validated_inputs(y_true, probabilities)
    position = CLASS_LABELS.index(label)
    return _reliability_frame(
        values[:, position],
        target == label,
        bins=bins,
        observed_name="observed_frequency",
    )


def classwise_probability_scores(
    y_true: _pd.Series | _np.ndarray,
    probabilities: _np.ndarray,
    *,
    bins: int = 10,
) -> _pd.DataFrame:
    """Return one-v-rest Brier score and calibration error per class."""

    target, values = _validated_inputs(y_true, probabilities)
    rows = []
    for position, label in enumerate(CLASS_LABELS):
        actual = target == label
        reliability = classwise_reliability(
            target,
            values,
            label=label,
            bins=bins,
        )
        rows.append(
            {
                "class": label,
                "natural_share": float(actual.mean()),
                "mean_probability": float(values[:, position].mean()),
                "probability_minus_share": float(
                    values[:, position].mean() - actual.mean()
                ),
                "one_vs_rest_brier": float(
                    _np.mean((values[:, position] - actual) ** 2)
                ),
                "one_vs_rest_ece": float(
                    _np.sum(
                        reliability["rows"]
                        * reliability["absolute_gap"].fillna(0.0)
                    )
                    / len(target)
                ),
            }
        )
    return _pd.DataFrame(rows).set_index("class")


def component_disagreement_summary(
    component_probabilities: dict[str, _np.ndarray],
) -> dict[str, float | int]:
    """Summarise whether recipe members choose the same winning label."""

    if len(component_probabilities) < 2:
        raise ValueError("At least two components are required for disagreement.")
    validated = {}
    expected_rows = None
    for name, probabilities in component_probabilities.items():
        values = _np.asarray(probabilities, dtype="float64")
        if expected_rows is None:
            expected_rows = len(values)
        _validated_inputs(_np.repeat(CLASS_LABELS[0], expected_rows), values)
        if len(values) != expected_rows:
            raise ValueError("Component probability row counts differ.")
        validated[name] = values.argmax(axis=1)
    names = list(validated)
    pairwise = [
        _np.mean(validated[left] != validated[right])
        for left, right in _combinations(names, 2)
    ]
    matrix = _np.column_stack([validated[name] for name in names])
    non_unanimous = _np.any(matrix != matrix[:, [0]], axis=1)
    return {
        "components": len(names),
        "non_unanimous_rows": int(non_unanimous.sum()),
        "non_unanimous_share": float(non_unanimous.mean()),
        "mean_pairwise_disagreement": float(_np.mean(pairwise)),
        "maximum_pairwise_disagreement": float(_np.max(pairwise)),
    }


def edge_case_table(
    y_true: _pd.Series | _np.ndarray,
    probabilities: _np.ndarray,
) -> _pd.DataFrame:
    """Count interpretable groups that deserve error-analysis attention."""

    target, values = _validated_inputs(y_true, probabilities)
    positions = values.argmax(axis=1)
    predictions = _np.asarray(CLASS_LABELS)[positions]
    confidence = values.max(axis=1)
    margin = _np.sort(values, axis=1)[:, -1] - _np.sort(values, axis=1)[:, -2]
    correct = predictions == target
    masks = {
        "high-confidence correct (>=80%)": (confidence >= 0.80) & correct,
        "high-confidence error (>=80%)": (confidence >= 0.80) & ~correct,
        "close decision (margin <10 points)": margin < 0.10,
        "repair predicted correctly": (
            (predictions == REPAIR_LABEL) & (target == REPAIR_LABEL)
        ),
        "repair false alarm": (
            (predictions == REPAIR_LABEL) & (target != REPAIR_LABEL)
        ),
        "repair missed as functional": (
            (target == REPAIR_LABEL) & (predictions == CLASS_LABELS[0])
        ),
        "repair missed as non-functional": (
            (target == REPAIR_LABEL) & (predictions == CLASS_LABELS[2])
        ),
    }
    rows = []
    for group, mask in masks.items():
        rows.append(
            {
                "group": group,
                "rows": int(mask.sum()),
                "share": float(mask.mean()),
                "mean_confidence": float(
                    _np.nan if not mask.any() else confidence[mask].mean()
                ),
                "accuracy": float(
                    _np.nan if not mask.any() else correct[mask].mean()
                ),
            }
        )
    return _pd.DataFrame(rows).set_index("group")


def _reliability_frame(
    confidence: _np.ndarray,
    observed: _np.ndarray,
    *,
    bins: int,
    observed_name: str,
) -> _pd.DataFrame:
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer of at least two.")
    edges = _np.linspace(0.0, 1.0, bins + 1)
    positions = _np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for position in range(bins):
        mask = positions == position
        mean_confidence = _np.nan if not mask.any() else confidence[mask].mean()
        observed_rate = _np.nan if not mask.any() else observed[mask].mean()
        rows.append(
            {
                "bin": f"{edges[position]:.1f}–{edges[position + 1]:.1f}",
                "rows": int(mask.sum()),
                "mean_probability": float(mean_confidence),
                observed_name: float(observed_rate),
                "gap": float(observed_rate - mean_confidence),
                "absolute_gap": float(abs(observed_rate - mean_confidence)),
            }
        )
    return _pd.DataFrame(rows).set_index("bin")


def _validated_inputs(
    y_true: _pd.Series | _np.ndarray,
    probabilities: _np.ndarray,
) -> tuple[_np.ndarray, _np.ndarray]:
    target = _np.asarray(y_true, dtype=object)
    values = _np.asarray(probabilities, dtype="float64")
    if values.shape != (len(target), len(CLASS_LABELS)):
        raise ValueError("Probability matrix shape does not match the target.")
    if not _np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probability matrix contains invalid values.")
    if not _np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("Probability rows must sum to one.")
    if len(target) == 0 or not set(target).issubset(CLASS_LABELS):
        raise ValueError("Target contains no rows or unknown classes.")
    # XGBoost float32 values can leave row sums a few machine units from one.
    # Normalise after the contract check so strict scoring functions do not
    # warn about harmless representation error.
    values = values / values.sum(axis=1, keepdims=True)
    return target, values

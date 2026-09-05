"""Reusable summaries for the full-label generalisation diagnosis."""

from __future__ import annotations

from copy import deepcopy as _deepcopy
import math as _math
from numbers import Real as _Real
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd

from final_model import validate_probabilities
from gpu_model_evaluation import CLASS_LABELS


FOLD_CACHE_VERSION = 1


def make_fold_cache_payload(
    metadata: _Mapping[str, object],
    validation_ids: _np.ndarray,
    probabilities: _np.ndarray,
    fit_seconds: float,
) -> dict[str, object]:
    """Build and validate one strict, versioned OOF fold cache payload."""

    payload: dict[str, object] = {
        "cache_version": FOLD_CACHE_VERSION,
        "metadata": _deepcopy(dict(metadata)),
        "validation_ids": validation_ids.copy(),
        "probabilities": probabilities.copy(),
        "fit_seconds": fit_seconds,
    }
    validate_fold_cache_payload(
        payload,
        metadata,
        validation_ids,
    )
    return payload


def validate_fold_cache_payload(
    payload: object,
    expected_metadata: _Mapping[str, object],
    expected_validation_ids: _np.ndarray,
    *,
    cache_name: str = "generalisation-diagnosis fold cache",
) -> tuple[_np.ndarray, float]:
    """Return probabilities and timing only for an exact compatible cache."""

    required = {
        "cache_version",
        "metadata",
        "validation_ids",
        "probabilities",
        "fit_seconds",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"Malformed {cache_name}: exact schema changed.")
    if (
        type(payload["cache_version"]) is not int
        or payload["cache_version"] != FOLD_CACHE_VERSION
    ):
        raise ValueError(f"Stale {cache_name}: cache version changed.")
    if not isinstance(payload["metadata"], dict) or payload["metadata"] != dict(
        expected_metadata
    ):
        raise ValueError(f"Stale {cache_name}: pinned metadata changed.")

    validation_ids = payload["validation_ids"]
    if not isinstance(validation_ids, _np.ndarray) or validation_ids.ndim != 1:
        raise ValueError(
            f"Malformed {cache_name}: validation IDs must be one-dimensional."
        )
    expected_ids = _np.asarray(expected_validation_ids)
    if expected_ids.ndim != 1:
        raise ValueError("Expected validation IDs must be one-dimensional.")
    if (
        validation_ids.dtype != expected_ids.dtype
        or not _np.array_equal(validation_ids, expected_ids)
    ):
        raise ValueError(f"Stale {cache_name}: validation IDs changed or are misordered.")

    probabilities = payload["probabilities"]
    expected_shape = (len(expected_ids), len(CLASS_LABELS))
    if (
        not isinstance(probabilities, _np.ndarray)
        or probabilities.shape != expected_shape
        or not _np.issubdtype(probabilities.dtype, _np.floating)
    ):
        actual_shape = getattr(probabilities, "shape", None)
        raise ValueError(
            f"Malformed {cache_name}: expected probability shape "
            f"{expected_shape}, found {actual_shape}."
        )
    validate_probabilities(probabilities, len(expected_ids))

    fit_seconds = payload["fit_seconds"]
    if (
        isinstance(fit_seconds, bool)
        or not isinstance(fit_seconds, _Real)
        or not _np.isfinite(fit_seconds)
        or fit_seconds <= 0
    ):
        raise ValueError(f"Malformed {cache_name}: fit timing must be finite and positive.")
    return probabilities, float(fit_seconds)


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the 95% Wilson score interval for a binomial proportion."""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must describe a non-empty binomial sample.")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z / denominator * _math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return centre - radius, centre + radius


def summarise_predictions(predictions: _pd.DataFrame) -> tuple[_pd.DataFrame, _pd.DataFrame]:
    """Summarise accuracy and per-class recall for the two historical-ID groups."""

    required = {"actual", "predicted", "historical_local"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Prediction columns missing: {sorted(missing)!r}.")
    if predictions.empty or predictions[list(required)].isna().any().any():
        raise ValueError("Predictions must be non-empty and complete.")
    if not predictions["historical_local"].isin([True, False]).all():
        raise ValueError("historical_local must be boolean.")

    groups = (
        ("all_labelled", predictions),
        ("historical_local", predictions.loc[predictions["historical_local"]]),
        ("all_other_ids", predictions.loc[~predictions["historical_local"]]),
    )
    accuracy_rows: list[dict[str, object]] = []
    recall_rows: list[dict[str, object]] = []
    for group_name, group in groups:
        if group.empty:
            raise ValueError(f"Prediction group {group_name!r} is empty.")
        correct = group["actual"].eq(group["predicted"])
        lower, upper = wilson_interval(int(correct.sum()), len(group))
        accuracy_rows.append(
            {
                "group": group_name,
                "rows": len(group),
                "correct": int(correct.sum()),
                "accuracy": float(correct.mean()),
                "accuracy_ci95_low": lower,
                "accuracy_ci95_high": upper,
            }
        )
        for label in CLASS_LABELS:
            class_rows = group.loc[group["actual"].eq(label)]
            class_correct = class_rows["predicted"].eq(label)
            lower, upper = wilson_interval(int(class_correct.sum()), len(class_rows))
            recall_rows.append(
                {
                    "group": group_name,
                    "class": label,
                    "actual_rows": len(class_rows),
                    "true_positives": int(class_correct.sum()),
                    "recall": float(class_correct.mean()),
                    "recall_ci95_low": lower,
                    "recall_ci95_high": upper,
                }
            )
    return _pd.DataFrame(accuracy_rows), _pd.DataFrame(recall_rows)


def stratified_accuracy_difference_interval(
    predictions: _pd.DataFrame,
    *,
    seed: int,
    draws: int = 20_000,
) -> dict[str, float | int | str | bool]:
    """Parametric-bootstrap local-minus-other accuracy at fixed class counts.

    Each draw samples a true-positive count for every class/group from a
    Binomial(observed class rows, observed recall). Summed counts therefore
    estimate the accuracy contrast at each group's observed class composition
    without materialising row-level resamples.
    """

    if draws <= 0:
        raise ValueError("draws must be positive.")
    local = predictions.loc[predictions["historical_local"]].copy()
    other = predictions.loc[~predictions["historical_local"]].copy()
    if local.empty or other.empty:
        raise ValueError("Both historical-ID groups are required.")
    local["correct"] = local["actual"].eq(local["predicted"])
    other["correct"] = other["actual"].eq(other["predicted"])
    rng = _np.random.default_rng(seed)
    sampled_accuracies = []
    for group in (local, other):
        sampled_correct = _np.zeros(draws, dtype=_np.int64)
        for label in CLASS_LABELS:
            class_rows = group.loc[group["actual"].eq(label), "correct"]
            if class_rows.empty:
                raise ValueError(f"Group has no rows for class {label!r}.")
            sampled_correct += rng.binomial(
                n=len(class_rows),
                p=float(class_rows.mean()),
                size=draws,
            )
        sampled_accuracies.append(sampled_correct / len(group))
    differences = sampled_accuracies[0] - sampled_accuracies[1]
    observed = float(local["correct"].mean() - other["correct"].mean())
    low, high = _np.quantile(differences, [0.025, 0.975])
    return {
        "contrast": "historical_local_minus_all_other_ids",
        "method": "class-stratified parametric binomial-count bootstrap",
        "estimand": (
            "historical-local minus all-other OOF accuracy at each group's "
            "observed class composition"
        ),
        "draws": draws,
        "seed": seed,
        "accuracy_difference": observed,
        "difference_ci95_low": float(low),
        "difference_ci95_high": float(high),
        "paired": False,
    }

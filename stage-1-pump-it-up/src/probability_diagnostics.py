"""Probability reliability, disagreement and edge-case diagnostics."""

from __future__ import annotations

from itertools import combinations as _combinations
from pathlib import Path as _Path

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
CLOSE_MEMBERSHIP_MARGIN = 0.10
NO_MAJORITY_CONFIDENCE = 0.50
SUBSTANTIAL_SECOND_MEMBERSHIP = 0.25


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


def class_membership_table(
    identifiers: _pd.Series | _np.ndarray,
    probabilities: _pd.DataFrame | _np.ndarray,
    *,
    actual: _pd.Series | _np.ndarray | None = None,
) -> _pd.DataFrame:
    """Build a row-level three-class membership table without discarding mass."""

    ids = _pd.Series(identifiers).reset_index(drop=True)
    if ids.empty or ids.isna().any() or ids.duplicated().any():
        raise ValueError("Membership identifiers must be complete and unique.")
    values = _validated_probability_matrix(probabilities, len(ids))
    winning_positions = values.argmax(axis=1)
    runner_values = values.copy()
    runner_values[_np.arange(len(values)), winning_positions] = -1.0
    runner_positions = runner_values.argmax(axis=1)
    confidence = values[_np.arange(len(values)), winning_positions]
    runner_membership = values[_np.arange(len(values)), runner_positions]
    log_values = _np.zeros_like(values)
    _np.log(values, out=log_values, where=values > 0)
    normalised_entropy = -_np.sum(values * log_values, axis=1) / _np.log(
        len(CLASS_LABELS)
    )

    table = _pd.DataFrame({"id": ids})
    for position, label in enumerate(CLASS_LABELS):
        table[f"membership_{_column_label(label)}"] = values[:, position]
    table["predicted_class"] = _np.asarray(CLASS_LABELS)[winning_positions]
    table["confidence"] = confidence
    table["runner_up_class"] = _np.asarray(CLASS_LABELS)[runner_positions]
    table["runner_up_membership"] = runner_membership
    table["winning_margin"] = confidence - runner_membership
    table["normalised_entropy"] = normalised_entropy
    table["no_majority_membership"] = confidence < NO_MAJORITY_CONFIDENCE
    table["close_membership"] = (
        table["winning_margin"] < CLOSE_MEMBERSHIP_MARGIN
    )
    table["substantial_second_membership"] = (
        runner_membership >= SUBSTANTIAL_SECOND_MEMBERSHIP
    )
    if actual is not None:
        target = _np.asarray(actual, dtype=object)
        if len(target) != len(table) or not set(target).issubset(CLASS_LABELS):
            raise ValueError("Membership actual labels are missing or invalid.")
        table["actual_class"] = target
        table["correct"] = table["predicted_class"].to_numpy() == target
    return table


def class_membership_summary(table: _pd.DataFrame) -> _pd.DataFrame:
    """Summarise average and winning membership for each target class."""

    _validate_membership_table(table)
    rows = []
    has_actual = "actual_class" in table
    for label in CLASS_LABELS:
        membership = table[f"membership_{_column_label(label)}"]
        row = {
            "class": label,
            "mean_membership": float(membership.mean()),
            "predicted_share": float(table["predicted_class"].eq(label).mean()),
            "membership_at_least_25_share": float((membership >= 0.25).mean()),
            "membership_at_least_50_share": float((membership >= 0.50).mean()),
        }
        if has_actual:
            actual = table["actual_class"].eq(label)
            row["actual_share"] = float(actual.mean())
            row["mean_membership_when_actual"] = float(
                membership[actual].mean()
            )
        rows.append(row)
    return _pd.DataFrame(rows).set_index("class")


def membership_ambiguity_summary(table: _pd.DataFrame) -> _pd.Series:
    """Summarise confidence, entropy and explicitly defined ambiguity groups."""

    _validate_membership_table(table)
    result: dict[str, int | float] = {
        "rows": len(table),
        "mean_confidence": float(table["confidence"].mean()),
        "mean_winning_margin": float(table["winning_margin"].mean()),
        "mean_normalised_entropy": float(table["normalised_entropy"].mean()),
        "no_majority_rows": int(table["no_majority_membership"].sum()),
        "no_majority_share": float(table["no_majority_membership"].mean()),
        "close_membership_rows": int(table["close_membership"].sum()),
        "close_membership_share": float(table["close_membership"].mean()),
        "substantial_second_rows": int(
            table["substantial_second_membership"].sum()
        ),
        "substantial_second_share": float(
            table["substantial_second_membership"].mean()
        ),
    }
    if "correct" in table:
        result["accuracy"] = float(table["correct"].mean())
        for group in (
            "no_majority_membership",
            "close_membership",
            "substantial_second_membership",
        ):
            mask = table[group]
            result[f"{group}_accuracy"] = float(
                _np.nan if not mask.any() else table.loc[mask, "correct"].mean()
            )
    return _pd.Series(result, name="value")


def top_two_membership_summary(table: _pd.DataFrame) -> _pd.DataFrame:
    """Summarise the directed winning and runner-up class boundaries."""

    _validate_membership_table(table)
    aggregations: dict[str, tuple[str, str]] = {
        "rows": ("id", "size"),
        "mean_confidence": ("confidence", "mean"),
        "mean_runner_up_membership": ("runner_up_membership", "mean"),
        "mean_winning_margin": ("winning_margin", "mean"),
        "mean_normalised_entropy": ("normalised_entropy", "mean"),
    }
    if "correct" in table:
        aggregations["accuracy"] = ("correct", "mean")
    summary = table.groupby(
        ["predicted_class", "runner_up_class"],
        observed=True,
        sort=False,
    ).agg(**aggregations)
    summary["share"] = summary["rows"] / len(table)
    return summary.sort_values("rows", ascending=False)


def write_class_membership_table(
    table: _pd.DataFrame,
    destination: str | _Path,
) -> _Path:
    """Write or verify a membership CSV without overwriting different data."""

    _validate_membership_table(table)
    path = _Path(destination).resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(
            f"Membership output directory is missing: {path.parent}."
        )
    if path.exists():
        existing = _pd.read_csv(path)
        _assert_membership_frames_match(existing, table.reset_index(drop=True))
        return path
    table.to_csv(path, index=False, float_format="%.10f")
    _assert_membership_frames_match(
        _pd.read_csv(path),
        table.reset_index(drop=True),
    )
    return path


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
    values = _validated_probability_matrix(probabilities, len(target))
    if len(target) == 0 or not set(target).issubset(CLASS_LABELS):
        raise ValueError("Target contains no rows or unknown classes.")
    return target, values


def _validated_probability_matrix(
    probabilities: _pd.DataFrame | _np.ndarray,
    expected_rows: int,
) -> _np.ndarray:
    values = _np.asarray(probabilities, dtype="float64")
    if values.shape != (expected_rows, len(CLASS_LABELS)):
        raise ValueError("Probability matrix shape does not match expected rows.")
    if not _np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probability matrix contains invalid values.")
    if not _np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("Probability rows must sum to one.")
    return values / values.sum(axis=1, keepdims=True)


def _validate_membership_table(table: _pd.DataFrame) -> None:
    required = {
        "id",
        *(f"membership_{_column_label(label)}" for label in CLASS_LABELS),
        "predicted_class",
        "confidence",
        "runner_up_class",
        "runner_up_membership",
        "winning_margin",
        "normalised_entropy",
        "no_majority_membership",
        "close_membership",
        "substantial_second_membership",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Membership table is missing columns: {sorted(missing)!r}.")
    if table.empty or table["id"].isna().any() or table["id"].duplicated().any():
        raise ValueError("Membership table identifiers must be complete and unique.")
    probability_columns = [
        f"membership_{_column_label(label)}" for label in CLASS_LABELS
    ]
    _validated_probability_matrix(table[probability_columns], len(table))


def _assert_membership_frames_match(
    actual: _pd.DataFrame,
    expected: _pd.DataFrame,
) -> None:
    if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
        raise FileExistsError("Existing membership CSV has a different shape.")
    for column in expected.columns:
        if _pd.api.types.is_numeric_dtype(expected[column]):
            if not _np.allclose(
                actual[column].to_numpy(dtype="float64"),
                expected[column].to_numpy(dtype="float64"),
                equal_nan=True,
                atol=1e-9,
            ):
                raise FileExistsError(
                    f"Existing membership CSV differs in {column!r}."
                )
        elif not actual[column].astype(str).equals(expected[column].astype(str)):
            raise FileExistsError(
                f"Existing membership CSV differs in {column!r}."
            )


def _column_label(label: str) -> str:
    return label.replace(" ", "_")

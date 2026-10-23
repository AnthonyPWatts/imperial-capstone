"""Load exploration snapshots and format saved plans for any input dimension."""

from operator import index
from pathlib import Path

import numpy as np
import pandas as pd

from output_helpers import summarise_output_observations


__all__ = ["load_exploration_review", "remaining_exploration_queries"]


def load_exploration_review(data_path, *, input_columns, output_column="y",
                            round_column="source_round", expected_round_counts=None):
    """Load and validate a local snapshot, retaining full float64 precision.

    Return the existing output-review dictionary plus inputs, outputs and a
    separate plot_data frame with descending observed ranks. No source file is
    changed, and the input dimension is determined by input_columns.
    """
    input_columns = list(input_columns)
    columns = ["observation_id", round_column, *input_columns, output_column]
    if not input_columns or len(set(columns)) != len(columns):
        raise ValueError("Supply distinct identifier, round, input and output columns.")
    data = pd.read_csv(Path(data_path), float_precision="round_trip")
    if list(data.columns) != columns:
        raise ValueError(f"Expected observation columns in this order: {columns}.")
    if data.isna().any().any() or not data.observation_id.is_unique:
        raise ValueError("Require complete observations with distinct IDs.")

    review = summarise_output_observations(
        data, input_columns=input_columns, output_column=output_column,
        round_column=round_column,
    )
    inputs = data[input_columns].copy()
    if not ((inputs >= 0) & (inputs < 1)).to_numpy().all():
        raise ValueError("Observation inputs must lie in [0, 1).")
    if (expected_round_counts is not None
            and data[round_column].value_counts().to_dict() != expected_round_counts):
        raise ValueError("Observation counts do not match the expected round history.")

    plot_data = data.copy(deep=True)
    plot_data["rank"] = data[output_column].rank(ascending=False, method="min").astype(int)
    review.update(inputs=inputs, outputs=data[output_column].copy(), plot_data=plot_data)
    return review


def remaining_exploration_queries(plan, *, current_week, first_plan_week=2, precision=6):
    """Return unevaluated saved inputs with their week labels and portal strings.

    The plan's columns define its input order and dimension. Earlier rows are
    skipped by week; a completed plan returns an empty table with the same
    columns and a Portal input column. Planning coordinates retain their original
    precision, while portal strings use the requested submission precision.
    """
    current_week, first_plan_week, precision = map(
        index, (current_week, first_plan_week, precision),
    )
    if min(current_week, first_plan_week) < 1 or precision < 0:
        raise ValueError("Weeks must be positive and precision must be non-negative.")
    if (plan.empty or not plan.columns.is_unique
            or "Portal input" in plan.columns):
        raise ValueError("Supply a non-empty plan with distinct input columns.")
    points = plan.to_numpy(dtype=float)
    if not np.isfinite(points).all() or not ((points >= 0) & (points < 1)).all():
        raise ValueError("Planned inputs must be finite and lie in [0, 1).")
    rounded = points.round(precision)
    if not ((rounded >= 0) & (rounded < 1)).all():
        raise ValueError("Planned inputs leave [0, 1) at the submission precision.")

    completed = max(0, current_week - first_plan_week)
    remaining = plan.iloc[completed:].copy(deep=True)
    first_remaining_week = first_plan_week + completed
    remaining.index = pd.RangeIndex(
        first_remaining_week, first_remaining_week + len(remaining), name="Week",
    )
    remaining["Portal input"] = [
        "-".join(f"{value:.{precision}f}" for value in point)
        for point in remaining.to_numpy(dtype=float)
    ]
    return remaining

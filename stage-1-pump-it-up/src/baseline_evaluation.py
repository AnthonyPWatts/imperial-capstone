"""Evaluate non-model references across the frozen development folds."""

from __future__ import annotations

import numpy as _np
import pandas as _pd
from sklearn.dummy import DummyClassifier as _DummyClassifier
from sklearn.metrics import make_scorer as _make_scorer
from sklearn.metrics import recall_score as _recall_score
from sklearn.model_selection import PredefinedSplit as _PredefinedSplit
from sklearn.model_selection import cross_validate as _cross_validate

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData


_SCORING = {
    "accuracy": "accuracy",
    "recall_functional": _make_scorer(
        _recall_score,
        labels=["functional"],
        average="macro",
        zero_division=0,
    ),
    "recall_functional_needs_repair": _make_scorer(
        _recall_score,
        labels=["functional needs repair"],
        average="macro",
        zero_division=0,
    ),
    "recall_non_functional": _make_scorer(
        _recall_score,
        labels=["non functional"],
        average="macro",
        zero_division=0,
    ),
}

_DISPLAY_COLUMNS = {
    "test_accuracy": "accuracy",
    "test_recall_functional": "recall: functional",
    "test_recall_functional_needs_repair": "recall: functional needs repair",
    "test_recall_non_functional": "recall: non functional",
}


def evaluate_majority_reference(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
) -> tuple[_pd.DataFrame, _pd.DataFrame]:
    """Return fold and aggregate metrics for an always-majority classifier.

    Only development data is supplied to scikit-learn. The local test and
    competition rows therefore cannot contribute to fitting or scoring.
    """

    raw_results = _cross_validate(
        _DummyClassifier(strategy="most_frequent"),
        partitioned_data.X_development,
        partitioned_data.y_development,
        cv=cross_validation,
        scoring=_SCORING,
        error_score="raise",
    )

    fold_results = (
        _pd.DataFrame(raw_results)
        .loc[:, list(_DISPLAY_COLUMNS)]
        .rename(columns=_DISPLAY_COLUMNS)
    )
    fold_results.index = _pd.RangeIndex(
        start=1,
        stop=len(fold_results) + 1,
        name="validation_fold",
    )

    summary = fold_results.agg(["mean", "std", "min", "max"]).transpose()
    summary.index.name = "metric"
    _validate_majority_reference(partitioned_data, fold_results)
    return fold_results, summary


def _validate_majority_reference(
    partitioned_data: PartitionedData,
    fold_results: _pd.DataFrame,
) -> None:
    if len(fold_results) != CROSS_VALIDATION_FOLDS:
        raise ValueError(
            "Majority reference did not evaluate every frozen development fold."
        )

    expected_recall = {
        "recall: functional": 1.0,
        "recall: functional needs repair": 0.0,
        "recall: non functional": 0.0,
    }
    for metric, expected_value in expected_recall.items():
        if not _np.allclose(fold_results[metric], expected_value):
            raise ValueError(
                f"Unexpected majority-reference {metric}; expected {expected_value}."
            )

    expected_accuracy = partitioned_data.y_development.eq("functional").mean()
    if not _np.isclose(fold_results["accuracy"].mean(), expected_accuracy):
        raise ValueError(
            "Mean majority-reference accuracy does not match the development "
            "majority-class share."
        )

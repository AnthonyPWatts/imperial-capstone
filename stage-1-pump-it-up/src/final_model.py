"""Evaluate and refit the frozen forest-plus-boosting submission workflow."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
import time as _time

import numpy as _np
import pandas as _pd
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import recall_score as _recall_score
from sklearn.pipeline import Pipeline as _Pipeline

from data_partitioning import PartitionedData
from model_evaluation import make_initial_histogram_boosting_pipeline
from model_evaluation import make_random_forest_pipeline
from modelling_data import ModellingData


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
SELECTED_WORKFLOW_NAME = "equal-weight Random Forest + histogram boosting"


@_dataclass(frozen=True)
class LocalTestEvaluation:
    """One-time local-test evidence for the frozen selected workflow."""

    metrics: _pd.Series
    confusion_counts: _pd.DataFrame
    confusion_recall: _pd.DataFrame
    component_seconds: _pd.Series


@_dataclass(frozen=True)
class CompetitionPrediction:
    """Validated competition predictions from a full-data refit."""

    submission: _pd.DataFrame
    class_shares: _pd.Series
    component_seconds: _pd.Series


def evaluate_selected_workflow_on_local_test(
    partitioned_data: PartitionedData,
) -> LocalTestEvaluation:
    """Fit on development rows and score the untouched local test once."""

    probabilities, component_seconds = _fit_and_predict_components(
        partitioned_data.X_development,
        partitioned_data.y_development,
        partitioned_data.X_local_test,
    )
    predictions = _probabilities_to_labels(probabilities)
    actual = partitioned_data.y_local_test
    recalls = _recall_score(
        actual,
        predictions,
        labels=list(CLASS_LABELS),
        average=None,
        zero_division=0,
    )
    metrics = _pd.Series(
        {
            "accuracy": _accuracy_score(actual, predictions),
            **{
                f"recall: {label}": value
                for label, value in zip(CLASS_LABELS, recalls, strict=True)
            },
        },
        name="local test",
    )
    confusion_counts = _confusion_frame(
        _confusion_matrix(actual, predictions, labels=list(CLASS_LABELS))
    )
    confusion_recall = confusion_counts.div(
        confusion_counts.sum(axis="columns"),
        axis="index",
    )
    if int(confusion_counts.to_numpy().sum()) != len(actual):
        raise ValueError("Local-test confusion counts do not cover every row.")
    return LocalTestEvaluation(
        metrics=metrics,
        confusion_counts=confusion_counts,
        confusion_recall=confusion_recall,
        component_seconds=component_seconds,
    )


def fit_selected_workflow_for_competition(
    modelling_data: ModellingData,
    submission_template: _pd.DataFrame,
) -> CompetitionPrediction:
    """Refit on every labelled row and classify the competition rows."""

    _validate_submission_template(modelling_data, submission_template)
    probabilities, component_seconds = _fit_and_predict_components(
        modelling_data.X_original,
        modelling_data.y_original,
        modelling_data.X_competition,
    )
    submission = submission_template.loc[:, ["id"]].copy()
    submission["status_group"] = _probabilities_to_labels(probabilities)
    _validate_submission(modelling_data, submission)
    class_shares = (
        submission["status_group"]
        .value_counts(normalize=True)
        .reindex(CLASS_LABELS, fill_value=0.0)
        .rename("competition prediction share")
    )
    return CompetitionPrediction(
        submission=submission,
        class_shares=class_shares,
        component_seconds=component_seconds,
    )


def write_validated_submission(
    prediction: CompetitionPrediction,
    destination: _Path,
) -> _Path:
    """Write or verify the matching submission CSV without overwriting it."""

    destination = _Path(destination).resolve()
    if destination.exists():
        existing = _pd.read_csv(destination)
        expected = prediction.submission.reset_index(drop=True)
        if not existing.equals(expected):
            raise FileExistsError(
                f"Refusing to overwrite a different submission: {destination}."
            )
        return destination
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Submission directory does not exist: {destination.parent}."
        )

    prediction.submission.to_csv(destination, index=False)
    reloaded = _pd.read_csv(destination)
    if not reloaded.equals(prediction.submission.reset_index(drop=True)):
        raise ValueError("Reloaded submission does not match generated predictions.")
    return destination


def _fit_and_predict_components(
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_prediction: _pd.DataFrame,
) -> tuple[_np.ndarray, _pd.Series]:
    components = (
        ("Random Forest", make_random_forest_pipeline()),
        ("histogram boosting", make_initial_histogram_boosting_pipeline()),
    )
    probability_values = []
    elapsed_values = {}
    for name, pipeline in components:
        started = _time.perf_counter()
        pipeline.fit(X_training, y_training)
        probability_values.append(
            _ordered_probabilities(pipeline, X_prediction)
        )
        elapsed_values[name] = _time.perf_counter() - started

    blended_probabilities = _np.mean(probability_values, axis=0)
    _validate_probabilities(blended_probabilities, len(X_prediction))
    return blended_probabilities, _pd.Series(
        elapsed_values,
        name="fit and predict seconds",
    )


def _ordered_probabilities(
    pipeline: _Pipeline,
    X_prediction: _pd.DataFrame,
) -> _np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    return (
        _pd.DataFrame(
            pipeline.predict_proba(X_prediction),
            columns=classifier.classes_,
        )
        .loc[:, list(CLASS_LABELS)]
        .to_numpy()
    )


def _probabilities_to_labels(probabilities: _np.ndarray) -> _np.ndarray:
    return _np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]


def _validate_probabilities(probabilities: _np.ndarray, expected_rows: int) -> None:
    expected_shape = (expected_rows, len(CLASS_LABELS))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected probability shape {expected_shape}, found "
            f"{probabilities.shape}."
        )
    if not _np.isfinite(probabilities).all():
        raise ValueError("Selected-workflow probabilities are not all finite.")
    if not _np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Selected-workflow probabilities do not sum to one.")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("Selected-workflow probabilities fall outside zero to one.")


def _validate_submission_template(
    modelling_data: ModellingData,
    submission_template: _pd.DataFrame,
) -> None:
    if list(submission_template.columns) != ["id", "status_group"]:
        raise ValueError(
            "SubmissionFormat.csv must contain ['id', 'status_group'] in order."
        )
    if not submission_template["id"].reset_index(drop=True).equals(
        modelling_data.competition_ids.reset_index(drop=True)
    ):
        raise ValueError(
            "Competition IDs do not match the submission template in row order."
        )


def _validate_submission(
    modelling_data: ModellingData,
    submission: _pd.DataFrame,
) -> None:
    if list(submission.columns) != ["id", "status_group"]:
        raise ValueError("Generated submission columns are invalid.")
    if len(submission) != len(modelling_data.competition_ids):
        raise ValueError("Generated submission has the wrong row count.")
    if not submission["id"].reset_index(drop=True).equals(
        modelling_data.competition_ids.reset_index(drop=True)
    ):
        raise ValueError("Generated submission IDs are misordered.")
    if submission.isna().any().any():
        raise ValueError("Generated submission contains missing values.")
    if not set(submission["status_group"]).issubset(CLASS_LABELS):
        raise ValueError("Generated submission contains an invalid target label.")


def _confusion_frame(values: _np.ndarray) -> _pd.DataFrame:
    return _pd.DataFrame(
        values,
        index=_pd.Index(CLASS_LABELS, name="actual"),
        columns=_pd.Index(CLASS_LABELS, name="predicted"),
    )

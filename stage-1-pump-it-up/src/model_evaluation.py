"""Evaluate feature-based candidates on the frozen development folds."""

from __future__ import annotations

from collections.abc import Callable as _Callable
from dataclasses import dataclass as _dataclass
import time as _time

import numpy as _np
import pandas as _pd
from sklearn.ensemble import ExtraTreesClassifier as _ExtraTreesClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier as _HistGradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import recall_score as _recall_score
from sklearn.model_selection import PredefinedSplit as _PredefinedSplit
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from model_preprocessing import make_initial_preprocessor


DECISION_TREE_MAX_DEPTH = 12
DECISION_TREE_MIN_SAMPLES_LEAF = 20
DECISION_TREE_SEED = 20260820

EXTRA_TREES_ESTIMATORS = 500
EXTRA_TREES_MAX_FEATURES = 0.7
EXTRA_TREES_MIN_SAMPLES_LEAF = 1
EXTRA_TREES_SEED = 20260815

HISTOGRAM_BOOSTING_LEARNING_RATE = 0.08
HISTOGRAM_BOOSTING_ITERATIONS = 300
HISTOGRAM_BOOSTING_MAX_LEAF_NODES = 31
HISTOGRAM_BOOSTING_MIN_SAMPLES_LEAF = 20
HISTOGRAM_BOOSTING_L2_REGULARIZATION = 1.0
HISTOGRAM_BOOSTING_MAX_FEATURES = 0.8
HISTOGRAM_BOOSTING_SEED = 20260815

_CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)

_METRIC_COLUMNS = (
    "accuracy",
    "recall: functional",
    "recall: functional needs repair",
    "recall: non functional",
)


@_dataclass(frozen=True)
class CandidateEvaluation:
    """Cross-validated metrics and diagnostics for one candidate method."""

    model_name: str
    fold_metrics: _pd.DataFrame
    metric_summary: _pd.DataFrame
    diagnostics: _pd.DataFrame
    confusion_counts: _pd.DataFrame
    confusion_recall: _pd.DataFrame


def make_initial_decision_tree_pipeline() -> _Pipeline:
    """Return the initial preprocessor and constrained decision tree."""

    classifier = _DecisionTreeClassifier(
        criterion="gini",
        max_depth=DECISION_TREE_MAX_DEPTH,
        min_samples_leaf=DECISION_TREE_MIN_SAMPLES_LEAF,
        class_weight=None,
        random_state=DECISION_TREE_SEED,
    )
    return _make_pipeline(classifier)


def make_initial_extra_trees_pipeline() -> _Pipeline:
    """Return the initial preprocessor and the prior Extra Trees setup."""

    classifier = _ExtraTreesClassifier(
        n_estimators=EXTRA_TREES_ESTIMATORS,
        criterion="gini",
        max_features=EXTRA_TREES_MAX_FEATURES,
        min_samples_leaf=EXTRA_TREES_MIN_SAMPLES_LEAF,
        bootstrap=False,
        class_weight=None,
        n_jobs=-1,
        random_state=EXTRA_TREES_SEED,
    )
    return _make_pipeline(classifier)


def make_initial_histogram_boosting_pipeline() -> _Pipeline:
    """Return dense initial preprocessing and the prior boosting setup."""

    classifier = _HistGradientBoostingClassifier(
        learning_rate=HISTOGRAM_BOOSTING_LEARNING_RATE,
        max_iter=HISTOGRAM_BOOSTING_ITERATIONS,
        max_leaf_nodes=HISTOGRAM_BOOSTING_MAX_LEAF_NODES,
        min_samples_leaf=HISTOGRAM_BOOSTING_MIN_SAMPLES_LEAF,
        l2_regularization=HISTOGRAM_BOOSTING_L2_REGULARIZATION,
        max_features=HISTOGRAM_BOOSTING_MAX_FEATURES,
        early_stopping=False,
        class_weight=None,
        random_state=HISTOGRAM_BOOSTING_SEED,
    )
    return _make_pipeline(classifier, sparse_output=False)


def summarise_initial_decision_tree() -> _pd.DataFrame:
    """Return the fixed decision-tree choices and their purpose."""

    rows = [
        {
            "setting": "criterion",
            "value": "gini",
            "purpose": "Standard transparent multiclass split criterion",
        },
        {
            "setting": "max_depth",
            "value": DECISION_TREE_MAX_DEPTH,
            "purpose": "Limit memorisation and keep the first tree inspectable",
        },
        {
            "setting": "min_samples_leaf",
            "value": DECISION_TREE_MIN_SAMPLES_LEAF,
            "purpose": "Prevent leaves supported by only a handful of rows",
        },
        {
            "setting": "class_weight",
            "value": "None",
            "purpose": "Measure unweighted behaviour before imbalance experiments",
        },
        {
            "setting": "random_state",
            "value": DECISION_TREE_SEED,
            "purpose": "Make tied split choices reproducible",
        },
    ]
    return _pd.DataFrame(rows).set_index("setting")


def summarise_initial_extra_trees() -> _pd.DataFrame:
    """Return the fixed Extra Trees choices and their comparison purpose."""

    rows = [
        {
            "setting": "n_estimators",
            "value": EXTRA_TREES_ESTIMATORS,
            "purpose": "Reuse the prior early-experiment forest size",
        },
        {
            "setting": "max_features",
            "value": EXTRA_TREES_MAX_FEATURES,
            "purpose": "Reuse the prior feature subsampling rate",
        },
        {
            "setting": "min_samples_leaf",
            "value": EXTRA_TREES_MIN_SAMPLES_LEAF,
            "purpose": "Reuse the prior unrestricted leaf size",
        },
        {
            "setting": "bootstrap",
            "value": False,
            "purpose": "Use the complete fold-training sample for every tree",
        },
        {
            "setting": "class_weight",
            "value": "None",
            "purpose": "Isolate model capacity before imbalance experiments",
        },
        {
            "setting": "random_state",
            "value": EXTRA_TREES_SEED,
            "purpose": "Match the reproducible early-experiment configuration",
        },
    ]
    return _pd.DataFrame(rows).set_index("setting")


def summarise_initial_histogram_boosting() -> _pd.DataFrame:
    """Return the fixed boosting choices and their comparison purpose."""

    rows = [
        {
            "setting": "learning_rate",
            "value": HISTOGRAM_BOOSTING_LEARNING_RATE,
            "purpose": "Reuse the prior early-experiment shrinkage rate",
        },
        {
            "setting": "max_iter",
            "value": HISTOGRAM_BOOSTING_ITERATIONS,
            "purpose": "Reuse the prior number of boosting iterations",
        },
        {
            "setting": "max_leaf_nodes",
            "value": HISTOGRAM_BOOSTING_MAX_LEAF_NODES,
            "purpose": "Reuse the prior per-tree complexity limit",
        },
        {
            "setting": "min_samples_leaf",
            "value": HISTOGRAM_BOOSTING_MIN_SAMPLES_LEAF,
            "purpose": "Reuse the prior minimum leaf support",
        },
        {
            "setting": "l2_regularization",
            "value": HISTOGRAM_BOOSTING_L2_REGULARIZATION,
            "purpose": "Reuse the prior leaf-value regularisation",
        },
        {
            "setting": "max_features",
            "value": HISTOGRAM_BOOSTING_MAX_FEATURES,
            "purpose": "Reuse the prior feature subsampling rate",
        },
        {
            "setting": "early_stopping",
            "value": False,
            "purpose": "Keep every fold on the same fixed iteration budget",
        },
        {
            "setting": "random_state",
            "value": HISTOGRAM_BOOSTING_SEED,
            "purpose": "Match the reproducible early-experiment configuration",
        },
    ]
    return _pd.DataFrame(rows).set_index("setting")


def evaluate_initial_decision_tree(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
) -> CandidateEvaluation:
    """Fit and score the constrained tree on every development fold."""

    return _evaluate_candidate(
        model_name="decision tree",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        pipeline_factory=make_initial_decision_tree_pipeline,
        diagnostics_factory=_decision_tree_diagnostics,
    )


def evaluate_initial_extra_trees(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
) -> CandidateEvaluation:
    """Fit and score the prior Extra Trees setup on every development fold."""

    return _evaluate_candidate(
        model_name="Extra Trees",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        pipeline_factory=make_initial_extra_trees_pipeline,
        diagnostics_factory=_extra_trees_diagnostics,
    )


def evaluate_initial_histogram_boosting(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
) -> CandidateEvaluation:
    """Fit and score the prior boosting setup on every development fold."""

    return _evaluate_candidate(
        model_name="histogram boosting",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        pipeline_factory=make_initial_histogram_boosting_pipeline,
        diagnostics_factory=_histogram_boosting_diagnostics,
        record_elapsed=True,
    )


def compare_candidate_evaluations(
    majority_summary: _pd.DataFrame,
    *candidate_evaluations: CandidateEvaluation,
) -> _pd.DataFrame:
    """Return mean metrics for the reference and candidate methods."""

    comparison_columns = {
        "majority reference": majority_summary.loc[
            list(_METRIC_COLUMNS),
            "mean",
        ]
    }
    for evaluation in candidate_evaluations:
        if evaluation.model_name in comparison_columns:
            raise ValueError(
                f"Duplicate comparison name: {evaluation.model_name!r}."
            )
        comparison_columns[evaluation.model_name] = evaluation.metric_summary.loc[
            list(_METRIC_COLUMNS),
            "mean",
        ]

    comparison = _pd.DataFrame(comparison_columns)
    comparison.index.name = "metric"
    return comparison


def compare_with_majority_reference(
    majority_summary: _pd.DataFrame,
    candidate_evaluation: CandidateEvaluation,
) -> _pd.DataFrame:
    """Return one candidate's mean metrics and change from the reference."""

    comparison = compare_candidate_evaluations(
        majority_summary,
        candidate_evaluation,
    )
    comparison["absolute change"] = (
        comparison[candidate_evaluation.model_name]
        - comparison["majority reference"]
    )
    return comparison


def _make_pipeline(
    classifier: object,
    *,
    sparse_output: bool = True,
) -> _Pipeline:
    return _Pipeline(
        steps=[
            (
                "preprocessing",
                make_initial_preprocessor(sparse_output=sparse_output),
            ),
            ("classifier", classifier),
        ]
    )


def _evaluate_candidate(
    *,
    model_name: str,
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
    pipeline_factory: _Callable[[], _Pipeline],
    diagnostics_factory: _Callable[[_Pipeline], dict[str, int | float]],
    record_elapsed: bool = False,
) -> CandidateEvaluation:
    metric_rows = []
    diagnostic_rows = []
    aggregate_confusion = _np.zeros(
        (len(_CLASS_LABELS), len(_CLASS_LABELS)),
        dtype="int64",
    )

    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        y_validation = partitioned_data.y_development.iloc[validation_positions]

        fold_started = _time.perf_counter()
        pipeline = pipeline_factory()
        fit_started = _time.perf_counter()
        pipeline.fit(X_training, y_training)
        fit_seconds = _time.perf_counter() - fit_started
        predict_started = _time.perf_counter()
        predictions = pipeline.predict(X_validation)
        predict_seconds = _time.perf_counter() - predict_started
        total_seconds = _time.perf_counter() - fold_started

        recalls = _recall_score(
            y_validation,
            predictions,
            labels=list(_CLASS_LABELS),
            average=None,
            zero_division=0,
        )
        metric_rows.append(
            {
                "validation_fold": fold_number,
                "accuracy": _accuracy_score(y_validation, predictions),
                **{
                    f"recall: {label}": value
                    for label, value in zip(_CLASS_LABELS, recalls, strict=True)
                },
            }
        )
        diagnostics = diagnostics_factory(pipeline)
        if record_elapsed:
            diagnostics.update(
                {
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "total_seconds": total_seconds,
                }
            )
            print(
                f"Completed {model_name} fold {fold_number}/"
                f"{CROSS_VALIDATION_FOLDS} in {total_seconds:.1f} seconds."
            )
        diagnostic_rows.append(
            {"validation_fold": fold_number, **diagnostics}
        )
        aggregate_confusion += _confusion_matrix(
            y_validation,
            predictions,
            labels=list(_CLASS_LABELS),
        )

    fold_metrics = _pd.DataFrame(metric_rows).set_index("validation_fold")
    metric_summary = fold_metrics.loc[:, _METRIC_COLUMNS].agg(
        ["mean", "std", "min", "max"]
    ).transpose()
    metric_summary.index.name = "metric"
    diagnostics = _pd.DataFrame(diagnostic_rows).set_index("validation_fold")
    confusion_counts = _confusion_frame(aggregate_confusion)
    confusion_recall = confusion_counts.div(
        confusion_counts.sum(axis="columns"),
        axis="index",
    )

    evaluation = CandidateEvaluation(
        model_name=model_name,
        fold_metrics=fold_metrics,
        metric_summary=metric_summary,
        diagnostics=diagnostics,
        confusion_counts=confusion_counts,
        confusion_recall=confusion_recall,
    )
    _validate_candidate_evaluation(partitioned_data, evaluation)
    return evaluation


def _decision_tree_diagnostics(pipeline: _Pipeline) -> dict[str, int | float]:
    classifier = pipeline.named_steps["classifier"]
    return {
        "actual_depth": classifier.get_depth(),
        "leaves": classifier.get_n_leaves(),
    }


def _extra_trees_diagnostics(pipeline: _Pipeline) -> dict[str, int | float]:
    classifier = pipeline.named_steps["classifier"]
    depths = _np.array(
        [estimator.get_depth() for estimator in classifier.estimators_]
    )
    leaves = _np.array(
        [estimator.get_n_leaves() for estimator in classifier.estimators_]
    )
    return {
        "trees": len(classifier.estimators_),
        "mean_depth": depths.mean(),
        "maximum_depth": depths.max(),
        "mean_leaves": leaves.mean(),
    }


def _histogram_boosting_diagnostics(
    pipeline: _Pipeline,
) -> dict[str, int | float]:
    classifier = pipeline.named_steps["classifier"]
    return {
        "iterations": classifier.n_iter_,
        "trees_per_iteration": classifier.n_trees_per_iteration_,
        "total_trees": classifier.n_iter_ * classifier.n_trees_per_iteration_,
    }


def _confusion_frame(values: _np.ndarray) -> _pd.DataFrame:
    return _pd.DataFrame(
        values,
        index=_pd.Index(_CLASS_LABELS, name="actual"),
        columns=_pd.Index(_CLASS_LABELS, name="predicted"),
    )


def _validate_candidate_evaluation(
    partitioned_data: PartitionedData,
    evaluation: CandidateEvaluation,
) -> None:
    if len(evaluation.fold_metrics) != CROSS_VALIDATION_FOLDS:
        raise ValueError("The candidate did not evaluate every development fold.")
    if not _np.isfinite(evaluation.fold_metrics.to_numpy()).all():
        raise ValueError("Candidate fold metrics contain non-finite values.")
    if not _np.isfinite(evaluation.diagnostics.to_numpy()).all():
        raise ValueError("Candidate diagnostics contain non-finite values.")
    if int(evaluation.confusion_counts.to_numpy().sum()) != len(
        partitioned_data.y_development
    ):
        raise ValueError(
            "The aggregate confusion matrix does not cover every development row."
        )

    expected_actual_counts = (
        partitioned_data.y_development.value_counts()
        .reindex(_CLASS_LABELS)
        .to_numpy()
    )
    actual_counts = evaluation.confusion_counts.sum(axis="columns").to_numpy()
    if not _np.array_equal(actual_counts, expected_actual_counts):
        raise ValueError(
            "Aggregate confusion-matrix rows do not match development labels."
        )

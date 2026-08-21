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
from sklearn.ensemble import RandomForestClassifier as _RandomForestClassifier
from sklearn.ensemble import StackingClassifier as _StackingClassifier
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import recall_score as _recall_score
from sklearn.model_selection import PredefinedSplit as _PredefinedSplit
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold
from sklearn.naive_bayes import GaussianNB as _GaussianNB
from sklearn.neighbors import KNeighborsClassifier as _KNeighborsClassifier
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

LOGISTIC_REGRESSION_C = 1.0
LOGISTIC_REGRESSION_MAX_ITERATIONS = 500

K_NEAREST_NEIGHBOURS = 25

GAUSSIAN_NAIVE_BAYES_VAR_SMOOTHING = 1e-9

RANDOM_FOREST_ESTIMATORS = 300
RANDOM_FOREST_MAX_FEATURES = "sqrt"
RANDOM_FOREST_MIN_SAMPLES_LEAF = 1
RANDOM_FOREST_SEED = 20260821

CALIBRATED_STACK_INNER_FOLDS = 4
CALIBRATED_STACK_C = 1.0
CALIBRATED_STACK_MAX_ITERATIONS = 500
CALIBRATED_STACK_SEED = 20260821

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
    cross_validation_fingerprint: str
    fold_metrics: _pd.DataFrame
    metric_summary: _pd.DataFrame
    diagnostics: _pd.DataFrame
    out_of_fold_probabilities: _pd.DataFrame
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


def make_logistic_regression_pipeline() -> _Pipeline:
    """Return scaled preprocessing and a regularised linear classifier."""

    classifier = _LogisticRegression(
        C=LOGISTIC_REGRESSION_C,
        solver="lbfgs",
        max_iter=LOGISTIC_REGRESSION_MAX_ITERATIONS,
        class_weight=None,
    )
    return _make_pipeline(classifier, scale_numeric=True)


def make_k_nearest_neighbours_pipeline() -> _Pipeline:
    """Return scaled preprocessing and a brute-force KNN classifier."""

    classifier = _KNeighborsClassifier(
        n_neighbors=K_NEAREST_NEIGHBOURS,
        weights="distance",
        metric="minkowski",
        p=2,
        n_jobs=-1,
    )
    return _make_pipeline(classifier, scale_numeric=True)


def make_gaussian_naive_bayes_pipeline() -> _Pipeline:
    """Return dense scaled preprocessing and Gaussian naïve Bayes."""

    classifier = _GaussianNB(
        var_smoothing=GAUSSIAN_NAIVE_BAYES_VAR_SMOOTHING,
    )
    return _make_pipeline(
        classifier,
        sparse_output=False,
        scale_numeric=True,
    )


def make_random_forest_pipeline(
    *,
    preprocessor_factory: _Callable[..., _Pipeline] = make_initial_preprocessor,
) -> _Pipeline:
    """Return initial preprocessing and a conventional Random Forest."""

    classifier = _RandomForestClassifier(
        n_estimators=RANDOM_FOREST_ESTIMATORS,
        criterion="gini",
        max_features=RANDOM_FOREST_MAX_FEATURES,
        min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
        bootstrap=True,
        class_weight=None,
        n_jobs=-1,
        random_state=RANDOM_FOREST_SEED,
    )
    return _make_pipeline(
        classifier,
        preprocessor_factory=preprocessor_factory,
    )


def make_calibrated_stack_pipeline() -> _Pipeline:
    """Return a leakage-safe probability stack over the selected components.

    Each component owns its preprocessing pipeline. ``StackingClassifier``
    creates inner out-of-fold probabilities on the current outer-training
    rows before fitting the regularised multinomial combiner, then refits both
    components on all outer-training rows. The frozen outer validation fold is
    therefore absent from every learned part of that fold's stack.
    """

    inner_cross_validation = _StratifiedKFold(
        n_splits=CALIBRATED_STACK_INNER_FOLDS,
        shuffle=True,
        random_state=CALIBRATED_STACK_SEED,
    )
    combiner = _LogisticRegression(
        C=CALIBRATED_STACK_C,
        solver="lbfgs",
        max_iter=CALIBRATED_STACK_MAX_ITERATIONS,
        class_weight=None,
        random_state=CALIBRATED_STACK_SEED,
    )
    classifier = _StackingClassifier(
        estimators=[
            ("random_forest", make_random_forest_pipeline()),
            (
                "histogram_boosting",
                make_initial_histogram_boosting_pipeline(),
            ),
        ],
        final_estimator=combiner,
        cv=inner_cross_validation,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=None,
    )
    return _Pipeline(steps=[("classifier", classifier)])


def summarise_classifier_screen() -> _pd.DataFrame:
    """Return the deliberately varied methods in the broad first screen."""

    rows = [
        {
            "method": "logistic regression",
            "family": "linear",
            "key_setting": f"L2 regularisation, C={LOGISTIC_REGRESSION_C}",
            "reason": "Test whether mostly additive effects are sufficient",
        },
        {
            "method": "KNN",
            "family": "neighbour",
            "key_setting": (
                f"k={K_NEAREST_NEIGHBOURS}, distance weighted, scaled numeric"
            ),
            "reason": "Show the effect of a local distance-based method",
        },
        {
            "method": "Gaussian naïve Bayes",
            "family": "probabilistic",
            "key_setting": (
                f"var_smoothing={GAUSSIAN_NAIVE_BAYES_VAR_SMOOTHING:g}"
            ),
            "reason": "Test a fast model with strong independence assumptions",
        },
        {
            "method": "Random Forest",
            "family": "bagged trees",
            "key_setting": (
                f"{RANDOM_FOREST_ESTIMATORS} trees, sqrt feature sampling"
            ),
            "reason": "Compare conventional bagging with Extra Trees",
        },
    ]
    return _pd.DataFrame(rows).set_index("method")


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
    *,
    record_elapsed: bool = False,
) -> CandidateEvaluation:
    """Fit and score the prior Extra Trees setup on every development fold."""

    return _evaluate_candidate(
        model_name="Extra Trees",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        pipeline_factory=make_initial_extra_trees_pipeline,
        diagnostics_factory=_extra_trees_diagnostics,
        record_elapsed=record_elapsed,
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


def evaluate_logistic_regression(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
) -> CandidateEvaluation:
    """Fit and score regularised logistic regression on every fold."""

    return _evaluate_candidate(
        model_name="logistic regression",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        pipeline_factory=make_logistic_regression_pipeline,
        diagnostics_factory=_logistic_regression_diagnostics,
        record_elapsed=True,
    )


def evaluate_k_nearest_neighbours(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
) -> CandidateEvaluation:
    """Fit and score scaled KNN on every development fold."""

    return _evaluate_candidate(
        model_name="KNN",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        pipeline_factory=make_k_nearest_neighbours_pipeline,
        diagnostics_factory=_k_nearest_neighbours_diagnostics,
        record_elapsed=True,
    )


def evaluate_gaussian_naive_bayes(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
) -> CandidateEvaluation:
    """Fit and score Gaussian naïve Bayes on every development fold."""

    return _evaluate_candidate(
        model_name="Gaussian naïve Bayes",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        pipeline_factory=make_gaussian_naive_bayes_pipeline,
        diagnostics_factory=_gaussian_naive_bayes_diagnostics,
        record_elapsed=True,
    )


def evaluate_random_forest(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
    *,
    preprocessor_factory: _Callable[..., _Pipeline] = make_initial_preprocessor,
    model_name: str = "Random Forest",
) -> CandidateEvaluation:
    """Fit and score a conventional Random Forest on every fold."""

    return _evaluate_candidate(
        model_name=model_name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        pipeline_factory=lambda: make_random_forest_pipeline(
            preprocessor_factory=preprocessor_factory,
        ),
        diagnostics_factory=_random_forest_diagnostics,
        record_elapsed=True,
    )


def evaluate_calibrated_stack(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
) -> CandidateEvaluation:
    """Evaluate the nested, cross-fitted forest-plus-boosting stack."""

    return _evaluate_candidate(
        model_name="calibrated stack",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        pipeline_factory=make_calibrated_stack_pipeline,
        diagnostics_factory=_calibrated_stack_diagnostics,
        record_elapsed=True,
    )


def evaluate_equal_weight_soft_vote(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
    *component_evaluations: CandidateEvaluation,
    model_name: str = "soft vote",
) -> CandidateEvaluation:
    """Average aligned out-of-fold probabilities from two or more candidates."""

    if len(component_evaluations) < 2:
        raise ValueError("A soft vote requires at least two component models.")
    component_weight = 1.0 / len(component_evaluations)
    return evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        *component_evaluations,
        weights=[component_weight] * len(component_evaluations),
        model_name=model_name,
    )


def evaluate_weighted_soft_vote(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
    *component_evaluations: CandidateEvaluation,
    weights: list[float] | tuple[float, ...],
    model_name: str,
) -> CandidateEvaluation:
    """Combine aligned out-of-fold probabilities with fixed weights."""

    if len(component_evaluations) < 2:
        raise ValueError("A soft vote requires at least two component models.")
    component_names = [
        evaluation.model_name for evaluation in component_evaluations
    ]
    if len(component_names) != len(set(component_names)):
        raise ValueError("Soft-vote component names must be unique.")
    for evaluation in component_evaluations:
        if (
            evaluation.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError(
                f"Mismatched cross-validation design for {evaluation.model_name}."
            )
        if tuple(evaluation.out_of_fold_probabilities.columns) != _CLASS_LABELS:
            raise ValueError(
                f"Unexpected probability columns for {evaluation.model_name}."
            )
        if not evaluation.out_of_fold_probabilities.index.equals(
            partitioned_data.y_development.index
        ):
            raise ValueError(
                f"Misaligned out-of-fold rows for {evaluation.model_name}."
            )

    weight_values = _np.asarray(weights, dtype="float64")
    if weight_values.shape != (len(component_evaluations),):
        raise ValueError(
            "Soft-vote weights must match the number of components."
        )
    if not _np.isfinite(weight_values).all() or (weight_values < 0).any():
        raise ValueError("Soft-vote weights must be finite and non-negative.")
    if not _np.isclose(weight_values.sum(), 1.0):
        raise ValueError("Soft-vote weights must sum to one.")

    blended_probabilities = _np.average(
        _np.stack(
            [
                evaluation.out_of_fold_probabilities.to_numpy()
                for evaluation in component_evaluations
            ]
        ),
        axis=0,
        weights=weight_values,
    )
    timed_components = all(
        "total_seconds" in evaluation.diagnostics
        for evaluation in component_evaluations
    )
    diagnostic_rows = []
    for fold_number in range(1, CROSS_VALIDATION_FOLDS + 1):
        row: dict[str, int | float] = {
            "validation_fold": fold_number,
            **{
                f"{evaluation.model_name} weight": weight
                for evaluation, weight in zip(
                    component_evaluations,
                    weight_values,
                    strict=True,
                )
            },
        }
        if timed_components:
            row["component_seconds"] = sum(
                evaluation.diagnostics.loc[fold_number, "total_seconds"]
                for evaluation in component_evaluations
            )
        diagnostic_rows.append(row)

    return build_candidate_evaluation(
        model_name=model_name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=blended_probabilities,
        diagnostic_rows=diagnostic_rows,
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


def compare_candidate_diversity(
    partitioned_data: PartitionedData,
    *candidate_evaluations: CandidateEvaluation,
) -> _pd.DataFrame:
    """Compare pairwise errors from aligned out-of-fold predictions."""

    prediction_columns = {}
    class_labels = _np.asarray(_CLASS_LABELS)
    for evaluation in candidate_evaluations:
        _validate_candidate_evaluation(partitioned_data, evaluation)
        prediction_columns[evaluation.model_name] = class_labels[
            evaluation.out_of_fold_probabilities.to_numpy().argmax(axis=1)
        ]

    predictions = _pd.DataFrame(
        prediction_columns,
        index=partitioned_data.y_development.index,
    )
    actual = partitioned_data.y_development
    rows = []
    model_names = list(predictions.columns)
    for left_position, left_name in enumerate(model_names):
        for right_name in model_names[left_position + 1 :]:
            left_correct = predictions[left_name].eq(actual)
            right_correct = predictions[right_name].eq(actual)
            rows.append(
                {
                    "left_model": left_name,
                    "right_model": right_name,
                    "disagreement": predictions[left_name]
                    .ne(predictions[right_name])
                    .mean(),
                    "left_only_correct": (left_correct & ~right_correct).mean(),
                    "right_only_correct": (~left_correct & right_correct).mean(),
                    "both_wrong": (~left_correct & ~right_correct).mean(),
                }
            )
    return _pd.DataFrame(rows).set_index(["left_model", "right_model"])


def _make_pipeline(
    classifier: object,
    *,
    sparse_output: bool = True,
    scale_numeric: bool = False,
    preprocessor_factory: _Callable[..., _Pipeline] = make_initial_preprocessor,
) -> _Pipeline:
    return _Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor_factory(
                    sparse_output=sparse_output,
                    scale_numeric=scale_numeric,
                ),
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
    diagnostic_rows = []
    probability_values = _np.full(
        (len(partitioned_data.y_development), len(_CLASS_LABELS)),
        fill_value=_np.nan,
    )

    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]

        fold_started = _time.perf_counter()
        pipeline = pipeline_factory()
        fit_started = _time.perf_counter()
        pipeline.fit(X_training, y_training)
        fit_seconds = _time.perf_counter() - fit_started
        predict_started = _time.perf_counter()
        classifier = pipeline.named_steps["classifier"]
        fold_probabilities = _pd.DataFrame(
            pipeline.predict_proba(X_validation),
            columns=classifier.classes_,
        ).loc[:, list(_CLASS_LABELS)]
        predict_seconds = _time.perf_counter() - predict_started
        total_seconds = _time.perf_counter() - fold_started
        probability_values[validation_positions] = fold_probabilities.to_numpy()
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

    return build_candidate_evaluation(
        model_name=model_name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostic_rows,
    )


def build_candidate_evaluation(
    *,
    model_name: str,
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
    probability_values: _np.ndarray,
    diagnostic_rows: list[dict[str, int | float]],
) -> CandidateEvaluation:
    """Build and validate fold metrics from aligned OOF probabilities."""

    metric_rows = []
    aggregate_confusion = _np.zeros(
        (len(_CLASS_LABELS), len(_CLASS_LABELS)),
        dtype="int64",
    )
    class_labels = _np.asarray(_CLASS_LABELS)

    for fold_number, (_, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        y_validation = partitioned_data.y_development.iloc[validation_positions]
        predictions = class_labels[
            probability_values[validation_positions].argmax(axis=1)
        ]
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
    out_of_fold_probabilities = _pd.DataFrame(
        probability_values,
        index=partitioned_data.y_development.index.copy(),
        columns=_pd.Index(_CLASS_LABELS, name="predicted_class"),
    )
    confusion_counts = _confusion_frame(aggregate_confusion)
    confusion_recall = confusion_counts.div(
        confusion_counts.sum(axis="columns"),
        axis="index",
    )

    evaluation = CandidateEvaluation(
        model_name=model_name,
        cross_validation_fingerprint=(
            partitioned_data.cross_validation_fingerprint
        ),
        fold_metrics=fold_metrics,
        metric_summary=metric_summary,
        diagnostics=diagnostics,
        out_of_fold_probabilities=out_of_fold_probabilities,
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
    return _forest_diagnostics(pipeline)


def _histogram_boosting_diagnostics(
    pipeline: _Pipeline,
) -> dict[str, int | float]:
    classifier = pipeline.named_steps["classifier"]
    return {
        "iterations": classifier.n_iter_,
        "trees_per_iteration": classifier.n_trees_per_iteration_,
        "total_trees": classifier.n_iter_ * classifier.n_trees_per_iteration_,
    }


def _logistic_regression_diagnostics(
    pipeline: _Pipeline,
) -> dict[str, int | float]:
    classifier = pipeline.named_steps["classifier"]
    return {
        "iterations": int(classifier.n_iter_.max()),
        "features": classifier.coef_.shape[1],
        "coefficient_l2_norm": float(_np.linalg.norm(classifier.coef_)),
    }


def _k_nearest_neighbours_diagnostics(
    pipeline: _Pipeline,
) -> dict[str, int | float]:
    classifier = pipeline.named_steps["classifier"]
    return {
        "neighbours": classifier.n_neighbors,
        "training_rows": classifier.n_samples_fit_,
    }


def _gaussian_naive_bayes_diagnostics(
    pipeline: _Pipeline,
) -> dict[str, int | float]:
    classifier = pipeline.named_steps["classifier"]
    return {
        "features": classifier.theta_.shape[1],
        "var_smoothing": classifier.var_smoothing,
    }


def _random_forest_diagnostics(
    pipeline: _Pipeline,
) -> dict[str, int | float]:
    return _forest_diagnostics(pipeline)


def _calibrated_stack_diagnostics(
    pipeline: _Pipeline,
) -> dict[str, int | float]:
    classifier = pipeline.named_steps["classifier"]
    combiner = classifier.final_estimator_
    return {
        "inner_folds": CALIBRATED_STACK_INNER_FOLDS,
        "meta_features": combiner.coef_.shape[1],
        "combiner_iterations": int(combiner.n_iter_.max()),
        "coefficient_l2_norm": float(_np.linalg.norm(combiner.coef_)),
    }


def _forest_diagnostics(pipeline: _Pipeline) -> dict[str, int | float]:
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
    if (
        evaluation.cross_validation_fingerprint
        != partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError("Candidate evaluation uses the wrong cross-validation design.")
    if tuple(evaluation.out_of_fold_probabilities.columns) != _CLASS_LABELS:
        raise ValueError("Candidate probability columns do not match class labels.")
    if not evaluation.out_of_fold_probabilities.index.equals(
        partitioned_data.y_development.index
    ):
        raise ValueError("Candidate probability rows do not match development rows.")
    probability_values = evaluation.out_of_fold_probabilities.to_numpy()
    if not _np.isfinite(probability_values).all():
        raise ValueError("Candidate probabilities contain non-finite values.")
    if not _np.allclose(probability_values.sum(axis=1), 1.0):
        raise ValueError("Candidate probabilities do not sum to one by row.")
    if (probability_values < 0).any() or (probability_values > 1).any():
        raise ValueError("Candidate probabilities fall outside zero to one.")
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

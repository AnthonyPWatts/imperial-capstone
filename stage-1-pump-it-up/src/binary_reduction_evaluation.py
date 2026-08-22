"""Evaluate models trained without repair rows against the full three-class task."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import time as _time
from typing import Callable as _Callable
from typing import Mapping as _Mapping

from catboost import CatBoostClassifier as _CatBoostClassifier
from lightgbm import early_stopping as _lgb_early_stopping
from lightgbm import LGBMClassifier as _LGBMClassifier
from lightgbm import log_evaluation as _lgb_log_evaluation
import numpy as _np
import pandas as _pd
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import log_loss as _log_loss
from sklearn.metrics import recall_score as _recall_score
from sklearn.model_selection import train_test_split as _train_test_split
from xgboost import XGBClassifier as _XGBClassifier

from ann_evaluation import ANN_CANDIDATES
from ann_evaluation import make_ann_pipeline
from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import engineer_initial_features
from gpu_model_evaluation import CATBOOST_ITERATION_CAP
from gpu_model_evaluation import CATBOOST_STOPPING_ROUNDS
from gpu_model_evaluation import CATBOOST_VARIANTS
from gpu_model_evaluation import INNER_STOP_FRACTION
from gpu_model_evaluation import INNER_STOP_SEED
from gpu_model_evaluation import LIGHTGBM_ITERATION_CAP
from gpu_model_evaluation import LIGHTGBM_STOPPING_ROUNDS
from gpu_model_evaluation import LIGHTGBM_VARIANTS
from gpu_model_evaluation import XGBOOST_ITERATION_CAP
from gpu_model_evaluation import XGBOOST_STOPPING_ROUNDS
from gpu_model_evaluation import XGBOOST_VARIANTS
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_evaluation import evaluate_weighted_soft_vote
from model_evaluation import make_gaussian_naive_bayes_pipeline
from model_evaluation import make_initial_decision_tree_pipeline
from model_evaluation import make_initial_extra_trees_pipeline
from model_evaluation import make_initial_histogram_boosting_pipeline
from model_evaluation import make_k_nearest_neighbours_pipeline
from model_evaluation import make_logistic_regression_pipeline
from model_evaluation import make_random_forest_pipeline
from model_preprocessing import make_initial_preprocessor


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
BINARY_LABELS = (
    "functional",
    "non functional",
)
REPAIR_LABEL = "functional needs repair"
BINARY_TO_INTEGER = {
    "functional": 0,
    "non functional": 1,
}

SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_FOLD_WINS = 3
SELECTION_GATE_WORST_FOLD = -0.0025


def _make_binary_mlp_pipeline():
    return make_ann_pipeline(ANN_CANDIDATES["relu_128_64"])


PIPELINE_FACTORIES: dict[str, _Callable[[], object]] = {
    "decision_tree": make_initial_decision_tree_pipeline,
    "logistic_regression": make_logistic_regression_pipeline,
    "gaussian_naive_bayes": make_gaussian_naive_bayes_pipeline,
    "knn": make_k_nearest_neighbours_pipeline,
    "extra_trees": make_initial_extra_trees_pipeline,
    "histogram_boosting": make_initial_histogram_boosting_pipeline,
    "random_forest": make_random_forest_pipeline,
    "mlp_relu_128_64": _make_binary_mlp_pipeline,
}

PIPELINE_MODEL_NAMES = {
    "decision_tree": "Binary constrained decision tree",
    "logistic_regression": "Binary logistic regression",
    "gaussian_naive_bayes": "Binary Gaussian naïve Bayes",
    "knn": "Binary KNN",
    "extra_trees": "Binary Extra Trees",
    "histogram_boosting": "Binary histogram boosting",
    "random_forest": "Binary Random Forest",
    "mlp_relu_128_64": "Binary MLP ReLU 128-64",
}


@_dataclass(frozen=True)
class BinaryAcceleratedSpec:
    """One binary accelerated-tree candidate using an existing variant."""

    key: str
    model_name: str
    family: str
    variant: str
    seed: int = INNER_STOP_SEED


ACCELERATED_SPECS = {
    "catboost_d8": BinaryAcceleratedSpec(
        key="catboost_d8",
        model_name="Binary CatBoost d8",
        family="catboost",
        variant="d8",
    ),
    "xgboost_child1": BinaryAcceleratedSpec(
        key="xgboost_child1",
        model_name="Binary XGBoost depth 8 child 1",
        family="xgboost",
        variant="depth 8 child 1",
    ),
    "lightgbm_bagged63": BinaryAcceleratedSpec(
        key="lightgbm_bagged63",
        model_name="Binary LightGBM leaves 63 bagged",
        family="lightgbm",
        variant="leaves 63 bagged",
    ),
}


@_dataclass(frozen=True)
class BinaryEnsembleSpec:
    """One fixed soft-vote recipe over binary component probabilities."""

    key: str
    model_name: str
    components: tuple[str, ...]
    weights: tuple[float, ...]


BINARY_ENSEMBLE_SPECS = (
    BinaryEnsembleSpec(
        key="rf_hgb_equal",
        model_name="Binary equal Random Forest + histogram boosting",
        components=("random_forest", "histogram_boosting"),
        weights=(0.50, 0.50),
    ),
    BinaryEnsembleSpec(
        key="extra_hgb_equal",
        model_name="Binary equal Extra Trees + histogram boosting",
        components=("extra_trees", "histogram_boosting"),
        weights=(0.50, 0.50),
    ),
    BinaryEnsembleSpec(
        key="xgb_rf_55_45",
        model_name="Binary 55% XGBoost + 45% Random Forest",
        components=("xgboost_child1", "random_forest"),
        weights=(0.55, 0.45),
    ),
    BinaryEnsembleSpec(
        key="xgb_lgb_equal",
        model_name="Binary equal XGBoost + LightGBM",
        components=("xgboost_child1", "lightgbm_bagged63"),
        weights=(0.50, 0.50),
    ),
    BinaryEnsembleSpec(
        key="xgb_cat_equal",
        model_name="Binary equal XGBoost + CatBoost",
        components=("xgboost_child1", "catboost_d8"),
        weights=(0.50, 0.50),
    ),
    BinaryEnsembleSpec(
        key="rf_lgb_equal",
        model_name="Binary equal Random Forest + LightGBM",
        components=("random_forest", "lightgbm_bagged63"),
        weights=(0.50, 0.50),
    ),
    BinaryEnsembleSpec(
        key="xgb_rf_lgb",
        model_name="Binary 50% XGBoost + 35% Random Forest + 15% LightGBM",
        components=("xgboost_child1", "random_forest", "lightgbm_bagged63"),
        weights=(0.50, 0.35, 0.15),
    ),
    BinaryEnsembleSpec(
        key="xgb_rf_cat",
        model_name="Binary 50% XGBoost + 35% Random Forest + 15% CatBoost",
        components=("xgboost_child1", "random_forest", "catboost_d8"),
        weights=(0.50, 0.35, 0.15),
    ),
    BinaryEnsembleSpec(
        key="xgb_rf_mlp",
        model_name="Binary 52.25% XGBoost + 42.75% Random Forest + 5% MLP",
        components=("xgboost_child1", "random_forest", "mlp_relu_128_64"),
        weights=(0.5225, 0.4275, 0.05),
    ),
)


@_dataclass(frozen=True)
class BinaryLocalTestResult:
    """Selected development recipe evaluated on the full three-class local test."""

    selected_key: str
    selected_model_name: str
    component_weights: tuple[tuple[str, float], ...]
    metrics: _pd.Series
    predictions: _pd.Series
    probabilities: _pd.DataFrame
    confusion_counts: _pd.DataFrame


def evaluate_binary_pipeline_candidate(
    key: str,
    partitioned_data: PartitionedData,
    cross_validation: object,
    *,
    pipeline_factory: _Callable[[], object] | None = None,
    model_name: str | None = None,
) -> CandidateEvaluation:
    """Remove repair rows from training and score all outer-validation rows."""

    if pipeline_factory is None:
        if key not in PIPELINE_FACTORIES:
            raise ValueError(f"Unknown binary pipeline candidate: {key!r}.")
        pipeline_factory = PIPELINE_FACTORIES[key]
    if model_name is None:
        model_name = PIPELINE_MODEL_NAMES.get(key, key)
    probability_values = _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostics = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_outer_training = partitioned_data.X_development.iloc[training_positions]
        y_outer_training = partitioned_data.y_development.iloc[training_positions]
        keep = y_outer_training.ne(REPAIR_LABEL).to_numpy()
        X_training = X_outer_training.iloc[_np.flatnonzero(keep)]
        y_training = y_outer_training.iloc[_np.flatnonzero(keep)]
        X_validation = partitioned_data.X_development.iloc[validation_positions]

        started = _time.perf_counter()
        pipeline = pipeline_factory()
        pipeline.fit(X_training, y_training)
        fit_seconds = _time.perf_counter() - started
        predict_started = _time.perf_counter()
        probabilities = _align_binary_probabilities(
            pipeline,
            X_validation,
        )
        predict_seconds = _time.perf_counter() - predict_started
        probability_values[validation_positions] = probabilities
        classifier = (
            pipeline.named_steps.get("classifier")
            if hasattr(pipeline, "named_steps")
            else pipeline
        )
        diagnostics.append(
            {
                "validation_fold": fold_number,
                "outer_training_rows": len(training_positions),
                "binary_training_rows": len(y_training),
                "stripped_repair_rows": int((~keep).sum()),
                "selected_iterations": int(
                    getattr(classifier, "n_iter_", 0)
                    if _np.isscalar(getattr(classifier, "n_iter_", 0))
                    else _np.max(getattr(classifier, "n_iter_", [0]))
                ),
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": fit_seconds + predict_seconds,
            }
        )
        print(
            f"Completed {model_name} fold {fold_number}/{CROSS_VALIDATION_FOLDS} "
            f"in {fit_seconds + predict_seconds:.1f} seconds.",
            flush=True,
        )
    return build_candidate_evaluation(
        model_name=model_name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )


def evaluate_binary_accelerated_candidate(
    spec: BinaryAcceleratedSpec,
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> CandidateEvaluation:
    """Select binary tree counts inside each outer training fold, then refit."""

    probability_values = _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostics = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_outer_training = partitioned_data.X_development.iloc[training_positions]
        y_outer_training = partitioned_data.y_development.iloc[training_positions]
        keep = y_outer_training.ne(REPAIR_LABEL).to_numpy()
        X_training = X_outer_training.iloc[_np.flatnonzero(keep)]
        y_training = y_outer_training.iloc[_np.flatnonzero(keep)]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        started = _time.perf_counter()
        probabilities, fold_diagnostics = _fit_binary_accelerated_fold(
            spec,
            X_training,
            y_training,
            X_validation,
            fold_number=fold_number,
        )
        probability_values[validation_positions] = probabilities
        total_seconds = _time.perf_counter() - started
        diagnostics.append(
            {
                "validation_fold": fold_number,
                "outer_training_rows": len(training_positions),
                "binary_training_rows": len(y_training),
                "stripped_repair_rows": int((~keep).sum()),
                **fold_diagnostics,
                "total_seconds": total_seconds,
            }
        )
        print(
            f"Completed {spec.model_name} fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS}: "
            f"{fold_diagnostics['selected_iterations']} trees in "
            f"{total_seconds:.1f} seconds.",
            flush=True,
        )
    return build_candidate_evaluation(
        model_name=spec.model_name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )


def evaluate_binary_ensembles(
    partitioned_data: PartitionedData,
    cross_validation: object,
    components: _Mapping[str, CandidateEvaluation],
    *,
    specs: tuple[BinaryEnsembleSpec, ...] = BINARY_ENSEMBLE_SPECS,
) -> dict[str, CandidateEvaluation]:
    """Combine the predeclared binary OOF component probabilities."""

    evaluations = {}
    for spec in specs:
        missing = set(spec.components) - set(components)
        if missing:
            raise ValueError(
                f"Binary ensemble {spec.key!r} is missing components: "
                f"{sorted(missing)!r}."
            )
        if len(spec.components) != len(spec.weights):
            raise ValueError(f"Binary ensemble weights do not match: {spec.key!r}.")
        evaluations[spec.key] = evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            *(components[key] for key in spec.components),
            weights=spec.weights,
            model_name=spec.model_name,
        )
    return evaluations


def summarise_binary_evaluations(
    partitioned_data: PartitionedData,
    cross_validation: object,
    evaluations: _Mapping[str, CandidateEvaluation],
) -> _pd.DataFrame:
    """Summarise full-task and allowed-class performance for each candidate."""

    rows = []
    for key, evaluation in evaluations.items():
        conditional_fold_scores = []
        for _, validation_positions in cross_validation.split():
            y_validation = partitioned_data.y_development.iloc[validation_positions]
            allowed = y_validation.ne(REPAIR_LABEL).to_numpy()
            predictions = evaluation.out_of_fold_probabilities.iloc[
                validation_positions
            ].idxmax(axis=1)
            conditional_fold_scores.append(
                _accuracy_score(
                    y_validation.iloc[_np.flatnonzero(allowed)],
                    predictions.iloc[_np.flatnonzero(allowed)],
                )
            )
        metrics = score_binary_reduction(
            partitioned_data.y_development,
            evaluation.out_of_fold_probabilities,
        )
        rows.append(
            {
                "candidate": key,
                "model_name": evaluation.model_name,
                "mean_three_class_accuracy": float(
                    evaluation.metric_summary.loc["accuracy", "mean"]
                ),
                "three_class_accuracy_std": float(
                    evaluation.metric_summary.loc["accuracy", "std"]
                ),
                "mean_conditional_binary_accuracy": float(
                    _np.mean(conditional_fold_scores)
                ),
                "conditional_binary_accuracy_std": float(
                    _np.std(conditional_fold_scores, ddof=1)
                ),
                "functional_recall": float(
                    evaluation.metric_summary.loc[
                        "recall: functional",
                        "mean",
                    ]
                ),
                "repair_recall": float(
                    evaluation.metric_summary.loc[
                        "recall: functional needs repair",
                        "mean",
                    ]
                ),
                "non_functional_recall": float(
                    evaluation.metric_summary.loc[
                        "recall: non functional",
                        "mean",
                    ]
                ),
                "conditional_log_loss": metrics["conditional_log_loss"],
                "conditional_brier_score": metrics[
                    "conditional_brier_score"
                ],
                "predicted_functional_share": metrics[
                    "predicted_functional_share"
                ],
                "predicted_non_functional_share": metrics[
                    "predicted_non_functional_share"
                ],
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("candidate")
        .sort_values("mean_three_class_accuracy", ascending=False)
    )


def compare_to_reference(
    evaluations: _Mapping[str, CandidateEvaluation],
    *,
    reference_key: str,
) -> _pd.DataFrame:
    """Apply the established fold gate relative to one binary reference model."""

    if reference_key not in evaluations:
        raise ValueError(f"Binary reference candidate is missing: {reference_key!r}.")
    reference = evaluations[reference_key]
    reference_accuracy = reference.fold_metrics["accuracy"]
    rows = []
    for key, evaluation in evaluations.items():
        fold_change = evaluation.fold_metrics["accuracy"] - reference_accuracy
        accuracy_change = float(fold_change.mean())
        rows.append(
            {
                "candidate": key,
                "reference": reference_key,
                "accuracy_change": accuracy_change,
                "fold_wins": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "passes_gate": (
                    key != reference_key
                    and accuracy_change >= SELECTION_GATE_ACCURACY
                    and int(fold_change.gt(0).sum()) >= SELECTION_GATE_FOLD_WINS
                    and float(fold_change.min()) >= SELECTION_GATE_WORST_FOLD
                ),
            }
        )
    return _pd.DataFrame(rows).set_index("candidate")


def compare_binary_diversity(
    partitioned_data: PartitionedData,
    evaluations: _Mapping[str, CandidateEvaluation],
) -> _pd.DataFrame:
    """Return pairwise hard-label disagreement and unique-correct shares."""

    target = partitioned_data.y_development
    predictions = {
        key: evaluation.out_of_fold_probabilities.idxmax(axis=1)
        for key, evaluation in evaluations.items()
    }
    rows = []
    keys = list(predictions)
    for left_position, left_key in enumerate(keys):
        for right_key in keys[left_position + 1 :]:
            left = predictions[left_key]
            right = predictions[right_key]
            left_correct = left.eq(target)
            right_correct = right.eq(target)
            rows.append(
                {
                    "left": left_key,
                    "right": right_key,
                    "disagreement": float(left.ne(right).mean()),
                    "left_only_correct": float(
                        (left_correct & ~right_correct).mean()
                    ),
                    "right_only_correct": float(
                        (~left_correct & right_correct).mean()
                    ),
                    "both_wrong": float((~left_correct & ~right_correct).mean()),
                }
            )
    return _pd.DataFrame(rows).set_index(["left", "right"])


def score_binary_reduction(
    target: _pd.Series,
    probabilities: _pd.DataFrame | _np.ndarray,
) -> _pd.Series:
    """Score two-label probabilities against an unchanged three-class target."""

    values = _np.asarray(probabilities, dtype="float64")
    if values.shape == (len(target), len(CLASS_LABELS)):
        binary_values = values[:, [0, 2]]
    elif values.shape == (len(target), len(BINARY_LABELS)):
        binary_values = values
    else:
        raise ValueError("Binary-reduction probabilities have an unexpected shape.")
    if not _np.isfinite(binary_values).all() or (binary_values < 0).any():
        raise ValueError(
            "Binary-reduction probabilities must be finite and non-negative."
        )
    row_sums = binary_values.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Binary-reduction probability rows must have positive mass.")
    binary_values = binary_values / row_sums
    predictions = _np.asarray(BINARY_LABELS)[binary_values.argmax(axis=1)]
    allowed = target.ne(REPAIR_LABEL).to_numpy()
    allowed_target = target.iloc[_np.flatnonzero(allowed)]
    allowed_probabilities = binary_values[allowed]
    one_hot = _np.column_stack(
        [
            allowed_target.eq(label).to_numpy(dtype="float64")
            for label in BINARY_LABELS
        ]
    )
    recalls = _recall_score(
        target,
        predictions,
        labels=list(CLASS_LABELS),
        average=None,
        zero_division=0,
    )
    return _pd.Series(
        {
            "rows": len(target),
            "repair_rows": int((~allowed).sum()),
            "three_class_accuracy_ceiling": float(allowed.mean()),
            "three_class_accuracy": float(_accuracy_score(target, predictions)),
            "conditional_binary_accuracy": float(
                _accuracy_score(allowed_target, predictions[allowed])
            ),
            "functional_recall": float(recalls[0]),
            "repair_recall": float(recalls[1]),
            "non_functional_recall": float(recalls[2]),
            "conditional_log_loss": float(
                _log_loss(
                    allowed_target,
                    allowed_probabilities,
                    labels=list(BINARY_LABELS),
                )
            ),
            "conditional_brier_score": float(
                _np.mean(
                    _np.square(allowed_probabilities - one_hot).sum(axis=1)
                )
            ),
            "predicted_functional_share": float(
                _np.mean(predictions == "functional")
            ),
            "predicted_repair_share": 0.0,
            "predicted_non_functional_share": float(
                _np.mean(predictions == "non functional")
            ),
        }
    )


def fit_selected_binary_recipe_on_local_test(
    partitioned_data: PartitionedData,
    *,
    selected_key: str,
    component_evaluations: _Mapping[str, CandidateEvaluation],
    ensemble_specs: tuple[BinaryEnsembleSpec, ...] = BINARY_ENSEMBLE_SPECS,
) -> BinaryLocalTestResult:
    """Fit a frozen binary component or ensemble on development and score local test."""

    ensemble_by_key = {spec.key: spec for spec in ensemble_specs}
    if selected_key in ensemble_by_key:
        selected_spec = ensemble_by_key[selected_key]
        component_weights = tuple(
            zip(selected_spec.components, selected_spec.weights, strict=True)
        )
        selected_model_name = selected_spec.model_name
    else:
        component_weights = ((selected_key, 1.0),)
        if selected_key not in component_evaluations:
            raise ValueError(f"Selected binary component is missing: {selected_key!r}.")
        selected_model_name = component_evaluations[selected_key].model_name

    keep = partitioned_data.y_development.ne(REPAIR_LABEL).to_numpy()
    X_training = partitioned_data.X_development.iloc[_np.flatnonzero(keep)]
    y_training = partitioned_data.y_development.iloc[_np.flatnonzero(keep)]
    component_probabilities = []
    weights = []
    for component_key, weight in component_weights:
        if component_key not in component_evaluations:
            raise ValueError(
                f"Selected binary recipe is missing component: {component_key!r}."
            )
        component_probabilities.append(
            fit_binary_component_probabilities(
                component_key,
                X_training,
                y_training,
                partitioned_data.X_local_test,
                development_evaluation=component_evaluations[component_key],
            )
        )
        weights.append(weight)
    values = _np.average(
        _np.stack(component_probabilities),
        axis=0,
        weights=_np.asarray(weights),
    )
    _validate_three_class_binary_probabilities(values)
    probabilities = _pd.DataFrame(
        values,
        index=partitioned_data.y_local_test.index,
        columns=_pd.Index(CLASS_LABELS, name="predicted_class"),
    )
    predictions = probabilities.idxmax(axis=1).rename("prediction")
    metrics = score_binary_reduction(
        partitioned_data.y_local_test,
        probabilities,
    )
    confusion = _pd.DataFrame(
        _confusion_matrix(
            partitioned_data.y_local_test,
            predictions,
            labels=list(CLASS_LABELS),
        ),
        index=_pd.Index(CLASS_LABELS, name="actual"),
        columns=_pd.Index(CLASS_LABELS, name="predicted"),
    )
    return BinaryLocalTestResult(
        selected_key=selected_key,
        selected_model_name=selected_model_name,
        component_weights=component_weights,
        metrics=metrics,
        predictions=predictions,
        probabilities=probabilities,
        confusion_counts=confusion,
    )


def fit_binary_component_probabilities(
    key: str,
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_prediction: _pd.DataFrame,
    *,
    development_evaluation: CandidateEvaluation,
) -> _np.ndarray:
    """Fit one frozen binary component for a selected downstream comparison."""

    if y_training.eq(REPAIR_LABEL).any():
        raise ValueError("Binary component fitting received repair training rows.")
    if key in PIPELINE_FACTORIES:
        model = PIPELINE_FACTORIES[key]()
        model.fit(X_training, y_training)
        return _align_binary_probabilities(model, X_prediction)
    if key not in ACCELERATED_SPECS:
        raise ValueError(f"Unknown binary component: {key!r}.")
    iterations = int(
        _np.median(development_evaluation.diagnostics["selected_iterations"])
    )
    return _fit_binary_accelerated_fixed(
        ACCELERATED_SPECS[key],
        X_training,
        y_training,
        X_prediction,
        iterations=iterations,
    )


def _fit_binary_accelerated_fold(
    spec: BinaryAcceleratedSpec,
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_validation: _pd.DataFrame,
    *,
    fold_number: int,
) -> tuple[_np.ndarray, dict[str, int | float]]:
    positions = _np.arange(len(y_training))
    fit_positions, stop_positions = _train_test_split(
        positions,
        test_size=INNER_STOP_FRACTION,
        random_state=INNER_STOP_SEED + fold_number,
        stratify=y_training,
    )
    X_inner_fit = X_training.iloc[fit_positions]
    y_inner_fit = _encode_binary_target(y_training.iloc[fit_positions])
    X_inner_stop = X_training.iloc[stop_positions]
    y_inner_stop = _encode_binary_target(y_training.iloc[stop_positions])
    stopping_started = _time.perf_counter()

    if spec.family == "catboost":
        X_inner_fit, categorical = _engineer_catboost_features(X_inner_fit)
        X_inner_stop, stop_categorical = _engineer_catboost_features(X_inner_stop)
        if categorical != stop_categorical:
            raise ValueError("Binary CatBoost category columns changed inside a fold.")
        model = _make_binary_catboost_model(
            spec,
            iterations=CATBOOST_ITERATION_CAP,
            early_stopping=True,
        )
        model.fit(
            X_inner_fit,
            y_inner_fit,
            cat_features=list(categorical),
            eval_set=(X_inner_stop, y_inner_stop),
            use_best_model=True,
            verbose=False,
        )
        selected_iterations = int(model.get_best_iteration()) + 1
    elif spec.family == "xgboost":
        preprocessor = make_initial_preprocessor()
        X_inner_fit = preprocessor.fit_transform(X_inner_fit)
        X_inner_stop = preprocessor.transform(X_inner_stop)
        model = _make_binary_xgboost_model(
            spec,
            iterations=XGBOOST_ITERATION_CAP,
            early_stopping=True,
        )
        model.fit(
            X_inner_fit,
            y_inner_fit,
            eval_set=[(X_inner_stop, y_inner_stop)],
            verbose=False,
        )
        selected_iterations = int(model.best_iteration) + 1
    elif spec.family == "lightgbm":
        preprocessor = make_initial_preprocessor()
        X_inner_fit = preprocessor.fit_transform(X_inner_fit)
        X_inner_stop = preprocessor.transform(X_inner_stop)
        model = _make_binary_lightgbm_model(
            spec,
            iterations=LIGHTGBM_ITERATION_CAP,
        )
        model.fit(
            X_inner_fit,
            y_inner_fit,
            eval_X=X_inner_stop,
            eval_y=y_inner_stop,
            eval_metric="binary_logloss",
            callbacks=[
                _lgb_early_stopping(LIGHTGBM_STOPPING_ROUNDS, verbose=False),
                _lgb_log_evaluation(period=0),
            ],
        )
        selected_iterations = int(model.best_iteration_)
    else:
        raise ValueError(f"Unknown binary accelerated family: {spec.family!r}.")
    stopping_seconds = _time.perf_counter() - stopping_started
    probabilities = _fit_binary_accelerated_fixed(
        spec,
        X_training,
        y_training,
        X_validation,
        iterations=selected_iterations,
    )
    return probabilities, {
        "selected_iterations": selected_iterations,
        "inner_fit_rows": len(fit_positions),
        "inner_stop_rows": len(stop_positions),
        "stopping_seconds": stopping_seconds,
    }


def _fit_binary_accelerated_fixed(
    spec: BinaryAcceleratedSpec,
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_prediction: _pd.DataFrame,
    *,
    iterations: int,
) -> _np.ndarray:
    if iterations <= 0:
        raise ValueError("Binary accelerated iterations must be positive.")
    encoded = _encode_binary_target(y_training)
    if spec.family == "catboost":
        X_fit, categorical = _engineer_catboost_features(X_training)
        X_predict, prediction_categorical = _engineer_catboost_features(
            X_prediction
        )
        if categorical != prediction_categorical:
            raise ValueError("Binary CatBoost category columns changed at prediction.")
        model = _make_binary_catboost_model(
            spec,
            iterations=iterations,
            early_stopping=False,
        )
        model.fit(X_fit, encoded, cat_features=list(categorical), verbose=False)
    else:
        preprocessor = make_initial_preprocessor()
        X_fit = preprocessor.fit_transform(X_training)
        X_predict = preprocessor.transform(X_prediction)
        if spec.family == "xgboost":
            model = _make_binary_xgboost_model(
                spec,
                iterations=iterations,
                early_stopping=False,
            )
            model.fit(X_fit, encoded, verbose=False)
        elif spec.family == "lightgbm":
            model = _make_binary_lightgbm_model(spec, iterations=iterations)
            model.fit(
                X_fit,
                encoded,
                callbacks=[_lgb_log_evaluation(period=0)],
            )
        else:
            raise ValueError(f"Unknown binary accelerated family: {spec.family!r}.")
    binary = _np.asarray(model.predict_proba(X_predict), dtype="float64")
    classes = _np.asarray(model.classes_, dtype="int64")
    ordered = _pd.DataFrame(binary, columns=classes).loc[:, [0, 1]].to_numpy()
    return _embed_binary_probabilities(ordered)


def _make_binary_catboost_model(
    spec: BinaryAcceleratedSpec,
    *,
    iterations: int,
    early_stopping: bool,
) -> _CatBoostClassifier:
    parameters = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "custom_metric": ["Accuracy"],
        "task_type": "GPU",
        "devices": "0",
        "boosting_type": "Plain",
        "grow_policy": "SymmetricTree",
        "iterations": iterations,
        "border_count": 128,
        "one_hot_max_size": 10,
        "max_ctr_complexity": 2,
        "bootstrap_type": "Bayesian",
        "random_seed": spec.seed,
        "leaf_estimation_method": "Newton",
        "leaf_estimation_iterations": 1,
        "gpu_ram_part": 0.85,
        "gpu_cat_features_storage": "GpuRam",
        "allow_writing_files": False,
        "verbose": False,
        **CATBOOST_VARIANTS[spec.variant],
    }
    if early_stopping:
        parameters.update(
            {
                "od_type": "Iter",
                "od_wait": CATBOOST_STOPPING_ROUNDS,
            }
        )
    return _CatBoostClassifier(**parameters)


def _make_binary_xgboost_model(
    spec: BinaryAcceleratedSpec,
    *,
    iterations: int,
    early_stopping: bool,
) -> _XGBClassifier:
    parameters = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": iterations,
        "max_bin": 256,
        "random_state": spec.seed,
        "n_jobs": 6,
        "validate_parameters": True,
        **XGBOOST_VARIANTS[spec.variant],
    }
    if early_stopping:
        parameters["early_stopping_rounds"] = XGBOOST_STOPPING_ROUNDS
    return _XGBClassifier(**parameters)


def _make_binary_lightgbm_model(
    spec: BinaryAcceleratedSpec,
    *,
    iterations: int,
) -> _LGBMClassifier:
    return _LGBMClassifier(
        objective="binary",
        n_estimators=iterations,
        max_depth=-1,
        random_state=spec.seed,
        n_jobs=6,
        verbosity=-1,
        **LIGHTGBM_VARIANTS[spec.variant],
    )


def _align_binary_probabilities(model: object, X: _pd.DataFrame) -> _np.ndarray:
    classifier = (
        model.named_steps.get("classifier")
        if hasattr(model, "named_steps")
        else model
    )
    classes = getattr(classifier, "classes_", getattr(model, "classes_", None))
    if classes is None:
        raise ValueError("Binary classifier did not expose fitted classes.")
    probabilities = _pd.DataFrame(
        model.predict_proba(X),
        columns=list(classes),
    )
    missing = set(BINARY_LABELS) - set(probabilities.columns)
    if missing:
        raise ValueError(f"Binary classifier omitted classes: {sorted(missing)!r}.")
    return _embed_binary_probabilities(
        probabilities.loc[:, list(BINARY_LABELS)].to_numpy(dtype="float64")
    )


def _embed_binary_probabilities(binary_values: _np.ndarray) -> _np.ndarray:
    if binary_values.ndim != 2 or binary_values.shape[1] != len(BINARY_LABELS):
        raise ValueError("Binary probabilities have an unexpected shape.")
    if not _np.isfinite(binary_values).all() or (binary_values < 0).any():
        raise ValueError("Binary probabilities must be finite and non-negative.")
    row_sums = binary_values.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Binary probability rows must have positive mass.")
    binary_values = binary_values / row_sums
    values = _np.zeros((len(binary_values), len(CLASS_LABELS)), dtype="float64")
    values[:, 0] = binary_values[:, 0]
    values[:, 2] = binary_values[:, 1]
    _validate_three_class_binary_probabilities(values)
    return values


def _validate_three_class_binary_probabilities(values: _np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != len(CLASS_LABELS):
        raise ValueError("Embedded binary probabilities have an unexpected shape.")
    if not _np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            "Embedded binary probabilities must be finite and non-negative."
        )
    if not _np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("Embedded binary probability rows must sum to one.")
    if not _np.allclose(values[:, 1], 0.0):
        raise ValueError("Embedded binary probabilities must assign zero repair mass.")


def _encode_binary_target(target: _pd.Series) -> _np.ndarray:
    encoded = target.map(BINARY_TO_INTEGER)
    if encoded.isna().any():
        unexpected = sorted(set(target) - set(BINARY_LABELS))
        raise ValueError(f"Unexpected binary target classes: {unexpected!r}.")
    return encoded.to_numpy(dtype="int64")


def _engineer_catboost_features(
    X: _pd.DataFrame,
) -> tuple[_pd.DataFrame, tuple[str, ...]]:
    engineered = engineer_initial_features(X).copy()
    for column in CATEGORICAL_FEATURES:
        engineered[column] = engineered[column].astype(str)
    return engineered, tuple(CATEGORICAL_FEATURES)

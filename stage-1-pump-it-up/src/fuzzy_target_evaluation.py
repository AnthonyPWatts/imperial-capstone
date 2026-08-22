"""Evaluate ordinally adjacent fuzzy target memberships on frozen folds."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import time as _time
from typing import Callable as _Callable
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import log_loss as _log_loss
from sklearn.metrics import recall_score as _recall_score
from sklearn.model_selection import train_test_split as _train_test_split
from xgboost import XGBClassifier as _XGBClassifier

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from gpu_model_evaluation import INNER_STOP_FRACTION
from gpu_model_evaluation import INNER_STOP_SEED
from gpu_model_evaluation import XGBOOST_ITERATION_CAP
from gpu_model_evaluation import XGBOOST_STOPPING_ROUNDS
from gpu_model_evaluation import XGBOOST_VARIANTS
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_evaluation import evaluate_weighted_soft_vote
from model_evaluation import make_random_forest_pipeline
from model_preprocessing import make_initial_preprocessor


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
CLASS_TO_INTEGER = {
    label: position for position, label in enumerate(CLASS_LABELS)
}

ACCEPTED_XGBOOST_WEIGHT = 0.55
ACCEPTED_RANDOM_FOREST_WEIGHT = 0.45
SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_FOLD_WINS = 3
SELECTION_GATE_WORST_FOLD = -0.0025
SELECTION_GATE_REPAIR_RECALL = -0.02


@_dataclass(frozen=True)
class FuzzyMembershipPolicy:
    """Triangular class memberships over the proposed condition ordering."""

    key: str
    adjacent_overlap: float

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("Fuzzy membership policy key must not be empty.")
        if (
            not _np.isfinite(self.adjacent_overlap)
            or self.adjacent_overlap <= 0
            or self.adjacent_overlap >= 1
        ):
            raise ValueError("Adjacent overlap must fall strictly between 0 and 1.")


FUZZY_POLICIES = (
    FuzzyMembershipPolicy("triangular_0_010", 0.010),
    FuzzyMembershipPolicy("triangular_0_025", 0.025),
    FuzzyMembershipPolicy("triangular_0_050", 0.050),
    FuzzyMembershipPolicy("triangular_0_100", 0.100),
)


@_dataclass(frozen=True)
class ExpandedFuzzyTarget:
    """Weighted hard-label expansion equivalent to one soft target matrix."""

    source_positions: _np.ndarray
    encoded_labels: _np.ndarray
    sample_weights: _np.ndarray
    memberships: _np.ndarray


@_dataclass(frozen=True)
class FuzzyPolicyEvaluation:
    """Component and blend evaluations for one membership policy."""

    policy: FuzzyMembershipPolicy
    membership_summary: _pd.DataFrame
    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    blend: CandidateEvaluation


@_dataclass(frozen=True)
class FuzzyLocalTestResult:
    """One selected fuzzy policy refitted and scored on crisp local labels."""

    policy: FuzzyMembershipPolicy
    xgboost_iterations: int
    metrics: _pd.Series
    predictions: _pd.Series
    probabilities: _pd.DataFrame
    confusion_counts: _pd.DataFrame


def triangular_memberships(
    target: _pd.Series,
    policy: FuzzyMembershipPolicy,
) -> _np.ndarray:
    """Return row-normalised memberships for each class and its neighbours.

    The observed class receives raw membership one. Each immediately adjacent
    class receives ``policy.adjacent_overlap`` and the row is normalised to one.
    Functional and non-functional are not direct neighbours.
    """

    encoded = target.map(CLASS_TO_INTEGER)
    if encoded.isna().any():
        unexpected = sorted(set(target) - set(CLASS_LABELS))
        raise ValueError(f"Unexpected target classes: {unexpected!r}.")
    actual = encoded.to_numpy(dtype="int64")
    memberships = _np.zeros((len(target), len(CLASS_LABELS)), dtype="float64")
    memberships[_np.arange(len(target)), actual] = 1.0
    overlap = policy.adjacent_overlap
    has_lower = actual > 0
    has_upper = actual < len(CLASS_LABELS) - 1
    memberships[_np.flatnonzero(has_lower), actual[has_lower] - 1] = overlap
    memberships[_np.flatnonzero(has_upper), actual[has_upper] + 1] = overlap
    memberships /= memberships.sum(axis=1, keepdims=True)
    _validate_memberships(memberships)
    return memberships


def expand_fuzzy_target(
    target: _pd.Series,
    policy: FuzzyMembershipPolicy,
) -> ExpandedFuzzyTarget:
    """Express fuzzy memberships as weighted labelled training observations."""

    memberships = triangular_memberships(target, policy)
    source_positions, encoded_labels = _np.nonzero(memberships > 0)
    weights = memberships[source_positions, encoded_labels]
    per_source_weight = _np.bincount(
        source_positions,
        weights=weights,
        minlength=len(target),
    )
    if not _np.allclose(per_source_weight, 1.0):
        raise ValueError("Expanded fuzzy weights do not preserve source-row mass.")
    return ExpandedFuzzyTarget(
        source_positions=source_positions.astype("int64", copy=False),
        encoded_labels=encoded_labels.astype("int64", copy=False),
        sample_weights=weights,
        memberships=memberships,
    )


def summarise_memberships(
    target: _pd.Series,
    policy: FuzzyMembershipPolicy,
) -> _pd.DataFrame:
    """Compare crisp class counts with effective fuzzy membership mass."""

    memberships = triangular_memberships(target, policy)
    crisp_counts = target.value_counts().reindex(CLASS_LABELS, fill_value=0)
    effective = memberships.sum(axis=0)
    return _pd.DataFrame(
        {
            "crisp_rows": crisp_counts.to_numpy(dtype="int64"),
            "crisp_share": crisp_counts.to_numpy(dtype="float64") / len(target),
            "effective_membership": effective,
            "effective_share": effective / effective.sum(),
        },
        index=_pd.Index(CLASS_LABELS, name="class_label"),
    )


def evaluate_fuzzy_policy(
    policy: FuzzyMembershipPolicy,
    partitioned_data: PartitionedData,
    cross_validation: object,
    *,
    preprocessor_factory: _Callable[[], object] = make_initial_preprocessor,
    xgboost_factory: _Callable[[int, bool], object] | None = None,
    random_forest_factory: _Callable[[], object] | None = None,
) -> FuzzyPolicyEvaluation:
    """Fit fuzzy XGBoost and Random Forest models within every outer fold."""

    if xgboost_factory is None:
        xgboost_factory = _make_fuzzy_xgboost
    if random_forest_factory is None:
        random_forest_factory = _make_fuzzy_random_forest
    row_count = len(partitioned_data.y_development)
    xgboost_values = _np.full((row_count, len(CLASS_LABELS)), _np.nan)
    forest_values = _np.full((row_count, len(CLASS_LABELS)), _np.nan)
    xgboost_diagnostics = []
    forest_diagnostics = []

    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        started = _time.perf_counter()
        xgboost_probabilities, forest_probabilities, diagnostics = (
            _fit_fuzzy_outer_fold(
                policy,
                X_training,
                y_training,
                X_validation,
                fold_number=fold_number,
                preprocessor_factory=preprocessor_factory,
                xgboost_factory=xgboost_factory,
                random_forest_factory=random_forest_factory,
            )
        )
        total_seconds = _time.perf_counter() - started
        xgboost_values[validation_positions] = xgboost_probabilities
        forest_values[validation_positions] = forest_probabilities
        common = {
            "validation_fold": fold_number,
            "source_training_rows": len(training_positions),
            "expanded_training_rows": diagnostics["expanded_training_rows"],
            "effective_training_weight": diagnostics[
                "effective_training_weight"
            ],
            "total_seconds": total_seconds,
        }
        xgboost_diagnostics.append(
            {
                **common,
                "selected_iterations": diagnostics["selected_iterations"],
                "inner_fit_rows": diagnostics["inner_fit_rows"],
                "inner_stop_rows": diagnostics["inner_stop_rows"],
                "stopping_seconds": diagnostics["stopping_seconds"],
                "refit_predict_seconds": diagnostics[
                    "xgboost_refit_predict_seconds"
                ],
            }
        )
        forest_diagnostics.append(
            {
                **common,
                "selected_iterations": diagnostics["forest_estimators"],
                "inner_fit_rows": len(training_positions),
                "inner_stop_rows": 0,
                "stopping_seconds": 0.0,
                "refit_predict_seconds": diagnostics[
                    "forest_fit_predict_seconds"
                ],
            }
        )
        print(
            f"Completed {policy.key} fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS}: "
            f"{diagnostics['selected_iterations']} XGBoost trees in "
            f"{total_seconds:.1f} seconds.",
            flush=True,
        )

    xgboost = build_candidate_evaluation(
        model_name=f"Fuzzy XGBoost [{policy.key}]",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=xgboost_values,
        diagnostic_rows=xgboost_diagnostics,
    )
    random_forest = build_candidate_evaluation(
        model_name=f"Fuzzy Random Forest [{policy.key}]",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=forest_values,
        diagnostic_rows=forest_diagnostics,
    )
    blend = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        xgboost,
        random_forest,
        weights=(ACCEPTED_XGBOOST_WEIGHT, ACCEPTED_RANDOM_FOREST_WEIGHT),
        model_name=f"Fuzzy 55% XGBoost + 45% Random Forest [{policy.key}]",
    )
    return FuzzyPolicyEvaluation(
        policy=policy,
        membership_summary=summarise_memberships(
            partitioned_data.y_development,
            policy,
        ),
        xgboost=xgboost,
        random_forest=random_forest,
        blend=blend,
    )


def summarise_fuzzy_trials(
    partitioned_data: PartitionedData,
    trials: _Mapping[str, FuzzyPolicyEvaluation],
) -> _pd.DataFrame:
    """Return strength, class and probability metrics for all fuzzy models."""

    rows = []
    target = partitioned_data.y_development
    for policy_key, trial in trials.items():
        effective_repair_share = float(
            trial.membership_summary.loc[
                "functional needs repair",
                "effective_share",
            ]
        )
        for component, evaluation in (
            ("xgboost", trial.xgboost),
            ("random_forest", trial.random_forest),
            ("blend", trial.blend),
        ):
            probability_metrics = _score_probabilities(
                target,
                evaluation.out_of_fold_probabilities,
            )
            rows.append(
                {
                    "policy": policy_key,
                    "component": component,
                    "adjacent_overlap": trial.policy.adjacent_overlap,
                    "effective_repair_share": effective_repair_share,
                    "model_name": evaluation.model_name,
                    "mean_accuracy": float(
                        evaluation.metric_summary.loc["accuracy", "mean"]
                    ),
                    "accuracy_std": float(
                        evaluation.metric_summary.loc["accuracy", "std"]
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
                    **probability_metrics,
                }
            )
    return (
        _pd.DataFrame(rows)
        .set_index(["policy", "component"])
        .sort_values("mean_accuracy", ascending=False)
    )


def evaluate_fuzzy_component_crosses(
    partitioned_data: PartitionedData,
    cross_validation: object,
    *,
    hard_xgboost: CandidateEvaluation,
    hard_random_forest: CandidateEvaluation,
    trials: _Mapping[str, FuzzyPolicyEvaluation],
) -> dict[str, CandidateEvaluation]:
    """Cross one fuzzy component at a time with its accepted hard counterpart."""

    evaluations = {}
    for policy_key, trial in trials.items():
        fuzzy_xgboost_key = f"{policy_key}__fuzzy_xgboost_hard_forest"
        evaluations[fuzzy_xgboost_key] = evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            trial.xgboost,
            hard_random_forest,
            weights=(ACCEPTED_XGBOOST_WEIGHT, ACCEPTED_RANDOM_FOREST_WEIGHT),
            model_name=(
                f"55% fuzzy XGBoost [{policy_key}] + "
                "45% hard Random Forest"
            ),
        )
        fuzzy_forest_key = f"{policy_key}__hard_xgboost_fuzzy_forest"
        evaluations[fuzzy_forest_key] = evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            hard_xgboost,
            trial.random_forest,
            weights=(ACCEPTED_XGBOOST_WEIGHT, ACCEPTED_RANDOM_FOREST_WEIGHT),
            model_name=(
                "55% hard XGBoost + "
                f"45% fuzzy Random Forest [{policy_key}]"
            ),
        )
    return evaluations


def summarise_candidate_evaluations(
    partitioned_data: PartitionedData,
    evaluations: _Mapping[str, CandidateEvaluation],
) -> _pd.DataFrame:
    """Summarise crisp-label performance for arbitrary aligned candidates."""

    rows = []
    target = partitioned_data.y_development
    for candidate_key, evaluation in evaluations.items():
        rows.append(
            {
                "candidate": candidate_key,
                "model_name": evaluation.model_name,
                "mean_accuracy": float(
                    evaluation.metric_summary.loc["accuracy", "mean"]
                ),
                "accuracy_std": float(
                    evaluation.metric_summary.loc["accuracy", "std"]
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
                **_score_probabilities(
                    target,
                    evaluation.out_of_fold_probabilities,
                ),
            }
        )
    return _pd.DataFrame(rows).set_index("candidate").sort_values(
        "mean_accuracy",
        ascending=False,
    )


def compare_fuzzy_blends_to_baseline(
    baseline: CandidateEvaluation,
    trials: _Mapping[str, FuzzyPolicyEvaluation],
) -> _pd.DataFrame:
    """Apply the accepted accuracy and repair-recall gate to fuzzy blends."""

    return compare_candidate_evaluations_to_baseline(
        baseline,
        {key: trial.blend for key, trial in trials.items()},
    ).rename_axis("policy")


def compare_candidate_evaluations_to_baseline(
    baseline: CandidateEvaluation,
    evaluations: _Mapping[str, CandidateEvaluation],
) -> _pd.DataFrame:
    """Apply the accepted promotion gate to arbitrary aligned candidates."""

    baseline_accuracy = baseline.fold_metrics["accuracy"]
    baseline_repair = baseline.fold_metrics[
        "recall: functional needs repair"
    ]
    rows = []
    for candidate_key, evaluation in evaluations.items():
        accuracy_change = evaluation.fold_metrics["accuracy"] - baseline_accuracy
        repair_change = (
            evaluation.fold_metrics["recall: functional needs repair"]
            - baseline_repair
        )
        row = {
            "candidate": candidate_key,
            "accuracy_change": float(accuracy_change.mean()),
            "fold_wins": int(accuracy_change.gt(0).sum()),
            "worst_fold_change": float(accuracy_change.min()),
            "repair_recall_change": float(repair_change.mean()),
        }
        row["passes_gate"] = (
            row["accuracy_change"] >= SELECTION_GATE_ACCURACY
            and row["fold_wins"] >= SELECTION_GATE_FOLD_WINS
            and row["worst_fold_change"] >= SELECTION_GATE_WORST_FOLD
            and row["repair_recall_change"] >= SELECTION_GATE_REPAIR_RECALL
        )
        rows.append(row)
    return _pd.DataFrame(rows).set_index("candidate").sort_values(
        "accuracy_change",
        ascending=False,
    )


def fit_fuzzy_policy_on_local_test(
    policy_trial: FuzzyPolicyEvaluation,
    partitioned_data: PartitionedData,
    *,
    preprocessor_factory: _Callable[[], object] = make_initial_preprocessor,
    xgboost_factory: _Callable[[int, bool], object] | None = None,
    random_forest_factory: _Callable[[], object] | None = None,
) -> FuzzyLocalTestResult:
    """Refit one frozen fuzzy policy and score unchanged crisp local labels."""

    if xgboost_factory is None:
        xgboost_factory = _make_fuzzy_xgboost
    if random_forest_factory is None:
        random_forest_factory = _make_fuzzy_random_forest
    iterations = int(
        _np.median(policy_trial.xgboost.diagnostics["selected_iterations"])
    )
    preprocessor = preprocessor_factory()
    X_training = preprocessor.fit_transform(partitioned_data.X_development)
    X_local_test = preprocessor.transform(partitioned_data.X_local_test)
    expanded = expand_fuzzy_target(
        partitioned_data.y_development,
        policy_trial.policy,
    )
    X_expanded = _take_rows(X_training, expanded.source_positions)

    xgboost = xgboost_factory(iterations, False)
    xgboost.fit(
        X_expanded,
        expanded.encoded_labels,
        sample_weight=expanded.sample_weights,
        verbose=False,
    )
    random_forest = random_forest_factory()
    random_forest.fit(
        X_expanded,
        expanded.encoded_labels,
        sample_weight=expanded.sample_weights,
    )
    values = (
        ACCEPTED_XGBOOST_WEIGHT
        * _ordered_probabilities(xgboost, X_local_test)
        + ACCEPTED_RANDOM_FOREST_WEIGHT
        * _ordered_probabilities(random_forest, X_local_test)
    )
    values /= values.sum(axis=1, keepdims=True)
    probabilities = _pd.DataFrame(
        values,
        index=partitioned_data.y_local_test.index,
        columns=_pd.Index(CLASS_LABELS, name="predicted_class"),
    )
    predictions = probabilities.idxmax(axis=1).rename("prediction")
    confusion = _pd.DataFrame(
        _confusion_matrix(
            partitioned_data.y_local_test,
            predictions,
            labels=list(CLASS_LABELS),
        ),
        index=_pd.Index(CLASS_LABELS, name="actual"),
        columns=_pd.Index(CLASS_LABELS, name="predicted"),
    )
    return FuzzyLocalTestResult(
        policy=policy_trial.policy,
        xgboost_iterations=iterations,
        metrics=_pd.Series(
            {
                "accuracy": _accuracy_score(
                    partitioned_data.y_local_test,
                    predictions,
                ),
                **_recall_metrics(
                    partitioned_data.y_local_test,
                    predictions,
                ),
                **_score_probabilities(
                    partitioned_data.y_local_test,
                    probabilities,
                ),
            }
        ),
        predictions=predictions,
        probabilities=probabilities,
        confusion_counts=confusion,
    )


def _fit_fuzzy_outer_fold(
    policy: FuzzyMembershipPolicy,
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_validation: _pd.DataFrame,
    *,
    fold_number: int,
    preprocessor_factory: _Callable[[], object],
    xgboost_factory: _Callable[[int, bool], object],
    random_forest_factory: _Callable[[], object],
) -> tuple[_np.ndarray, _np.ndarray, dict[str, int | float]]:
    positions = _np.arange(len(y_training))
    inner_fit_positions, inner_stop_positions = _train_test_split(
        positions,
        test_size=INNER_STOP_FRACTION,
        random_state=INNER_STOP_SEED + fold_number,
        stratify=y_training,
    )
    X_inner_fit = X_training.iloc[inner_fit_positions]
    y_inner_fit = y_training.iloc[inner_fit_positions]
    X_inner_stop = X_training.iloc[inner_stop_positions]
    y_inner_stop = _encode_target(y_training.iloc[inner_stop_positions])

    stopping_started = _time.perf_counter()
    stopping_preprocessor = preprocessor_factory()
    X_inner_fit = stopping_preprocessor.fit_transform(X_inner_fit)
    X_inner_stop = stopping_preprocessor.transform(X_inner_stop)
    inner_expanded = expand_fuzzy_target(y_inner_fit, policy)
    stopping_model = xgboost_factory(XGBOOST_ITERATION_CAP, True)
    stopping_model.fit(
        _take_rows(X_inner_fit, inner_expanded.source_positions),
        inner_expanded.encoded_labels,
        sample_weight=inner_expanded.sample_weights,
        eval_set=[(X_inner_stop, y_inner_stop)],
        verbose=False,
    )
    selected_iterations = int(stopping_model.best_iteration) + 1
    stopping_seconds = _time.perf_counter() - stopping_started
    if selected_iterations <= 0:
        raise ValueError("Fuzzy early stopping selected no XGBoost trees.")

    outer_preprocessor = preprocessor_factory()
    X_outer_training = outer_preprocessor.fit_transform(X_training)
    X_outer_validation = outer_preprocessor.transform(X_validation)
    outer_expanded = expand_fuzzy_target(y_training, policy)
    X_outer_expanded = _take_rows(
        X_outer_training,
        outer_expanded.source_positions,
    )

    xgboost_started = _time.perf_counter()
    xgboost = xgboost_factory(selected_iterations, False)
    xgboost.fit(
        X_outer_expanded,
        outer_expanded.encoded_labels,
        sample_weight=outer_expanded.sample_weights,
        verbose=False,
    )
    xgboost_probabilities = _ordered_probabilities(
        xgboost,
        X_outer_validation,
    )
    xgboost_seconds = _time.perf_counter() - xgboost_started

    forest_started = _time.perf_counter()
    random_forest = random_forest_factory()
    random_forest.fit(
        X_outer_expanded,
        outer_expanded.encoded_labels,
        sample_weight=outer_expanded.sample_weights,
    )
    forest_probabilities = _ordered_probabilities(
        random_forest,
        X_outer_validation,
    )
    forest_seconds = _time.perf_counter() - forest_started
    return xgboost_probabilities, forest_probabilities, {
        "selected_iterations": selected_iterations,
        "forest_estimators": int(random_forest.n_estimators),
        "inner_fit_rows": len(inner_fit_positions),
        "inner_stop_rows": len(inner_stop_positions),
        "expanded_training_rows": len(outer_expanded.encoded_labels),
        "effective_training_weight": float(
            outer_expanded.sample_weights.sum()
        ),
        "stopping_seconds": stopping_seconds,
        "xgboost_refit_predict_seconds": xgboost_seconds,
        "forest_fit_predict_seconds": forest_seconds,
    }


def _make_fuzzy_xgboost(iterations: int, early_stopping: bool) -> _XGBClassifier:
    parameters = {
        "objective": "multi:softprob",
        "num_class": len(CLASS_LABELS),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "n_estimators": iterations,
        "max_bin": 256,
        "random_state": INNER_STOP_SEED,
        "n_jobs": 6,
        "validate_parameters": True,
        **XGBOOST_VARIANTS["depth 8 child 1"],
    }
    if early_stopping:
        parameters["early_stopping_rounds"] = XGBOOST_STOPPING_ROUNDS
    return _XGBClassifier(**parameters)


def _make_fuzzy_random_forest() -> object:
    return make_random_forest_pipeline().named_steps["classifier"]


def _take_rows(values: object, positions: _np.ndarray) -> object:
    if hasattr(values, "iloc"):
        return values.iloc[positions]
    return values[positions]


def _ordered_probabilities(model: object, X: object) -> _np.ndarray:
    probabilities = _pd.DataFrame(
        model.predict_proba(X),
        columns=_np.asarray(model.classes_, dtype="int64"),
    )
    missing = set(range(len(CLASS_LABELS))) - set(probabilities.columns)
    if missing:
        raise ValueError(f"Fuzzy classifier omitted classes: {sorted(missing)!r}.")
    values = probabilities.loc[:, range(len(CLASS_LABELS))].to_numpy(
        dtype="float64"
    )
    if (
        not _np.isfinite(values).all()
        or (values < 0).any()
        or not _np.allclose(values.sum(axis=1), 1.0)
    ):
        raise ValueError("Fuzzy classifier produced invalid probabilities.")
    return values


def _encode_target(target: _pd.Series) -> _np.ndarray:
    encoded = target.map(CLASS_TO_INTEGER)
    if encoded.isna().any():
        unexpected = sorted(set(target) - set(CLASS_LABELS))
        raise ValueError(f"Unexpected target classes: {unexpected!r}.")
    return encoded.to_numpy(dtype="int64")


def _score_probabilities(
    target: _pd.Series,
    probabilities: _pd.DataFrame | _np.ndarray,
) -> dict[str, float]:
    values = _np.asarray(probabilities, dtype="float64")
    if values.shape != (len(target), len(CLASS_LABELS)):
        raise ValueError("Fuzzy probability matrix has an unexpected shape.")
    if (
        not _np.isfinite(values).all()
        or (values < 0).any()
        or not _np.allclose(values.sum(axis=1), 1.0)
    ):
        raise ValueError("Fuzzy probability matrix is invalid.")
    values = values / values.sum(axis=1, keepdims=True)
    one_hot = _np.column_stack(
        [
            target.eq(label).to_numpy(dtype="float64")
            for label in CLASS_LABELS
        ]
    )
    return {
        "log_loss": float(_log_loss(target, values, labels=list(CLASS_LABELS))),
        "brier_score": float(_np.mean(_np.square(values - one_hot).sum(axis=1))),
        **{
            f"predicted_{label.replace(' ', '_')}_share": float(value)
            for label, value in zip(
                CLASS_LABELS,
                _np.bincount(values.argmax(axis=1), minlength=3) / len(target),
                strict=True,
            )
        },
    }


def _recall_metrics(
    target: _pd.Series,
    predictions: _pd.Series,
) -> dict[str, float]:
    recalls = _recall_score(
        target,
        predictions,
        labels=list(CLASS_LABELS),
        average=None,
        zero_division=0,
    )
    return {
        f"recall_{label.replace(' ', '_')}": float(value)
        for label, value in zip(CLASS_LABELS, recalls, strict=True)
    }


def _validate_memberships(memberships: _np.ndarray) -> None:
    if memberships.ndim != 2 or memberships.shape[1] != len(CLASS_LABELS):
        raise ValueError("Fuzzy memberships have an unexpected shape.")
    if (
        not _np.isfinite(memberships).all()
        or (memberships < 0).any()
        or (memberships > 1).any()
        or not _np.allclose(memberships.sum(axis=1), 1.0)
    ):
        raise ValueError("Fuzzy memberships are not valid row distributions.")

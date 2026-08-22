"""Confirm the frozen complete-identity CatBoost recipe on the local test."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import time as _time
from typing import Callable as _Callable

import numpy as _np
import pandas as _pd
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import balanced_accuracy_score as _balanced_accuracy_score
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import f1_score as _f1_score
from sklearn.metrics import log_loss as _log_loss
from sklearn.metrics import recall_score as _recall_score
from scipy.stats import binomtest as _binomtest

from catboost_identity_evaluation import (
    engineer_complete_identity_catboost_features,
)
from data_partitioning import PartitionedData
from final_model import CLASS_LABELS
from final_model import CompetitionPrediction
from final_model import build_competition_prediction
from final_model import ordered_probabilities
from final_model import validate_probabilities
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_catboost_spec
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import median_selected_iterations
from model_evaluation import CandidateEvaluation
from model_evaluation import make_random_forest_pipeline
from modelling_data import ModellingData


ACCEPTED_WEIGHTS = (0.55, 0.45)
IDENTITY_CANDIDATE_WEIGHTS = (0.44, 0.36, 0.20)


@_dataclass(frozen=True)
class IdentityLocalTestConfirmation:
    """Frozen-recipe evidence from one development-to-local-test refit."""

    metrics: _pd.DataFrame
    confusion_counts: dict[str, _pd.DataFrame]
    component_seconds: _pd.Series
    selected_iterations: _pd.Series
    probabilities: dict[str, _np.ndarray]


def fit_frozen_identity_recipes_on_local_test(
    partitioned_data: PartitionedData,
    accepted_xgboost: CandidateEvaluation,
    identity_catboost: CandidateEvaluation,
    *,
    gpu_fitter: _Callable = fit_gpu_candidate_probabilities,
    random_forest_factory: _Callable[[], object] = make_random_forest_pipeline,
) -> IdentityLocalTestConfirmation:
    """Fit the preselected components on development and score local test."""

    for evaluation in (accepted_xgboost, identity_catboost):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Local-test candidate uses different frozen folds.")

    xgboost_iterations = median_selected_iterations(accepted_xgboost)
    catboost_iterations = median_selected_iterations(identity_catboost)
    xgboost_probabilities, xgboost_seconds = gpu_fitter(
        make_xgboost_spec(variant="depth 8 child 1"),
        partitioned_data.X_development,
        partitioned_data.y_development,
        partitioned_data.X_local_test,
        iterations=xgboost_iterations,
    )

    forest = random_forest_factory()
    forest_started = _time.perf_counter()
    forest.fit(partitioned_data.X_development, partitioned_data.y_development)
    forest_probabilities = ordered_probabilities(
        forest,
        partitioned_data.X_local_test,
    )
    forest_seconds = _time.perf_counter() - forest_started

    catboost_probabilities, catboost_seconds = gpu_fitter(
        make_catboost_spec(variant="d8"),
        partitioned_data.X_development,
        partitioned_data.y_development,
        partitioned_data.X_local_test,
        iterations=catboost_iterations,
        catboost_feature_engineer=(
            engineer_complete_identity_catboost_features
        ),
    )
    components = {
        "xgboost": xgboost_probabilities,
        "random_forest": forest_probabilities,
        "identity_catboost": catboost_probabilities,
    }
    recipes = blend_frozen_identity_recipes(components)
    metrics, confusion_counts = score_identity_recipes(
        partitioned_data.y_local_test,
        recipes,
    )
    return IdentityLocalTestConfirmation(
        metrics=metrics,
        confusion_counts=confusion_counts,
        component_seconds=_pd.Series(
            {
                "XGBoost": xgboost_seconds,
                "Random Forest": forest_seconds,
                "complete-identity CatBoost": catboost_seconds,
            },
            name="fit and predict seconds",
        ),
        selected_iterations=_pd.Series(
            {
                "XGBoost": xgboost_iterations,
                "complete-identity CatBoost": catboost_iterations,
            },
            name="development-refit tree count",
            dtype="int64",
        ),
        probabilities={**components, **recipes},
    )


def blend_frozen_identity_recipes(
    components: dict[str, _np.ndarray],
) -> dict[str, _np.ndarray]:
    """Build the accepted and preselected candidate probability recipes."""

    required = {"xgboost", "random_forest", "identity_catboost"}
    missing = required.difference(components)
    if missing:
        raise KeyError(f"Missing frozen recipe components: {sorted(missing)!r}.")
    expected_rows = len(components["xgboost"])
    for values in components.values():
        validate_probabilities(values, expected_rows)
    accepted = (
        ACCEPTED_WEIGHTS[0] * components["xgboost"]
        + ACCEPTED_WEIGHTS[1] * components["random_forest"]
    )
    candidate = (
        IDENTITY_CANDIDATE_WEIGHTS[0] * components["xgboost"]
        + IDENTITY_CANDIDATE_WEIGHTS[1] * components["random_forest"]
        + IDENTITY_CANDIDATE_WEIGHTS[2] * components["identity_catboost"]
    )
    for values in (accepted, candidate):
        validate_probabilities(values, expected_rows)
    return {
        "accepted_55_45": accepted,
        "identity_candidate_44_36_20": candidate,
    }


def build_frozen_identity_competition_predictions(
    modelling_data: ModellingData,
    submission_template: _pd.DataFrame,
    components: dict[str, _np.ndarray],
    component_seconds: _pd.Series,
) -> dict[str, CompetitionPrediction]:
    """Build validated accepted and identity-candidate competition records."""

    recipes = blend_frozen_identity_recipes(components)
    required_seconds = {
        "XGBoost",
        "Random Forest",
        "complete-identity CatBoost",
    }
    missing_seconds = required_seconds.difference(component_seconds.index)
    if missing_seconds:
        raise KeyError(
            f"Missing component timings: {sorted(missing_seconds)!r}."
        )
    accepted_seconds = component_seconds.loc[["XGBoost", "Random Forest"]]
    return {
        "accepted_55_45": build_competition_prediction(
            modelling_data,
            submission_template,
            recipes["accepted_55_45"],
            accepted_seconds,
        ),
        "identity_candidate_44_36_20": build_competition_prediction(
            modelling_data,
            submission_template,
            recipes["identity_candidate_44_36_20"],
            component_seconds,
        ),
    }


def score_identity_recipes(
    y_true: _pd.Series,
    recipes: dict[str, _np.ndarray],
) -> tuple[_pd.DataFrame, dict[str, _pd.DataFrame]]:
    """Score both frozen recipes with crisp and probability metrics."""

    metric_columns = {}
    confusion_counts = {}
    for name, probabilities in recipes.items():
        validate_probabilities(probabilities, len(y_true))
        scoring_probabilities = probabilities / probabilities.sum(
            axis=1,
            keepdims=True,
        )
        predictions = _np.asarray(CLASS_LABELS)[
            scoring_probabilities.argmax(axis=1)
        ]
        recalls = _recall_score(
            y_true,
            predictions,
            labels=list(CLASS_LABELS),
            average=None,
            zero_division=0,
        )
        one_hot = _np.column_stack(
            [y_true.eq(label).to_numpy(dtype=float) for label in CLASS_LABELS]
        )
        metric_columns[name] = _pd.Series(
            {
                "accuracy": float(_accuracy_score(y_true, predictions)),
                "balanced_accuracy": float(
                    _balanced_accuracy_score(y_true, predictions)
                ),
                "macro_f1": float(
                    _f1_score(
                        y_true,
                        predictions,
                        labels=list(CLASS_LABELS),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "log_loss": float(
                    _log_loss(
                        y_true,
                        scoring_probabilities,
                        labels=list(CLASS_LABELS),
                    )
                ),
                "multiclass_brier": float(
                    _np.mean(
                        _np.sum(
                            (scoring_probabilities - one_hot) ** 2,
                            axis=1,
                        )
                    )
                ),
                **{
                    f"recall: {label}": float(value)
                    for label, value in zip(CLASS_LABELS, recalls, strict=True)
                },
            }
        )
        counts = _confusion_matrix(
            y_true,
            predictions,
            labels=list(CLASS_LABELS),
        )
        confusion_counts[name] = _pd.DataFrame(
            counts,
            index=_pd.Index(CLASS_LABELS, name="actual"),
            columns=_pd.Index(CLASS_LABELS, name="predicted"),
        )
    metrics = _pd.DataFrame(metric_columns)
    metrics["candidate_change"] = (
        metrics["identity_candidate_44_36_20"]
        - metrics["accepted_55_45"]
    )
    return metrics, confusion_counts


def summarise_paired_predictions(
    y_true: _pd.Series,
    recipes: dict[str, _np.ndarray],
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260822,
) -> _pd.Series:
    """Summarise paired hard-label changes without altering the recipes."""

    if bootstrap_samples < 1:
        raise ValueError("At least one paired bootstrap sample is required.")
    predictions = {}
    for name in ("accepted_55_45", "identity_candidate_44_36_20"):
        probabilities = recipes[name]
        validate_probabilities(probabilities, len(y_true))
        predictions[name] = _np.asarray(CLASS_LABELS)[
            probabilities.argmax(axis=1)
        ]
    actual = y_true.to_numpy()
    accepted = predictions["accepted_55_45"]
    candidate = predictions["identity_candidate_44_36_20"]
    accepted_correct = accepted == actual
    candidate_correct = candidate == actual
    accepted_only = int((accepted_correct & ~candidate_correct).sum())
    candidate_only = int((candidate_correct & ~accepted_correct).sum())
    discordant_correctness = accepted_only + candidate_only
    differences = (
        candidate_correct.astype(float) - accepted_correct.astype(float)
    )
    random = _np.random.default_rng(bootstrap_seed)
    bootstrap_changes = _np.empty(bootstrap_samples, dtype=float)
    for sample in range(bootstrap_samples):
        positions = random.integers(0, len(differences), len(differences))
        bootstrap_changes[sample] = differences[positions].mean()
    return _pd.Series(
        {
            "rows": len(y_true),
            "hard_prediction_disagreements": int((accepted != candidate).sum()),
            "accepted_only_correct": accepted_only,
            "candidate_only_correct": candidate_only,
            "net_additional_correct": candidate_only - accepted_only,
            "exact_mcnemar_p_value": float(
                _binomtest(
                    min(accepted_only, candidate_only),
                    discordant_correctness,
                    0.5,
                ).pvalue
            ),
            "accuracy_change_bootstrap_95_low": float(
                _np.quantile(bootstrap_changes, 0.025)
            ),
            "accuracy_change_bootstrap_95_high": float(
                _np.quantile(bootstrap_changes, 0.975)
            ),
        },
        name="paired local-test comparison",
    )

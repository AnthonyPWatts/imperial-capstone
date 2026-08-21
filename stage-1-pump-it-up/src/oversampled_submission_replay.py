"""Replay the three 21 August submission recipes with minority oversampling.

This module keeps the recipes and fixed full-refit iteration counts explicit so
the experiment changes the training distribution, not the model definitions.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import hashlib as _hashlib
from pathlib import Path as _Path
import time as _time

import numpy as _np
import pandas as _pd
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import balanced_accuracy_score as _balanced_accuracy_score
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import f1_score as _f1_score
from sklearn.metrics import recall_score as _recall_score

from final_model import CLASS_LABELS
from final_model import CompetitionPrediction
from final_model import build_competition_prediction
from final_model import ordered_probabilities
from final_model import validate_probabilities
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import make_initial_histogram_boosting_pipeline
from model_evaluation import make_random_forest_pipeline
from modelling_data import ModellingData


REPLAY_SEED = 20260821

RANDOM_FOREST = "random_forest"
HISTOGRAM_BOOSTING = "histogram_boosting"
XGBOOST_CHILD_1 = "xgboost_depth_8_child_1"
XGBOOST_DEPTH_6 = "xgboost_depth_6"
XGBOOST_DEPTH_7 = "xgboost_depth_7"
XGBOOST_DEPTH_8 = "xgboost_depth_8"

COMPONENT_NAMES = (
    RANDOM_FOREST,
    HISTOGRAM_BOOSTING,
    XGBOOST_CHILD_1,
    XGBOOST_DEPTH_6,
    XGBOOST_DEPTH_7,
    XGBOOST_DEPTH_8,
)

XGBOOST_COMPONENTS = {
    XGBOOST_CHILD_1: ("depth 8 child 1", 1014),
    XGBOOST_DEPTH_6: ("depth 6 broad", 1444),
    XGBOOST_DEPTH_7: ("depth 7", 1352),
    XGBOOST_DEPTH_8: ("depth 8", 1090),
}

SUBMISSION_RECIPES = {
    "random-forest-histogram-boosting": (
        (RANDOM_FOREST, 0.50),
        (HISTOGRAM_BOOSTING, 0.50),
    ),
    "55-xgboost-depth-8-child-1-current-one-hot-45-random-forest": (
        (XGBOOST_CHILD_1, 0.55),
        (RANDOM_FOREST, 0.45),
    ),
    "60-xgboost-depth-6-7-8-local-bag-40-random-forest": (
        (XGBOOST_DEPTH_6, 0.20),
        (XGBOOST_DEPTH_7, 0.20),
        (XGBOOST_DEPTH_8, 0.20),
        (RANDOM_FOREST, 0.40),
    ),
}


@_dataclass(frozen=True)
class ComponentFit:
    """Ordered probabilities and elapsed time for one fitted component."""

    probabilities: _np.ndarray
    elapsed_seconds: float


def target_count_for_replica_fraction(
    counts: _pd.Series,
    *,
    replica_fraction: float,
    target_class: str = "functional needs repair",
) -> int:
    """Scale the replicas added by the default second-largest-class policy."""

    if (
        not _np.isfinite(replica_fraction)
        or replica_fraction < 0
        or replica_fraction > 1
    ):
        raise ValueError("replica_fraction must fall between zero and one.")
    if target_class not in counts.index:
        raise KeyError(f"Target class is missing from counts: {target_class!r}.")
    minority_count = int(counts[target_class])
    other_counts = counts.drop(target_class).sort_values(ascending=False)
    if len(other_counts) < 2 or minority_count >= int(other_counts.min()):
        raise ValueError("Target class must be the unique minority class.")
    default_target = int(other_counts.iloc[1])
    added_replicas = default_target - minority_count
    retained_replicas = int(_np.floor(added_replicas * replica_fraction + 0.5))
    return minority_count + retained_replicas


def target_count_for_multiplier(
    counts: _pd.Series,
    *,
    target_multiplier: float,
    target_class: str = "functional needs repair",
) -> int:
    """Return a half-up rounded final minority count for an explicit multiplier."""

    if not _np.isfinite(target_multiplier) or target_multiplier < 1:
        raise ValueError("target_multiplier must be at least one.")
    if target_class not in counts.index:
        raise KeyError(f"Target class is missing from counts: {target_class!r}.")
    minority_count = int(counts[target_class])
    if minority_count <= 0:
        raise ValueError("Target class must contain at least one row.")
    return int(_np.floor(minority_count * target_multiplier + 0.5))


def fit_replay_components(
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    X_prediction: _pd.DataFrame,
) -> dict[str, ComponentFit]:
    """Fit every unique component once using the frozen 21 August settings."""

    fits: dict[str, ComponentFit] = {}
    pipelines = {
        RANDOM_FOREST: make_random_forest_pipeline(),
        HISTOGRAM_BOOSTING: make_initial_histogram_boosting_pipeline(),
    }
    for component, pipeline in pipelines.items():
        started = _time.perf_counter()
        pipeline.fit(X_training, y_training)
        probabilities = ordered_probabilities(pipeline, X_prediction)
        elapsed = _time.perf_counter() - started
        validate_probabilities(probabilities, len(X_prediction))
        fits[component] = ComponentFit(probabilities, elapsed)
        print(f"Fitted {component} in {elapsed:.1f} seconds.")

    for component, (variant, iterations) in XGBOOST_COMPONENTS.items():
        spec = make_xgboost_spec(variant=variant, seed=REPLAY_SEED)
        probabilities, elapsed = fit_gpu_candidate_probabilities(
            spec,
            X_training,
            y_training,
            X_prediction,
            iterations=iterations,
        )
        validate_probabilities(probabilities, len(X_prediction))
        fits[component] = ComponentFit(probabilities, elapsed)
        print(
            f"Fitted {component} ({iterations} trees) in {elapsed:.1f} seconds."
        )

    if set(fits) != set(COMPONENT_NAMES):
        raise ValueError("Replay component cache is incomplete.")
    return fits


def blend_recipe_probabilities(
    component_fits: dict[str, ComponentFit],
    recipe: tuple[tuple[str, float], ...],
) -> _np.ndarray:
    """Combine cached component probabilities after validating the recipe."""

    if not recipe:
        raise ValueError("A replay recipe cannot be empty.")
    weights = _np.asarray([weight for _, weight in recipe], dtype="float64")
    if not _np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Replay recipe weights must be finite and non-negative.")
    if not _np.isclose(weights.sum(), 1.0):
        raise ValueError("Replay recipe weights must sum to one.")
    missing = [component for component, _ in recipe if component not in component_fits]
    if missing:
        raise KeyError(f"Replay component fits are missing: {missing!r}.")

    probabilities = _np.sum(
        [
            weight * component_fits[component].probabilities
            for component, weight in recipe
        ],
        axis=0,
    )
    expected_rows = component_fits[recipe[0][0]].probabilities.shape[0]
    validate_probabilities(probabilities, expected_rows)
    return probabilities


def score_recipe_probabilities(
    y_true: _pd.Series,
    probabilities: _np.ndarray,
) -> dict[str, object]:
    """Return nominal metrics, class recalls, shares and confusion counts."""

    validate_probabilities(probabilities, len(y_true))
    predictions = _np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]
    recalls = _recall_score(
        y_true,
        predictions,
        labels=list(CLASS_LABELS),
        average=None,
        zero_division=0,
    )
    confusion = _confusion_matrix(y_true, predictions, labels=list(CLASS_LABELS))
    shares = _pd.Series(predictions).value_counts(normalize=True).reindex(
        CLASS_LABELS,
        fill_value=0.0,
    )
    return {
        "accuracy": float(_accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(_balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(
            _f1_score(
                y_true,
                predictions,
                labels=list(CLASS_LABELS),
                average="macro",
                zero_division=0,
            )
        ),
        "class_recall": {
            label: float(value)
            for label, value in zip(CLASS_LABELS, recalls, strict=True)
        },
        "prediction_share": {
            label: float(shares[label]) for label in CLASS_LABELS
        },
        "confusion_counts": {
            actual: {
                predicted: int(confusion[actual_position, predicted_position])
                for predicted_position, predicted in enumerate(CLASS_LABELS)
            }
            for actual_position, actual in enumerate(CLASS_LABELS)
        },
    }


def evaluate_replay_recipes(
    y_true: _pd.Series,
    component_fits: dict[str, ComponentFit],
) -> dict[str, dict[str, object]]:
    """Score all three frozen recipes against one labelled prediction set."""

    return {
        name: score_recipe_probabilities(
            y_true,
            blend_recipe_probabilities(component_fits, recipe),
        )
        for name, recipe in SUBMISSION_RECIPES.items()
    }


def build_replay_competition_predictions(
    modelling_data: ModellingData,
    submission_template: _pd.DataFrame,
    component_fits: dict[str, ComponentFit],
) -> dict[str, CompetitionPrediction]:
    """Build one validated competition prediction per frozen recipe."""

    predictions = {}
    for name, recipe in SUBMISSION_RECIPES.items():
        involved = [component for component, _ in recipe]
        predictions[name] = build_competition_prediction(
            modelling_data,
            submission_template,
            blend_recipe_probabilities(component_fits, recipe),
            _pd.Series(
                {
                    component: component_fits[component].elapsed_seconds
                    for component in involved
                },
                name="fit and predict seconds",
            ),
        )
    return predictions


def summarise_submission_csv(path: str | _Path) -> dict[str, object]:
    """Read and summarise an existing submission without changing it."""

    source = _Path(path)
    frame = _pd.read_csv(source)
    if list(frame.columns) != ["id", "status_group"]:
        raise ValueError(f"Submission schema is invalid: {source}.")
    if frame["id"].duplicated().any() or frame.isna().any().any():
        raise ValueError(f"Submission rows are incomplete or duplicated: {source}.")
    invalid = set(frame["status_group"]) - set(CLASS_LABELS)
    if invalid:
        raise ValueError(f"Submission contains invalid labels: {sorted(invalid)!r}.")
    shares = frame["status_group"].value_counts(normalize=True).reindex(
        CLASS_LABELS,
        fill_value=0.0,
    )
    return {
        "path": source.as_posix(),
        "rows": len(frame),
        "sha256": _hashlib.sha256(source.read_bytes()).hexdigest(),
        "prediction_share": {
            label: float(shares[label]) for label in CLASS_LABELS
        },
    }

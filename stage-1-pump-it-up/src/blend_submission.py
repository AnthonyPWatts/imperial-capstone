"""Refit validated forest-plus-boosting blend challengers for submission."""

from __future__ import annotations

import time as _time

import numpy as _np
import pandas as _pd

from final_model import CompetitionPrediction
from final_model import build_competition_prediction
from final_model import ordered_probabilities
from model_evaluation import make_calibrated_stack_pipeline
from model_evaluation import make_initial_histogram_boosting_pipeline
from model_evaluation import make_random_forest_pipeline
from modelling_data import ModellingData


def fit_weighted_vote_for_competition(
    modelling_data: ModellingData,
    submission_template: _pd.DataFrame,
    *,
    random_forest_weight: float,
) -> CompetitionPrediction:
    """Refit both components and apply a validated fixed probability weight."""

    if not _np.isfinite(random_forest_weight):
        raise ValueError("Random Forest weight must be finite.")
    if random_forest_weight < 0 or random_forest_weight > 1:
        raise ValueError("Random Forest weight must fall between zero and one.")

    histogram_boosting_weight = 1.0 - random_forest_weight
    components = (
        (
            "Random Forest",
            random_forest_weight,
            make_random_forest_pipeline(),
        ),
        (
            "histogram boosting",
            histogram_boosting_weight,
            make_initial_histogram_boosting_pipeline(),
        ),
    )
    weighted_probabilities = []
    elapsed_values = {}
    for name, weight, pipeline in components:
        started = _time.perf_counter()
        pipeline.fit(modelling_data.X_original, modelling_data.y_original)
        probabilities = ordered_probabilities(
            pipeline,
            modelling_data.X_competition,
        )
        weighted_probabilities.append(weight * probabilities)
        elapsed_values[name] = _time.perf_counter() - started

    return build_competition_prediction(
        modelling_data,
        submission_template,
        _np.sum(weighted_probabilities, axis=0),
        _pd.Series(elapsed_values, name="fit and predict seconds"),
    )


def fit_calibrated_stack_for_competition(
    modelling_data: ModellingData,
    submission_template: _pd.DataFrame,
) -> CompetitionPrediction:
    """Fit the inner-cross-fitted calibrated stack on every labelled row."""

    started = _time.perf_counter()
    pipeline = make_calibrated_stack_pipeline()
    pipeline.fit(modelling_data.X_original, modelling_data.y_original)
    probabilities = ordered_probabilities(
        pipeline,
        modelling_data.X_competition,
    )
    elapsed = _time.perf_counter() - started
    return build_competition_prediction(
        modelling_data,
        submission_template,
        probabilities,
        _pd.Series(
            {"calibrated stack": elapsed},
            name="fit and predict seconds",
        ),
    )

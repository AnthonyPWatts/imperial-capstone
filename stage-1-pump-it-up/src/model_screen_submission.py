"""Select and refit validated candidates from the model-family screen."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import time as _time
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd

from final_model import CompetitionPrediction
from final_model import build_competition_prediction
from final_model import ordered_probabilities
from final_model import validate_probabilities
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import median_selected_iterations
from model_evaluation import CandidateEvaluation
from model_evaluation import make_initial_histogram_boosting_pipeline
from model_evaluation import make_random_forest_pipeline
from modelling_data import ModellingData


MINIMUM_SECOND_CANDIDATE_DISAGREEMENT = 0.01


@_dataclass(frozen=True)
class ScreenCompetitionCandidate:
    """One selected screen recipe and its validated competition prediction."""

    name: str
    recipe: tuple[tuple[str, float], ...]
    prediction: CompetitionPrediction
    selected_iterations: _pd.Series


def select_submission_candidate_names(
    screen_payload: dict,
    *,
    maximum_candidates: int = 2,
    minimum_second_disagreement: float = MINIMUM_SECOND_CANDIDATE_DISAGREEMENT,
) -> list[str]:
    """Select the strongest gate-passers, requiring a distinct second choice."""

    if maximum_candidates < 1:
        raise ValueError("At least one candidate must be permitted.")
    if minimum_second_disagreement < 0 or minimum_second_disagreement > 1:
        raise ValueError("Candidate disagreement must fall between zero and one.")

    summary = screen_payload["summary"]
    evaluations = screen_payload["evaluations"]
    passing_names = summary.index[summary["passes_gate"].astype(bool)].tolist()
    if not passing_names:
        return []

    selected = [passing_names[0]]
    first_predictions = _oof_predictions(evaluations[passing_names[0]])
    for name in passing_names[1:]:
        disagreement = _np.mean(
            _oof_predictions(evaluations[name]) != first_predictions
        )
        if disagreement >= minimum_second_disagreement:
            selected.append(name)
        if len(selected) >= maximum_candidates:
            break
    return selected


def fit_selected_screen_candidates(
    modelling_data: ModellingData,
    submission_template: _pd.DataFrame,
    screen_payload: dict,
    trial_payloads: _Mapping[str, dict],
    selected_names: list[str],
) -> list[ScreenCompetitionCandidate]:
    """Refit the unique recipe members once and construct each candidate."""

    if not selected_names:
        return []
    evaluations = screen_payload["evaluations"]
    recipes = screen_payload["recipes"]
    incumbent_name = screen_payload["incumbent_name"]
    incumbent_payload = trial_payloads[incumbent_name]
    incumbent_components = incumbent_payload["components"]
    component_names = {
        label: evaluation.model_name
        for label, evaluation in incumbent_components.items()
    }

    probability_cache: dict[str, _np.ndarray] = {}
    seconds: dict[str, float] = {}
    selected_iterations: dict[str, int] = {}

    def resolve(component_name: str) -> _np.ndarray:
        if component_name in probability_cache:
            return probability_cache[component_name]
        if component_name == incumbent_name:
            values = _np.mean(
                [
                    resolve(component_names["Random Forest"]),
                    resolve(component_names["histogram boosting"]),
                ],
                axis=0,
            )
        elif component_name == component_names["Random Forest"]:
            values, elapsed = _fit_pipeline_probabilities(
                make_random_forest_pipeline(),
                modelling_data,
            )
            seconds[component_name] = elapsed
        elif component_name == component_names["histogram boosting"]:
            values, elapsed = _fit_pipeline_probabilities(
                make_initial_histogram_boosting_pipeline(),
                modelling_data,
            )
            seconds[component_name] = elapsed
        elif (
            component_name in recipes
            and recipes[component_name] != ((component_name, 1.0),)
        ):
            values = _np.sum(
                [
                    weight * resolve(nested_component)
                    for nested_component, weight in recipes[component_name]
                ],
                axis=0,
            )
        else:
            payload = trial_payloads[component_name]
            spec = payload["spec"]
            evaluation = payload["evaluation"]
            iterations = median_selected_iterations(evaluation)
            values, elapsed = fit_gpu_candidate_probabilities(
                spec,
                modelling_data.X_original,
                modelling_data.y_original,
                modelling_data.X_competition,
                iterations=iterations,
            )
            seconds[component_name] = elapsed
            selected_iterations[component_name] = iterations
        validate_probabilities(values, len(modelling_data.X_competition))
        probability_cache[component_name] = values
        return values

    candidates = []
    for name in selected_names:
        if name not in evaluations or name not in recipes:
            raise KeyError(f"Selected screen candidate is missing: {name!r}.")
        recipe = tuple(recipes[name])
        probabilities = _np.sum(
            [weight * resolve(component) for component, weight in recipe],
            axis=0,
        )
        validate_probabilities(probabilities, len(modelling_data.X_competition))
        involved = _expanded_component_names(
            recipe,
            incumbent_name=incumbent_name,
            component_names=component_names,
            recipes=recipes,
        )
        candidates.append(
            ScreenCompetitionCandidate(
                name=name,
                recipe=recipe,
                prediction=build_competition_prediction(
                    modelling_data,
                    submission_template,
                    probabilities,
                    _pd.Series(
                        {
                            component: seconds[component]
                            for component in involved
                        },
                        name="fit and predict seconds",
                    ),
                ),
                selected_iterations=_pd.Series(
                    {
                        component: selected_iterations[component]
                        for component in involved
                        if component in selected_iterations
                    },
                    name="full-data tree count",
                    dtype="int64",
                ),
            )
        )
    return candidates


def _fit_pipeline_probabilities(
    pipeline: object,
    modelling_data: ModellingData,
) -> tuple[_np.ndarray, float]:
    started = _time.perf_counter()
    pipeline.fit(modelling_data.X_original, modelling_data.y_original)
    probabilities = ordered_probabilities(
        pipeline,
        modelling_data.X_competition,
    )
    return probabilities, _time.perf_counter() - started


def _expanded_component_names(
    recipe: tuple[tuple[str, float], ...],
    *,
    incumbent_name: str,
    component_names: dict[str, str],
    recipes: dict,
) -> list[str]:
    names = []
    def expand(component: str) -> None:
        if component == incumbent_name:
            names.extend(
                [
                    component_names["Random Forest"],
                    component_names["histogram boosting"],
                ]
            )
        elif (
            component in recipes
            and recipes[component] != ((component, 1.0),)
        ):
            for nested_component, _ in recipes[component]:
                expand(nested_component)
        else:
            names.append(component)

    for component, _ in recipe:
        expand(component)
    return list(dict.fromkeys(names))


def _oof_predictions(evaluation: CandidateEvaluation) -> _np.ndarray:
    return evaluation.out_of_fold_probabilities.to_numpy().argmax(axis=1)

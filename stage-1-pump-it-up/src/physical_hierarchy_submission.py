"""Refit the fixed exploratory physical-hierarchy submission recipes."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
from functools import partial as _partial
import time as _time

import numpy as _np
import pandas as _pd

from final_model import CompetitionPrediction
from final_model import build_competition_prediction
from final_model import ordered_probabilities
from final_model import validate_probabilities
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import median_selected_iterations
from model_evaluation import make_random_forest_pipeline
from modelling_data import ModellingData
from physical_hierarchy_evaluation import PhysicalHierarchyTrial
from physical_hierarchy_evaluation import RANDOM_FOREST_WEIGHT
from physical_hierarchy_evaluation import XGBOOST_WEIGHT
from physical_hierarchy_features import PHYSICAL_HIERARCHY_POLICIES
from physical_hierarchy_features import PhysicalHierarchyPolicy
from physical_hierarchy_features import make_physical_hierarchy_preprocessor


@_dataclass(frozen=True)
class PhysicalHierarchySubmissionSpec:
    """One fixed XGBoost and Random Forest hierarchy-policy pairing."""

    key: str
    label: str
    xgboost_policy: PhysicalHierarchyPolicy
    random_forest_policy: PhysicalHierarchyPolicy


@_dataclass(frozen=True)
class PhysicalHierarchyCompetitionCandidate:
    """One validated exploratory competition prediction and its provenance."""

    spec: PhysicalHierarchySubmissionSpec
    prediction: CompetitionPrediction
    selected_iterations: int


SOURCE_PLUS_CLASS = PHYSICAL_HIERARCHY_POLICIES["source"]["source_plus_class"]
WATERPOINT_BOTH_LEVELS = PHYSICAL_HIERARCHY_POLICIES["waterpoint"]["both_levels"]

PHYSICAL_HIERARCHY_SUBMISSION_SPECS = (
    PhysicalHierarchySubmissionSpec(
        key="source-plus-class",
        label="Source plus source class",
        xgboost_policy=SOURCE_PLUS_CLASS,
        random_forest_policy=SOURCE_PLUS_CLASS,
    ),
    PhysicalHierarchySubmissionSpec(
        key="waterpoint-both-levels",
        label="Waterpoint type plus group",
        xgboost_policy=WATERPOINT_BOTH_LEVELS,
        random_forest_policy=WATERPOINT_BOTH_LEVELS,
    ),
    PhysicalHierarchySubmissionSpec(
        key="waterpoint-xgb-source-rf",
        label="Waterpoint-aware XGBoost plus source-class Random Forest",
        xgboost_policy=WATERPOINT_BOTH_LEVELS,
        random_forest_policy=SOURCE_PLUS_CLASS,
    ),
)


def fit_physical_hierarchy_submission_candidates(
    modelling_data: ModellingData,
    submission_template: _pd.DataFrame,
    source_trial: PhysicalHierarchyTrial,
    waterpoint_trial: PhysicalHierarchyTrial,
) -> list[PhysicalHierarchyCompetitionCandidate]:
    """Refit four unique components once and build the three fixed entries."""

    trials = {
        _policy_key(SOURCE_PLUS_CLASS): source_trial,
        _policy_key(WATERPOINT_BOTH_LEVELS): waterpoint_trial,
    }
    for policy_key, trial in trials.items():
        if _policy_key(trial.policy) != policy_key:
            raise ValueError(
                "A hierarchy submission trial does not match its fixed policy: "
                f"expected {policy_key!r}, found {_policy_key(trial.policy)!r}."
            )

    probability_cache: dict[tuple[str, str, str], _np.ndarray] = {}
    elapsed_cache: dict[tuple[str, str, str], float] = {}
    iteration_cache: dict[tuple[str, str, str], int] = {}

    def fit_xgboost(policy: PhysicalHierarchyPolicy) -> _np.ndarray:
        component_key = ("xgboost", policy.family, policy.key)
        if component_key not in probability_cache:
            trial = trials[_policy_key(policy)]
            iterations = median_selected_iterations(trial.xgboost)
            spec = _replace(
                make_xgboost_spec(variant="depth 8 child 1"),
                name=f"XGBoost depth 8 child 1 [{policy.family}:{policy.key}]",
                feature_policy=f"{policy.family}:{policy.key}",
            )
            probabilities, elapsed = fit_gpu_candidate_probabilities(
                spec,
                modelling_data.X_original,
                modelling_data.y_original,
                modelling_data.X_competition,
                iterations=iterations,
                preprocessor_factory=_partial(
                    make_physical_hierarchy_preprocessor,
                    policy,
                ),
            )
            validate_probabilities(
                probabilities,
                len(modelling_data.X_competition),
            )
            probability_cache[component_key] = probabilities
            elapsed_cache[component_key] = elapsed
            iteration_cache[component_key] = iterations
        return probability_cache[component_key]

    def fit_random_forest(policy: PhysicalHierarchyPolicy) -> _np.ndarray:
        component_key = ("random_forest", policy.family, policy.key)
        if component_key not in probability_cache:
            pipeline = make_random_forest_pipeline(
                preprocessor_factory=_partial(
                    make_physical_hierarchy_preprocessor,
                    policy,
                )
            )
            started = _time.perf_counter()
            pipeline.fit(modelling_data.X_original, modelling_data.y_original)
            probabilities = ordered_probabilities(
                pipeline,
                modelling_data.X_competition,
            )
            elapsed = _time.perf_counter() - started
            validate_probabilities(
                probabilities,
                len(modelling_data.X_competition),
            )
            probability_cache[component_key] = probabilities
            elapsed_cache[component_key] = elapsed
        return probability_cache[component_key]

    candidates = []
    for spec in PHYSICAL_HIERARCHY_SUBMISSION_SPECS:
        xgboost_key = ("xgboost", spec.xgboost_policy.family, spec.xgboost_policy.key)
        forest_key = (
            "random_forest",
            spec.random_forest_policy.family,
            spec.random_forest_policy.key,
        )
        probabilities = (
            XGBOOST_WEIGHT * fit_xgboost(spec.xgboost_policy)
            + RANDOM_FOREST_WEIGHT * fit_random_forest(spec.random_forest_policy)
        )
        validate_probabilities(probabilities, len(modelling_data.X_competition))
        candidates.append(
            PhysicalHierarchyCompetitionCandidate(
                spec=spec,
                prediction=build_competition_prediction(
                    modelling_data,
                    submission_template,
                    probabilities,
                    _pd.Series(
                        {
                            _format_component_key(xgboost_key): elapsed_cache[
                                xgboost_key
                            ],
                            _format_component_key(forest_key): elapsed_cache[forest_key],
                        },
                        name="fit and predict seconds",
                    ),
                ),
                selected_iterations=iteration_cache[xgboost_key],
            )
        )
    return candidates


def _policy_key(policy: PhysicalHierarchyPolicy) -> tuple[str, str]:
    return policy.family, policy.key


def _format_component_key(component_key: tuple[str, str, str]) -> str:
    return ":".join(component_key)

"""Evaluate management hierarchy policies with the accepted model recipe."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
from functools import partial as _partial

import pandas as _pd
from sklearn.model_selection import PredefinedSplit as _PredefinedSplit

from data_partitioning import PartitionedData
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from management_features import SELECTION_GATE_ACCURACY
from management_features import SELECTION_GATE_REPAIR_RECALL
from management_features import SELECTION_GATE_WORST_FOLD
from management_features import ManagementFeaturePolicy
from management_features import make_management_preprocessor
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_evaluation import evaluate_weighted_soft_vote


XGBOOST_WEIGHT = 0.55
RANDOM_FOREST_WEIGHT = 0.45


@_dataclass(frozen=True)
class ManagementTrial:
    """Component and blend evidence for one management hierarchy policy."""

    policy: ManagementFeaturePolicy
    engineered_features: int
    transformed_features_fold_1: int
    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    blend: CandidateEvaluation


def evaluate_management_policy(
    policy: ManagementFeaturePolicy,
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
    *,
    evaluation_label: str = "frozen stratified folds",
) -> ManagementTrial:
    """Evaluate the accepted 55:45 recipe under one management policy."""

    preprocessor_factory = _partial(make_management_preprocessor, policy)
    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name=f"XGBoost depth 8 child 1 [{policy.key}; {evaluation_label}]",
        feature_policy=policy.key,
    )
    xgboost = evaluate_gpu_candidate(
        xgboost_spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=preprocessor_factory,
    )
    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=preprocessor_factory,
        model_name=f"Random Forest [{policy.key}; {evaluation_label}]",
    )
    blend = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        xgboost,
        random_forest,
        weights=[XGBOOST_WEIGHT, RANDOM_FOREST_WEIGHT],
        model_name=(
            f"55% XGBoost + 45% Random Forest "
            f"[{policy.key}; {evaluation_label}]"
        ),
    )

    first_training_positions, _ = next(cross_validation.split())
    first_preprocessor = preprocessor_factory()
    first_preprocessor.fit(
        partitioned_data.X_development.iloc[first_training_positions],
        partitioned_data.y_development.iloc[first_training_positions],
    )
    return ManagementTrial(
        policy=policy,
        engineered_features=len(
            first_preprocessor.named_steps[
                "feature_engineering"
            ].get_feature_names_out()
        ),
        transformed_features_fold_1=len(
            first_preprocessor.get_feature_names_out()
        ),
        xgboost=xgboost,
        random_forest=random_forest,
        blend=blend,
    )


def summarise_management_trials(
    trials: list[ManagementTrial] | tuple[ManagementTrial, ...],
) -> _pd.DataFrame:
    """Compare management policies with the predeclared promotion gate."""

    if not trials:
        raise ValueError("At least one management trial is required.")
    by_key = {trial.policy.key: trial for trial in trials}
    if len(by_key) != len(trials):
        raise ValueError("Management trial policies must be unique.")
    if "baseline" not in by_key:
        raise ValueError("The baseline management trial is required.")

    baseline = by_key["baseline"].blend
    baseline_accuracy = float(baseline.metric_summary.loc["accuracy", "mean"])
    baseline_repair = float(
        baseline.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    rows = []
    for trial in trials:
        candidate = trial.blend
        fold_delta = candidate.fold_metrics["accuracy"] - baseline.fold_metrics[
            "accuracy"
        ]
        mean_accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair_recall = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        accuracy_change = mean_accuracy - baseline_accuracy
        repair_change = repair_recall - baseline_repair
        fold_wins = int(fold_delta.gt(0).sum())
        worst_fold_change = float(fold_delta.min())
        passes_gate = trial.policy.key != "baseline" and (
            accuracy_change >= SELECTION_GATE_ACCURACY
            and fold_wins >= 3
            and worst_fold_change >= SELECTION_GATE_WORST_FOLD
            and repair_change >= SELECTION_GATE_REPAIR_RECALL
        )
        rows.append(
            {
                "policy": trial.policy.key,
                "label": trial.policy.label,
                "engineered_features": trial.engineered_features,
                "transformed_features_fold_1": trial.transformed_features_fold_1,
                "mean_accuracy": mean_accuracy,
                "accuracy_change": accuracy_change,
                "fold_wins": fold_wins,
                "worst_fold_change": worst_fold_change,
                "repair_recall": repair_recall,
                "repair_recall_change": repair_change,
                "non_functional_recall": float(
                    candidate.metric_summary.loc[
                        "recall: non functional",
                        "mean",
                    ]
                ),
                "xgboost_accuracy": float(
                    trial.xgboost.metric_summary.loc["accuracy", "mean"]
                ),
                "random_forest_accuracy": float(
                    trial.random_forest.metric_summary.loc["accuracy", "mean"]
                ),
                "passes_gate": passes_gate,
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("policy")
        .sort_values(["mean_accuracy", "policy"], ascending=[False, True])
    )

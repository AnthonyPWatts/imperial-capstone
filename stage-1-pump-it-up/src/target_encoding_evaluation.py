"""Evaluate cross-fitted multiclass target-encoding policies."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
from functools import partial as _partial

import pandas as _pd
from sklearn.model_selection import PredefinedSplit as _PredefinedSplit

from data_partitioning import PartitionedData
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_evaluation import evaluate_weighted_soft_vote
from target_encoding_features import SELECTION_GATE_ACCURACY
from target_encoding_features import SELECTION_GATE_REPAIR_RECALL
from target_encoding_features import SELECTION_GATE_WORST_FOLD
from target_encoding_features import TargetEncodingPolicy
from target_encoding_features import make_target_encoding_preprocessor


XGBOOST_WEIGHT = 0.55
RANDOM_FOREST_WEIGHT = 0.45


@_dataclass(frozen=True)
class TargetEncodingTrial:
    """Component and blend evidence for one target-encoding policy."""

    policy: TargetEncodingPolicy
    engineered_features: int
    transformed_features_fold_1: int
    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    blend: CandidateEvaluation


def evaluate_target_encoding_policy(
    policy: TargetEncodingPolicy,
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
) -> TargetEncodingTrial:
    """Evaluate the accepted 55:45 recipe with cross-fitted target encoding."""

    preprocessor_factory = _partial(make_target_encoding_preprocessor, policy)
    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name=f"XGBoost depth 8 child 1 [{policy.key}]",
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
        model_name=f"Random Forest [{policy.key}]",
    )
    blend = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        xgboost,
        random_forest,
        weights=[XGBOOST_WEIGHT, RANDOM_FOREST_WEIGHT],
        model_name=f"55% XGBoost + 45% Random Forest [{policy.key}]",
    )

    first_training_positions, _ = next(cross_validation.split())
    first_preprocessor = preprocessor_factory()
    first_preprocessor.fit(
        partitioned_data.X_development.iloc[first_training_positions],
        partitioned_data.y_development.iloc[first_training_positions],
    )
    return TargetEncodingTrial(
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


def summarise_target_encoding_trials(
    accepted: CandidateEvaluation,
    trials: list[TargetEncodingTrial] | tuple[TargetEncodingTrial, ...],
) -> _pd.DataFrame:
    """Compare target encodings with the unchanged accepted ensemble."""

    if not trials:
        raise ValueError("At least one target-encoding trial is required.")
    if len({trial.policy.key for trial in trials}) != len(trials):
        raise ValueError("Target-encoding trial policies must be unique.")
    baseline_accuracy = float(accepted.metric_summary.loc["accuracy", "mean"])
    baseline_repair = float(
        accepted.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = [
        {
            "policy": "accepted_baseline",
            "label": "Accepted feature policy",
            "identity_features": 0,
            "engineered_features": 29,
            "transformed_features_fold_1": 301,
            "mean_accuracy": baseline_accuracy,
            "accuracy_change": 0.0,
            "fold_wins": 0,
            "worst_fold_change": 0.0,
            "repair_recall": baseline_repair,
            "repair_recall_change": 0.0,
            "non_functional_recall": float(
                accepted.metric_summary.loc["recall: non functional", "mean"]
            ),
            "xgboost_accuracy": _pd.NA,
            "random_forest_accuracy": _pd.NA,
            "passes_gate": False,
        }
    ]
    for trial in trials:
        candidate = trial.blend
        _validate_alignment(accepted, candidate)
        fold_change = (
            candidate.fold_metrics["accuracy"]
            - accepted.fold_metrics["accuracy"]
        )
        accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        accuracy_change = accuracy - baseline_accuracy
        repair_change = repair - baseline_repair
        rows.append(
            {
                "policy": trial.policy.key,
                "label": trial.policy.label,
                "identity_features": len(trial.policy.identity_features),
                "engineered_features": trial.engineered_features,
                "transformed_features_fold_1": (
                    trial.transformed_features_fold_1
                ),
                "mean_accuracy": accuracy,
                "accuracy_change": accuracy_change,
                "fold_wins": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall": repair,
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
                "passes_gate": (
                    accuracy_change >= SELECTION_GATE_ACCURACY
                    and int(fold_change.gt(0).sum()) >= 3
                    and float(fold_change.min()) >= SELECTION_GATE_WORST_FOLD
                    and repair_change >= SELECTION_GATE_REPAIR_RECALL
                ),
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("policy")
        .sort_values(["mean_accuracy", "policy"], ascending=[False, True])
    )


def _validate_alignment(
    accepted: CandidateEvaluation,
    candidate: CandidateEvaluation,
) -> None:
    if accepted.cross_validation_fingerprint != candidate.cross_validation_fingerprint:
        raise ValueError("Target-encoding candidate uses different frozen folds.")
    if not accepted.out_of_fold_probabilities.index.equals(
        candidate.out_of_fold_probabilities.index
    ):
        raise ValueError("Target-encoding candidate rows do not align.")

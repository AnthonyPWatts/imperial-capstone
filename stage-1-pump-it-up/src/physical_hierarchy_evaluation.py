"""Evaluate physical-category hierarchy policies with the accepted recipe."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
from functools import partial as _partial

import pandas as _pd

from data_partitioning import PartitionedData
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_evaluation import evaluate_weighted_soft_vote
from physical_hierarchy_features import PARSIMONY_MAXIMUM_ACCURACY_LOSS
from physical_hierarchy_features import SELECTION_GATE_ACCURACY
from physical_hierarchy_features import SELECTION_GATE_REPAIR_RECALL
from physical_hierarchy_features import SELECTION_GATE_WORST_FOLD
from physical_hierarchy_features import PhysicalHierarchyPolicy
from physical_hierarchy_features import make_physical_hierarchy_preprocessor


XGBOOST_WEIGHT = 0.55
RANDOM_FOREST_WEIGHT = 0.45
ACCEPTED_BASELINE_NAME = (
    "55% XGBoost depth 8 child 1 [current one-hot] + 45% Random Forest"
)
ACCEPTED_XGBOOST_NAME = "XGBoost depth 8 child 1 [current one-hot]"
ACCEPTED_RANDOM_FOREST_NAME = "Random Forest"


@_dataclass(frozen=True)
class PhysicalHierarchyTrial:
    """Component and blend evidence for one physical hierarchy policy."""

    policy: PhysicalHierarchyPolicy
    engineered_features: int
    transformed_features_fold_1: int
    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    blend: CandidateEvaluation


@_dataclass(frozen=True)
class ComponentHybridSpec:
    """One no-refit cross-policy XGBoost and Random Forest pairing."""

    key: str
    label: str
    xgboost_trial: str
    random_forest_trial: str


COMPONENT_HYBRID_SPECS = (
    ComponentHybridSpec(
        key="extraction_xgb_baseline_rf",
        label="Extraction-group XGBoost + baseline Random Forest",
        xgboost_trial="extraction_group",
        random_forest_trial="baseline",
    ),
    ComponentHybridSpec(
        key="baseline_xgb_extraction_rf",
        label="Baseline XGBoost + extraction-group Random Forest",
        xgboost_trial="baseline",
        random_forest_trial="extraction_group",
    ),
    ComponentHybridSpec(
        key="baseline_xgb_source_rf",
        label="Baseline XGBoost + source-class Random Forest",
        xgboost_trial="baseline",
        random_forest_trial="source_class",
    ),
    ComponentHybridSpec(
        key="quality_xgb_baseline_rf",
        label="Quality-group XGBoost + baseline Random Forest",
        xgboost_trial="quality_group",
        random_forest_trial="baseline",
    ),
    ComponentHybridSpec(
        key="waterpoint_xgb_baseline_rf",
        label="Waterpoint-both XGBoost + baseline Random Forest",
        xgboost_trial="waterpoint_both",
        random_forest_trial="baseline",
    ),
    ComponentHybridSpec(
        key="waterpoint_xgb_source_rf",
        label="Waterpoint-both XGBoost + source-class Random Forest",
        xgboost_trial="waterpoint_both",
        random_forest_trial="source_class",
    ),
)


def evaluate_physical_hierarchy_policy(
    policy: PhysicalHierarchyPolicy,
    partitioned_data: PartitionedData,
    cross_validation: object,
    *,
    evaluation_label: str = "frozen stratified folds",
) -> PhysicalHierarchyTrial:
    """Evaluate the accepted 55:45 recipe under one hierarchy policy."""

    preprocessor_factory = _partial(make_physical_hierarchy_preprocessor, policy)
    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name=(
            f"XGBoost depth 8 child 1 "
            f"[{policy.family}:{policy.key}; {evaluation_label}]"
        ),
        feature_policy=f"{policy.family}:{policy.key}",
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
        model_name=(
            f"Random Forest [{policy.family}:{policy.key}; {evaluation_label}]"
        ),
    )
    blend = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        xgboost,
        random_forest,
        weights=[XGBOOST_WEIGHT, RANDOM_FOREST_WEIGHT],
        model_name=(
            f"55% XGBoost + 45% Random Forest "
            f"[{policy.family}:{policy.key}; {evaluation_label}]"
        ),
    )
    engineered_features, transformed_features = _first_fold_feature_counts(
        policy,
        partitioned_data,
        cross_validation,
    )
    return PhysicalHierarchyTrial(
        policy=policy,
        engineered_features=engineered_features,
        transformed_features_fold_1=transformed_features,
        xgboost=xgboost,
        random_forest=random_forest,
        blend=blend,
    )


def make_cached_baseline_trial(
    policy: PhysicalHierarchyPolicy,
    partitioned_data: PartitionedData,
    cross_validation: object,
    evaluations: dict[str, CandidateEvaluation],
) -> PhysicalHierarchyTrial:
    """Reuse the saved accepted components while verifying the feature shape."""

    required = (
        ACCEPTED_XGBOOST_NAME,
        ACCEPTED_RANDOM_FOREST_NAME,
        ACCEPTED_BASELINE_NAME,
    )
    missing = [name for name in required if name not in evaluations]
    if missing:
        raise ValueError(f"Accepted model cache is missing: {missing!r}.")
    selected = [evaluations[name] for name in required]
    for evaluation in selected:
        if (
            evaluation.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Accepted model cache uses different frozen folds.")
    engineered_features, transformed_features = _first_fold_feature_counts(
        policy,
        partitioned_data,
        cross_validation,
    )
    return PhysicalHierarchyTrial(
        policy=policy,
        engineered_features=engineered_features,
        transformed_features_fold_1=transformed_features,
        xgboost=evaluations[ACCEPTED_XGBOOST_NAME],
        random_forest=evaluations[ACCEPTED_RANDOM_FOREST_NAME],
        blend=evaluations[ACCEPTED_BASELINE_NAME],
    )


def summarise_physical_hierarchy_trials(
    trials: list[PhysicalHierarchyTrial] | tuple[PhysicalHierarchyTrial, ...],
) -> _pd.DataFrame:
    """Compare one family's policies with accuracy and parsimony gates."""

    if not trials:
        raise ValueError("At least one physical hierarchy trial is required.")
    families = {trial.policy.family for trial in trials}
    if len(families) != 1:
        raise ValueError("A summary cannot mix physical hierarchy families.")
    by_key = {trial.policy.key: trial for trial in trials}
    if len(by_key) != len(trials):
        raise ValueError("Physical hierarchy trial policies must be unique.")
    if "baseline" not in by_key:
        raise ValueError("The baseline hierarchy trial is required.")

    baseline = by_key["baseline"].blend
    baseline_features = by_key["baseline"].transformed_features_fold_1
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
        is_challenger = trial.policy.key != "baseline"
        passes_gate = is_challenger and (
            accuracy_change >= SELECTION_GATE_ACCURACY
            and fold_wins >= 3
            and worst_fold_change >= SELECTION_GATE_WORST_FOLD
            and repair_change >= SELECTION_GATE_REPAIR_RECALL
        )
        parsimony_candidate = is_challenger and (
            trial.transformed_features_fold_1 < baseline_features
            and accuracy_change >= PARSIMONY_MAXIMUM_ACCURACY_LOSS
            and worst_fold_change >= SELECTION_GATE_WORST_FOLD
            and repair_change >= SELECTION_GATE_REPAIR_RECALL
        )
        rows.append(
            {
                "policy": trial.policy.key,
                "label": trial.policy.label,
                "engineered_features": trial.engineered_features,
                "transformed_features_fold_1": trial.transformed_features_fold_1,
                "transformed_feature_change": (
                    trial.transformed_features_fold_1 - baseline_features
                ),
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
                "parsimony_candidate": parsimony_candidate,
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("policy")
        .sort_values(["mean_accuracy", "policy"], ascending=[False, True])
    )


def evaluate_component_hybrids(
    partitioned_data: PartitionedData,
    cross_validation: object,
    trials: dict[str, PhysicalHierarchyTrial],
) -> dict[str, CandidateEvaluation]:
    """Cross only components whose policy-specific mean improved."""

    required = {
        spec.xgboost_trial
        for spec in COMPONENT_HYBRID_SPECS
    } | {
        spec.random_forest_trial
        for spec in COMPONENT_HYBRID_SPECS
    }
    missing = required - set(trials)
    if missing:
        raise ValueError(f"Component hybrid trials are missing: {sorted(missing)!r}.")

    evaluations = {}
    for spec in COMPONENT_HYBRID_SPECS:
        evaluations[spec.key] = evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            trials[spec.xgboost_trial].xgboost,
            trials[spec.random_forest_trial].random_forest,
            weights=[XGBOOST_WEIGHT, RANDOM_FOREST_WEIGHT],
            model_name=spec.label,
        )
    return evaluations


def summarise_component_hybrids(
    baseline: CandidateEvaluation,
    hybrids: dict[str, CandidateEvaluation],
) -> _pd.DataFrame:
    """Compare no-refit component crosses with the accepted vote."""

    baseline_repair = float(
        baseline.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    rows = []
    for key, evaluation in hybrids.items():
        fold_delta = evaluation.fold_metrics["accuracy"] - baseline.fold_metrics[
            "accuracy"
        ]
        accuracy_change = float(fold_delta.mean())
        repair_recall = float(
            evaluation.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        repair_change = repair_recall - baseline_repair
        fold_wins = int(fold_delta.gt(0).sum())
        worst_fold_change = float(fold_delta.min())
        rows.append(
            {
                "hybrid": key,
                "label": evaluation.model_name,
                "mean_accuracy": float(
                    evaluation.metric_summary.loc["accuracy", "mean"]
                ),
                "accuracy_change": accuracy_change,
                "fold_wins": fold_wins,
                "worst_fold_change": worst_fold_change,
                "repair_recall": repair_recall,
                "repair_recall_change": repair_change,
                "non_functional_recall": float(
                    evaluation.metric_summary.loc[
                        "recall: non functional",
                        "mean",
                    ]
                ),
                "passes_gate": (
                    accuracy_change >= SELECTION_GATE_ACCURACY
                    and fold_wins >= 3
                    and worst_fold_change >= SELECTION_GATE_WORST_FOLD
                    and repair_change >= SELECTION_GATE_REPAIR_RECALL
                ),
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("hybrid")
        .sort_values(["mean_accuracy", "hybrid"], ascending=[False, True])
    )


def _first_fold_feature_counts(
    policy: PhysicalHierarchyPolicy,
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> tuple[int, int]:
    training_positions, _ = next(cross_validation.split())
    preprocessor = make_physical_hierarchy_preprocessor(policy)
    preprocessor.fit(
        partitioned_data.X_development.iloc[training_positions],
        partitioned_data.y_development.iloc[training_positions],
    )
    engineered = len(
        preprocessor.named_steps["feature_engineering"].get_feature_names_out()
    )
    transformed = len(preprocessor.get_feature_names_out())
    return engineered, transformed

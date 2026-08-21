"""Evaluate geography policies with fixed models and leakage-safe folds."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace
import hashlib as _hashlib
from functools import partial as _partial

import numpy as _np
import pandas as _pd
from sklearn.model_selection import PredefinedSplit as _PredefinedSplit
from sklearn.model_selection import StratifiedGroupKFold as _StratifiedGroupKFold

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from geography_features import GEOGRAPHY_SELECTION_GATE_ACCURACY
from geography_features import GEOGRAPHY_SELECTION_GATE_REPAIR_RECALL
from geography_features import GEOGRAPHY_SELECTION_GATE_WORST_FOLD
from geography_features import GeographyFeaturePolicy
from geography_features import make_geography_preprocessor
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_random_forest
from model_evaluation import evaluate_weighted_soft_vote


XGBOOST_WEIGHT = 0.55
RANDOM_FOREST_WEIGHT = 0.45
GROUPED_SENSITIVITY_SEED = 20260821


@_dataclass(frozen=True)
class GeographyTrial:
    """Component and blend evidence for one geography policy."""

    policy: GeographyFeaturePolicy
    engineered_features: int
    transformed_features_fold_1: int
    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    blend: CandidateEvaluation


def evaluate_geography_policy(
    policy: GeographyFeaturePolicy,
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
    *,
    evaluation_label: str = "frozen stratified folds",
) -> GeographyTrial:
    """Evaluate the accepted 55:45 recipe under one feature policy."""

    preprocessor_factory = _partial(make_geography_preprocessor, policy)
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
    transformed_features = len(first_preprocessor.get_feature_names_out())
    engineered_features = len(
        first_preprocessor.named_steps[
            "feature_engineering"
        ].get_feature_names_out()
    )
    return GeographyTrial(
        policy=policy,
        engineered_features=engineered_features,
        transformed_features_fold_1=transformed_features,
        xgboost=xgboost,
        random_forest=random_forest,
        blend=blend,
    )


def summarise_geography_trials(
    trials: list[GeographyTrial] | tuple[GeographyTrial, ...],
) -> _pd.DataFrame:
    """Compare policies with the pre-declared frozen-fold acceptance gate."""

    if not trials:
        raise ValueError("At least one geography trial is required.")
    by_key = {trial.policy.key: trial for trial in trials}
    if len(by_key) != len(trials):
        raise ValueError("Geography trial policies must be unique.")
    if "baseline" not in by_key:
        raise ValueError("The baseline geography trial is required.")

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
            accuracy_change >= GEOGRAPHY_SELECTION_GATE_ACCURACY
            and fold_wins >= 3
            and worst_fold_change >= GEOGRAPHY_SELECTION_GATE_WORST_FOLD
            and repair_change >= GEOGRAPHY_SELECTION_GATE_REPAIR_RECALL
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


def evaluate_cross_policy_hybrids(
    partitioned_data: PartitionedData,
    cross_validation: _PredefinedSplit,
    baseline: GeographyTrial,
    coordinate_centroids: GeographyTrial,
) -> tuple[CandidateEvaluation, CandidateEvaluation]:
    """Cross the two policies' components without refitting or changing weights."""

    baseline_xgboost = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        baseline.xgboost,
        coordinate_centroids.random_forest,
        weights=[XGBOOST_WEIGHT, RANDOM_FOREST_WEIGHT],
        model_name="baseline XGBoost + centroid Random Forest",
    )
    centroid_xgboost = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        coordinate_centroids.xgboost,
        baseline.random_forest,
        weights=[XGBOOST_WEIGHT, RANDOM_FOREST_WEIGHT],
        model_name="centroid XGBoost + baseline Random Forest",
    )
    return baseline_xgboost, centroid_xgboost


def summarise_cross_policy_hybrids(
    baseline: GeographyTrial,
    hybrids: tuple[CandidateEvaluation, ...],
) -> _pd.DataFrame:
    """Return the two bounded cross-policy votes relative to baseline."""

    rows = []
    for evaluation in hybrids:
        fold_delta = (
            evaluation.fold_metrics["accuracy"]
            - baseline.blend.fold_metrics["accuracy"]
        )
        rows.append(
            {
                "hybrid": evaluation.model_name,
                "mean_accuracy": float(
                    evaluation.metric_summary.loc["accuracy", "mean"]
                ),
                "accuracy_change": float(fold_delta.mean()),
                "fold_wins": int(fold_delta.gt(0).sum()),
                "worst_fold_change": float(fold_delta.min()),
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
            }
        )
    return _pd.DataFrame(rows).set_index("hybrid").sort_values(
        "mean_accuracy",
        ascending=False,
    )


def make_lga_grouped_partition(
    partitioned_data: PartitionedData,
) -> tuple[PartitionedData, _PredefinedSplit, _pd.DataFrame]:
    """Return five LGA-disjoint sensitivity folds over development rows only."""

    groups = partitioned_data.X_development["lga"].astype("string").fillna(
        "__missing__"
    )
    grouped_splitter = _StratifiedGroupKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=GROUPED_SENSITIVITY_SEED,
    )
    assignments = _np.zeros(len(groups), dtype="int8")
    rows = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        grouped_splitter.split(
            partitioned_data.X_development,
            partitioned_data.y_development,
            groups,
        ),
        start=1,
    ):
        training_groups = set(groups.iloc[training_positions])
        validation_groups = set(groups.iloc[validation_positions])
        if training_groups & validation_groups:
            raise ValueError("An LGA appears in both sides of a grouped fold.")
        assignments[validation_positions] = fold_number
        target = partitioned_data.y_development.iloc[validation_positions]
        rows.append(
            {
                "validation_fold": fold_number,
                "rows": len(validation_positions),
                "lgas": len(validation_groups),
                "functional_share": target.eq("functional").mean(),
                "repair_share": target.eq("functional needs repair").mean(),
                "non_functional_share": target.eq("non functional").mean(),
            }
        )
    if not assignments.all():
        raise ValueError("Grouped sensitivity folds do not cover development rows.")

    validation_folds = _pd.Series(assignments, name="validation_fold")
    fingerprint = _fingerprint_folds(
        partitioned_data.development_ids,
        validation_folds,
    )
    grouped_partition = _replace(
        partitioned_data,
        validation_folds=validation_folds,
        cross_validation_fingerprint=fingerprint,
    )
    cross_validation = _PredefinedSplit(test_fold=assignments - 1)
    return (
        grouped_partition,
        cross_validation,
        _pd.DataFrame(rows).set_index("validation_fold"),
    )


def _fingerprint_folds(ids: _pd.Series, folds: _pd.Series) -> str:
    digest = _hashlib.sha256()
    memberships = sorted(
        zip((str(value) for value in ids), folds, strict=True),
        key=lambda item: item[0],
    )
    for identifier, fold_number in memberships:
        digest.update(f"{identifier}:{fold_number}\n".encode("utf-8"))
    return digest.hexdigest()

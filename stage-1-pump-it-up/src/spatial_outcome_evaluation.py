"""Evaluate cross-fitted local spatial outcome features."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from dataclasses import replace as _replace

import numpy as _np
import pandas as _pd

from data_partitioning import PartitionedData
from gpu_model_evaluation import CLASS_LABELS
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_evaluation import evaluate_random_forest
from model_evaluation import evaluate_weighted_soft_vote
from spatial_outcome_features import SELECTION_GATE_ACCURACY
from spatial_outcome_features import SELECTION_GATE_REPAIR_RECALL
from spatial_outcome_features import SELECTION_GATE_WORST_FOLD
from spatial_outcome_features import SPATIAL_OUTCOME_POLICY
from spatial_outcome_features import fit_spatial_outcome_model
from spatial_outcome_features import make_spatial_outcome_preprocessor
from spatial_outcome_features import spatial_outcome_values


XGBOOST_WEIGHT = 0.55
RANDOM_FOREST_WEIGHT = 0.45
SPATIAL_VOTER_WEIGHT = 0.15


@_dataclass(frozen=True)
class SpatialOutcomeTrial:
    """Feature-refit and direct-voter evidence for local spatial outcomes."""

    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    blend: CandidateEvaluation
    spatial_voter: CandidateEvaluation
    spatial_voter_blend: CandidateEvaluation
    transformed_features_fold_1: int


def evaluate_spatial_outcomes(
    partitioned_data: PartitionedData,
    cross_validation: object,
    accepted: CandidateEvaluation,
) -> SpatialOutcomeTrial:
    """Evaluate one fixed cross-fitted spatial feature and voter policy."""

    _validate_accepted(partitioned_data, accepted)
    xgboost_spec = _replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [spatial k10 s20]",
        feature_policy=SPATIAL_OUTCOME_POLICY.key,
    )
    xgboost = evaluate_gpu_candidate(
        xgboost_spec,
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_spatial_outcome_preprocessor,
    )
    random_forest = evaluate_random_forest(
        partitioned_data,
        cross_validation,
        preprocessor_factory=make_spatial_outcome_preprocessor,
        model_name="Random Forest [spatial k10 s20]",
    )
    blend = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        xgboost,
        random_forest,
        weights=[XGBOOST_WEIGHT, RANDOM_FOREST_WEIGHT],
        model_name="55% XGBoost + 45% Random Forest [spatial k10 s20]",
    )
    spatial_voter = evaluate_spatial_voter(
        partitioned_data,
        cross_validation,
    )
    spatial_voter_blend = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        accepted,
        spatial_voter,
        weights=[1.0 - SPATIAL_VOTER_WEIGHT, SPATIAL_VOTER_WEIGHT],
        model_name="85% accepted ensemble + 15% spatial voter",
    )

    first_training_positions, _ = next(cross_validation.split())
    preprocessor = make_spatial_outcome_preprocessor()
    preprocessor.fit(
        partitioned_data.X_development.iloc[first_training_positions],
        partitioned_data.y_development.iloc[first_training_positions],
    )
    return SpatialOutcomeTrial(
        xgboost=xgboost,
        random_forest=random_forest,
        blend=blend,
        spatial_voter=spatial_voter,
        spatial_voter_blend=spatial_voter_blend,
        transformed_features_fold_1=len(preprocessor.get_feature_names_out()),
    )


def evaluate_spatial_voter(
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> CandidateEvaluation:
    """Create outer-OOF probabilities from training-only spatial neighbours."""

    probabilities = _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostic_rows = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        model = fit_spatial_outcome_model(X_training, y_training)
        values = spatial_outcome_values(X_validation, model)
        probabilities[validation_positions] = values[:, : len(CLASS_LABELS)]
        radius = values[:, -1]
        diagnostic_rows.append(
            {
                "validation_fold": fold_number,
                "reference_rows": len(model.encoded_target),
                "invalid_coordinate_rows": int(_np.isnan(radius).sum()),
                "median_radius_km": float(_np.nanmedian(radius)),
            }
        )
    return build_candidate_evaluation(
        model_name="Ten-neighbour spatial class-rate voter",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=diagnostic_rows,
    )


def summarise_spatial_outcomes(
    accepted: CandidateEvaluation,
    trial: SpatialOutcomeTrial,
) -> _pd.DataFrame:
    """Compare feature refits and direct spatial probabilities with baseline."""

    baseline_accuracy = float(accepted.metric_summary.loc["accuracy", "mean"])
    baseline_repair = float(
        accepted.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    candidates = {
        "accepted_baseline": accepted,
        "spatial_feature_vote": trial.blend,
        "spatial_voter": trial.spatial_voter,
        "accepted_plus_spatial_voter": trial.spatial_voter_blend,
    }
    rows = []
    for key, candidate in candidates.items():
        _validate_candidate_alignment(accepted, candidate)
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
                "candidate": key,
                "model_name": candidate.model_name,
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
                "passes_gate": key != "accepted_baseline" and (
                    accuracy_change >= SELECTION_GATE_ACCURACY
                    and int(fold_change.gt(0).sum()) >= 3
                    and float(fold_change.min()) >= SELECTION_GATE_WORST_FOLD
                    and repair_change >= SELECTION_GATE_REPAIR_RECALL
                ),
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("candidate")
        .sort_values(["mean_accuracy", "candidate"], ascending=[False, True])
    )


def _validate_accepted(
    partitioned_data: PartitionedData,
    accepted: CandidateEvaluation,
) -> None:
    if (
        accepted.cross_validation_fingerprint
        != partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError("Accepted ensemble uses different frozen folds.")


def _validate_candidate_alignment(
    accepted: CandidateEvaluation,
    candidate: CandidateEvaluation,
) -> None:
    if candidate.cross_validation_fingerprint != accepted.cross_validation_fingerprint:
        raise ValueError("Spatial candidate uses different frozen folds.")
    if not candidate.out_of_fold_probabilities.index.equals(
        accepted.out_of_fold_probabilities.index
    ):
        raise ValueError("Spatial candidate rows do not align with baseline.")

"""Evaluate hard component consensus with a soft-vote disagreement fallback."""

from __future__ import annotations

import numpy as _np

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from gpu_model_evaluation import CLASS_LABELS
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation


def consensus_majority_probabilities(
    component_probabilities: tuple[_np.ndarray, _np.ndarray, _np.ndarray],
    fallback_probabilities: _np.ndarray,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Return one-hot majority decisions and all-disagree row indicators."""

    components = tuple(
        _validate_probability_matrix(values, "component")
        for values in component_probabilities
    )
    fallback = _validate_probability_matrix(
        fallback_probabilities,
        "fallback",
    )
    if any(values.shape != fallback.shape for values in components):
        raise ValueError("Consensus probability matrices must have equal shapes.")
    predictions = _np.column_stack(
        [values.argmax(axis=1) for values in components]
    )
    fallback_predictions = fallback.argmax(axis=1)
    selected = fallback_predictions.copy()
    all_disagree = _np.zeros(len(fallback), dtype=bool)
    for row_position, votes in enumerate(predictions):
        counts = _np.bincount(votes, minlength=len(CLASS_LABELS))
        if counts.max() >= 2:
            selected[row_position] = int(counts.argmax())
        else:
            all_disagree[row_position] = True
    probabilities = _np.zeros_like(fallback)
    probabilities[_np.arange(len(selected)), selected] = 1.0
    return probabilities, all_disagree


def evaluate_consensus_majority_vote(
    partitioned_data: PartitionedData,
    cross_validation: object,
    xgboost: CandidateEvaluation,
    random_forest: CandidateEvaluation,
    identity_catboost: CandidateEvaluation,
    fallback_soft_vote: CandidateEvaluation,
) -> CandidateEvaluation:
    """Evaluate hard majority with the promoted soft vote for three-way ties."""

    evaluations = (
        xgboost,
        random_forest,
        identity_catboost,
        fallback_soft_vote,
    )
    for evaluation in evaluations:
        if (
            evaluation.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Consensus components use different frozen folds.")
    probability_values, all_disagree = consensus_majority_probabilities(
        tuple(
            evaluation.out_of_fold_probabilities.to_numpy(dtype="float64")
            for evaluation in evaluations[:3]
        ),
        fallback_soft_vote.out_of_fold_probabilities.to_numpy(dtype="float64"),
    )
    diagnostics = []
    for fold_number, (_, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        fold_all_disagree = all_disagree[validation_positions]
        diagnostics.append(
            {
                "validation_fold": fold_number,
                "majority_rows": int((~fold_all_disagree).sum()),
                "all_disagree_rows": int(fold_all_disagree.sum()),
                "all_disagree_share": float(fold_all_disagree.mean()),
            }
        )
    if len(diagnostics) != CROSS_VALIDATION_FOLDS:
        raise ValueError("Consensus evaluation did not cover all frozen folds.")
    return build_candidate_evaluation(
        model_name="Hard component majority with promoted soft tie-break",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )


def _validate_probability_matrix(values: _np.ndarray, name: str) -> _np.ndarray:
    probabilities = _np.asarray(values, dtype="float64")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_LABELS):
        raise ValueError(f"{name.capitalize()} probabilities have wrong shape.")
    if not _np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError(
            f"{name.capitalize()} probabilities must be finite and non-negative."
        )
    if not _np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError(f"{name.capitalize()} probabilities must sum to one.")
    return probabilities

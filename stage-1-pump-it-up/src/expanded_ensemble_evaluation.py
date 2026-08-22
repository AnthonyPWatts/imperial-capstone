"""Evaluate a bounded set of additional voters around the accepted ensemble."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd
from sklearn.metrics import log_loss as _log_loss

from data_partitioning import PartitionedData
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_weighted_soft_vote


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)

ACCEPTED_XGBOOST_WEIGHT = 0.55
ACCEPTED_RANDOM_FOREST_WEIGHT = 0.45

SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_FOLD_WINS = 3
SELECTION_GATE_WORST_FOLD = -0.0025
SELECTION_GATE_REPAIR_RECALL = -0.02


@_dataclass(frozen=True)
class ExpandedEnsembleSpec:
    """One fixed third-voter contribution to the accepted two-voter recipe."""

    key: str
    label: str
    voter_name: str
    voter_weight: float

    @property
    def xgboost_weight(self) -> float:
        """Preserve the accepted 55:45 ratio in the remaining probability mass."""

        return (1.0 - self.voter_weight) * ACCEPTED_XGBOOST_WEIGHT

    @property
    def random_forest_weight(self) -> float:
        """Preserve the accepted 55:45 ratio in the remaining probability mass."""

        return (1.0 - self.voter_weight) * ACCEPTED_RANDOM_FOREST_WEIGHT


EXPANDED_ENSEMBLE_SPECS = (
    ExpandedEnsembleSpec(
        key="catboost_d8_10",
        label="10% native-categorical CatBoost",
        voter_name="CatBoost d8 [current features]",
        voter_weight=0.10,
    ),
    ExpandedEnsembleSpec(
        key="lightgbm_bagged_63_15",
        label="15% bagged LightGBM",
        voter_name="LightGBM leaves 63 bagged [current one-hot]",
        voter_weight=0.15,
    ),
    ExpandedEnsembleSpec(
        key="mlp_relu_128_64_05",
        label="5% two-layer MLP",
        voter_name="MLP ReLU 128-64 [scaled current one-hot]",
        voter_weight=0.05,
    ),
)


@_dataclass(frozen=True)
class ExpandedEnsembleScreen:
    """No-refit evaluations and diagnostics for the bounded voter screen."""

    evaluations: dict[str, CandidateEvaluation]
    candidate_summary: _pd.DataFrame
    component_summary: _pd.DataFrame


def evaluate_expanded_ensembles(
    partitioned_data: PartitionedData,
    cross_validation: object,
    *,
    accepted: CandidateEvaluation,
    xgboost: CandidateEvaluation,
    random_forest: CandidateEvaluation,
    voters: _Mapping[str, CandidateEvaluation],
    specs: tuple[ExpandedEnsembleSpec, ...] = EXPANDED_ENSEMBLE_SPECS,
) -> ExpandedEnsembleScreen:
    """Evaluate fixed third voters without refitting or reopening the local test."""

    _validate_specifications(specs)
    _validate_accepted_components(accepted, xgboost, random_forest)
    missing = {spec.voter_name for spec in specs} - set(voters)
    if missing:
        raise ValueError(f"Expanded-ensemble voters are missing: {sorted(missing)!r}.")

    evaluations = {}
    for spec in specs:
        voter = voters[spec.voter_name]
        evaluation = evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            xgboost,
            random_forest,
            voter,
            weights=(
                spec.xgboost_weight,
                spec.random_forest_weight,
                spec.voter_weight,
            ),
            model_name=spec.label,
        )
        evaluations[spec.key] = evaluation

    return ExpandedEnsembleScreen(
        evaluations=evaluations,
        candidate_summary=_summarise_candidates(
            partitioned_data,
            accepted,
            evaluations,
            specs,
        ),
        component_summary=_summarise_components(
            partitioned_data,
            accepted,
            voters,
            specs,
        ),
    )


def probability_quality(
    target: _pd.Series,
    probabilities: _pd.DataFrame | _np.ndarray,
) -> tuple[float, float]:
    """Return multiclass log loss and the mean summed squared probability error."""

    values = _np.asarray(probabilities, dtype="float64")
    if values.shape != (len(target), len(CLASS_LABELS)):
        raise ValueError("Probability rows or class columns do not match the target.")
    if not _np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite and between zero and one.")
    if not _np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("Probability rows must sum to one.")
    values = values / values.sum(axis=1, keepdims=True)

    one_hot = _np.column_stack(
        [target.eq(label).to_numpy(dtype="float64") for label in CLASS_LABELS]
    )
    return (
        float(_log_loss(target, values, labels=list(CLASS_LABELS))),
        float(_np.mean(_np.square(values - one_hot).sum(axis=1))),
    )


def _validate_specifications(
    specs: tuple[ExpandedEnsembleSpec, ...],
) -> None:
    if not specs:
        raise ValueError("At least one expanded-ensemble specification is required.")
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("Expanded-ensemble specification keys must be unique.")
    if len({spec.voter_name for spec in specs}) != len(specs):
        raise ValueError("Each expanded-ensemble specification must add a distinct voter.")
    for spec in specs:
        if not 0 < spec.voter_weight < 1:
            raise ValueError("Each added-voter weight must be strictly between zero and one.")


def _validate_accepted_components(
    accepted: CandidateEvaluation,
    xgboost: CandidateEvaluation,
    random_forest: CandidateEvaluation,
) -> None:
    expected = (
        ACCEPTED_XGBOOST_WEIGHT
        * xgboost.out_of_fold_probabilities.to_numpy()
        + ACCEPTED_RANDOM_FOREST_WEIGHT
        * random_forest.out_of_fold_probabilities.to_numpy()
    )
    if not _np.allclose(
        accepted.out_of_fold_probabilities.to_numpy(),
        expected,
    ):
        raise ValueError("Accepted probabilities do not match the declared 55:45 recipe.")


def _summarise_candidates(
    partitioned_data: PartitionedData,
    accepted: CandidateEvaluation,
    evaluations: dict[str, CandidateEvaluation],
    specs: tuple[ExpandedEnsembleSpec, ...],
) -> _pd.DataFrame:
    accepted_accuracy = accepted.fold_metrics["accuracy"]
    accepted_repair = float(
        accepted.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    accepted_predictions = accepted.out_of_fold_probabilities.idxmax(axis=1)
    accepted_log_loss, accepted_brier = probability_quality(
        partitioned_data.y_development,
        accepted.out_of_fold_probabilities,
    )
    by_key = {spec.key: spec for spec in specs}
    rows = []
    for key, evaluation in evaluations.items():
        spec = by_key[key]
        fold_change = evaluation.fold_metrics["accuracy"] - accepted_accuracy
        repair_recall = float(
            evaluation.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        repair_change = repair_recall - accepted_repair
        accuracy_change = float(fold_change.mean())
        log_loss, brier = probability_quality(
            partitioned_data.y_development,
            evaluation.out_of_fold_probabilities,
        )
        predictions = evaluation.out_of_fold_probabilities.idxmax(axis=1)
        rows.append(
            {
                "candidate": key,
                "label": spec.label,
                "added_voter": spec.voter_name,
                "xgboost_weight": spec.xgboost_weight,
                "random_forest_weight": spec.random_forest_weight,
                "added_voter_weight": spec.voter_weight,
                "mean_accuracy": float(
                    evaluation.metric_summary.loc["accuracy", "mean"]
                ),
                "accuracy_change": accuracy_change,
                "fold_wins": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall": repair_recall,
                "repair_recall_change": repair_change,
                "non_functional_recall": float(
                    evaluation.metric_summary.loc[
                        "recall: non functional",
                        "mean",
                    ]
                ),
                "baseline_disagreement": float(
                    predictions.ne(accepted_predictions).mean()
                ),
                "log_loss": log_loss,
                "log_loss_change": log_loss - accepted_log_loss,
                "brier_score": brier,
                "brier_score_change": brier - accepted_brier,
                "passes_gate": (
                    accuracy_change >= SELECTION_GATE_ACCURACY
                    and int(fold_change.gt(0).sum()) >= SELECTION_GATE_FOLD_WINS
                    and float(fold_change.min()) >= SELECTION_GATE_WORST_FOLD
                    and repair_change >= SELECTION_GATE_REPAIR_RECALL
                ),
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("candidate")
        .sort_values("mean_accuracy", ascending=False)
    )


def _summarise_components(
    partitioned_data: PartitionedData,
    accepted: CandidateEvaluation,
    voters: _Mapping[str, CandidateEvaluation],
    specs: tuple[ExpandedEnsembleSpec, ...],
) -> _pd.DataFrame:
    target = partitioned_data.y_development
    accepted_predictions = accepted.out_of_fold_probabilities.idxmax(axis=1)
    accepted_correct = accepted_predictions.eq(target)
    accepted_values = accepted.out_of_fold_probabilities.to_numpy().ravel()
    rows = []
    for spec in specs:
        voter = voters[spec.voter_name]
        predictions = voter.out_of_fold_probabilities.idxmax(axis=1)
        correct = predictions.eq(target)
        log_loss, brier = probability_quality(
            target,
            voter.out_of_fold_probabilities,
        )
        rows.append(
            {
                "voter": spec.voter_name,
                "standalone_accuracy": float(correct.mean()),
                "repair_recall": float(
                    voter.metric_summary.loc[
                        "recall: functional needs repair",
                        "mean",
                    ]
                ),
                "baseline_disagreement": float(
                    predictions.ne(accepted_predictions).mean()
                ),
                "voter_only_correct": float((correct & ~accepted_correct).mean()),
                "baseline_only_correct": float((accepted_correct & ~correct).mean()),
                "probability_correlation": float(
                    _np.corrcoef(
                        accepted_values,
                        voter.out_of_fold_probabilities.to_numpy().ravel(),
                    )[0, 1]
                ),
                "log_loss": log_loss,
                "brier_score": brier,
            }
        )
    return _pd.DataFrame(rows).set_index("voter")

"""Evaluate regional probability layers and partially pooled local experts."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Callable as _Callable

import numpy as _np
import pandas as _pd
from sklearn.metrics import log_loss as _log_loss

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_evaluation import make_random_forest_pipeline


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)

REGION_FEATURE = "region"
REGIONAL_MINIMUM_TRAINING_ROWS = 500
REGIONAL_MINIMUM_CLASS_ROWS = 20
REGIONAL_EXPERT_WEIGHT = 0.20
PRIOR_SMOOTHING_LEVELS = (0.0, 10_000.0)
BOOTSTRAP_REPLICATIONS = 2_000
BOOTSTRAP_SEED = 20260822

SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_FOLD_WINS = 3
SELECTION_GATE_WORST_FOLD = -0.0025
SELECTION_GATE_REPAIR_RECALL = -0.02

_MISSING_REGION = "__missing_region__"


@_dataclass(frozen=True)
class RegionalSpecialisationScreen:
    """Fold-safe regional candidates, routing evidence and comparisons."""

    hard_routed_expert: CandidateEvaluation
    partially_pooled_expert: CandidateEvaluation
    prior_layers: dict[str, CandidateEvaluation]
    routing_summary: _pd.DataFrame
    candidate_summary: _pd.DataFrame
    regional_summary: _pd.DataFrame


def evaluate_regional_specialisation(
    partitioned_data: PartitionedData,
    cross_validation: object,
    accepted: CandidateEvaluation,
    *,
    pipeline_factory: _Callable[[], object] = make_random_forest_pipeline,
    minimum_training_rows: int = REGIONAL_MINIMUM_TRAINING_ROWS,
    minimum_class_rows: int = REGIONAL_MINIMUM_CLASS_ROWS,
    regional_weight: float = REGIONAL_EXPERT_WEIGHT,
    prior_smoothing_levels: tuple[float, ...] = PRIOR_SMOOTHING_LEVELS,
    bootstrap_replications: int = BOOTSTRAP_REPLICATIONS,
) -> RegionalSpecialisationScreen:
    """Test hard routing, partial pooling and label-prior regional layers."""

    _validate_arguments(
        partitioned_data,
        accepted,
        minimum_training_rows=minimum_training_rows,
        minimum_class_rows=minimum_class_rows,
        regional_weight=regional_weight,
        prior_smoothing_levels=prior_smoothing_levels,
    )
    accepted_values = accepted.out_of_fold_probabilities.to_numpy()
    routed_values = accepted_values.copy()
    expert_used = _np.zeros(len(partitioned_data.y_development), dtype=bool)
    routing_rows = []
    fold_diagnostics = []

    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        training_regions = _normalise_regions(X_training[REGION_FEATURE])
        validation_regions = _normalise_regions(X_validation[REGION_FEATURE])
        fold_expert_rows = 0
        eligible_regions = 0

        for region in sorted(_pd.unique(validation_regions)):
            training_mask = training_regions.eq(region).to_numpy()
            validation_mask = validation_regions.eq(region).to_numpy()
            local_validation_positions = _np.flatnonzero(validation_mask)
            target = y_training.iloc[_np.flatnonzero(training_mask)]
            class_counts = target.value_counts().reindex(CLASS_LABELS, fill_value=0)
            eligible = (
                len(target) >= minimum_training_rows
                and int(class_counts.min()) >= minimum_class_rows
            )
            reason = "eligible"
            if len(target) < minimum_training_rows:
                reason = "too_few_training_rows"
            elif int(class_counts.min()) < minimum_class_rows:
                reason = "too_few_rows_in_at_least_one_class"

            if eligible:
                model = pipeline_factory()
                model.fit(
                    X_training.iloc[_np.flatnonzero(training_mask)],
                    target,
                )
                probabilities = _align_probabilities(
                    model,
                    X_validation.iloc[local_validation_positions],
                )
                development_positions = validation_positions[
                    local_validation_positions
                ]
                routed_values[development_positions] = probabilities
                expert_used[development_positions] = True
                fold_expert_rows += len(local_validation_positions)
                eligible_regions += 1

            routing_rows.append(
                {
                    "validation_fold": fold_number,
                    "region": region,
                    "training_rows": len(target),
                    **{
                        f"training_{label}_rows": int(class_counts[label])
                        for label in CLASS_LABELS
                    },
                    "validation_rows": int(validation_mask.sum()),
                    "eligible": eligible,
                    "routing": "regional_expert" if eligible else "global_fallback",
                    "reason": reason,
                }
            )

        fold_diagnostics.append(
            {
                "validation_fold": fold_number,
                "eligible_regions": eligible_regions,
                "expert_rows": fold_expert_rows,
                "fallback_rows": len(validation_positions) - fold_expert_rows,
                "expert_coverage": fold_expert_rows / len(validation_positions),
            }
        )

    _validate_probabilities(routed_values)
    hard_routed = build_candidate_evaluation(
        model_name="Hard-routed regional Random Forest experts",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=routed_values,
        diagnostic_rows=fold_diagnostics,
    )
    pooled_values = (
        (1.0 - regional_weight) * accepted_values
        + regional_weight * routed_values
    )
    _validate_probabilities(pooled_values)
    pooled_diagnostics = [
        {
            **row,
            "global_weight": 1.0 - regional_weight,
            "regional_weight": regional_weight,
        }
        for row in fold_diagnostics
    ]
    partially_pooled = build_candidate_evaluation(
        model_name=(
            f"{1.0 - regional_weight:.0%} global + "
            f"{regional_weight:.0%} regional Random Forest"
        ),
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=pooled_values,
        diagnostic_rows=pooled_diagnostics,
    )
    prior_layers = {
        _prior_key(alpha): evaluate_regional_prior_layer(
            partitioned_data,
            cross_validation,
            accepted,
            smoothing=alpha,
        )
        for alpha in prior_smoothing_levels
    }
    evaluations = {
        "hard_routed_expert": hard_routed,
        "partial_pooling_20": partially_pooled,
        **prior_layers,
    }
    routing_summary = _pd.DataFrame(routing_rows).set_index(
        ["validation_fold", "region"]
    )
    return RegionalSpecialisationScreen(
        hard_routed_expert=hard_routed,
        partially_pooled_expert=partially_pooled,
        prior_layers=prior_layers,
        routing_summary=routing_summary,
        candidate_summary=_summarise_candidates(
            partitioned_data,
            accepted,
            evaluations,
            selection_key="partial_pooling_20",
            bootstrap_replications=bootstrap_replications,
        ),
        regional_summary=_summarise_regions(
            partitioned_data,
            accepted,
            evaluations,
            expert_used,
        ),
    )


def evaluate_regional_prior_layer(
    partitioned_data: PartitionedData,
    cross_validation: object,
    accepted: CandidateEvaluation,
    *,
    smoothing: float,
) -> CandidateEvaluation:
    """Apply outer-training regional-to-national prior ratios to global OOF rows."""

    if not _np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("Regional-prior smoothing must be finite and non-negative.")
    adjusted = accepted.out_of_fold_probabilities.to_numpy().copy()
    diagnostics = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        y_training = partitioned_data.y_development.iloc[training_positions]
        training_regions = _normalise_regions(
            partitioned_data.X_development.iloc[training_positions][REGION_FEATURE]
        )
        validation_regions = _normalise_regions(
            partitioned_data.X_development.iloc[validation_positions][REGION_FEATURE]
        )
        national_counts = y_training.value_counts().reindex(CLASS_LABELS, fill_value=0)
        national_prior = national_counts.to_numpy(dtype="float64") / len(y_training)
        adjusted_rows = 0
        for region in _pd.unique(validation_regions):
            training_mask = training_regions.eq(region).to_numpy()
            if not training_mask.any():
                continue
            regional_target = y_training.iloc[_np.flatnonzero(training_mask)]
            regional_counts = regional_target.value_counts().reindex(
                CLASS_LABELS,
                fill_value=0,
            ).to_numpy(dtype="float64")
            regional_prior = (
                regional_counts + smoothing * national_prior
            ) / (len(regional_target) + smoothing)
            ratios = regional_prior / national_prior
            local_positions = _np.flatnonzero(validation_regions.eq(region).to_numpy())
            development_positions = validation_positions[local_positions]
            values = adjusted[development_positions] * ratios
            row_sums = values.sum(axis=1, keepdims=True)
            valid = row_sums[:, 0] > 0
            values[valid] /= row_sums[valid]
            values[~valid] = accepted.out_of_fold_probabilities.to_numpy()[
                development_positions[~valid]
            ]
            adjusted[development_positions] = values
            adjusted_rows += len(local_positions)
        diagnostics.append(
            {
                "validation_fold": fold_number,
                "smoothing": smoothing,
                "adjusted_rows": adjusted_rows,
            }
        )
    _validate_probabilities(adjusted)
    return build_candidate_evaluation(
        model_name=(
            "Regional prior-ratio layer "
            f"(smoothing={smoothing:g})"
        ),
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=adjusted,
        diagnostic_rows=diagnostics,
    )


def _normalise_regions(values: _pd.Series) -> _pd.Series:
    return values.astype("string").fillna(_MISSING_REGION).str.strip()


def _align_probabilities(model: object, X: _pd.DataFrame) -> _np.ndarray:
    values = _np.asarray(model.predict_proba(X), dtype="float64")
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = getattr(model.named_steps["classifier"], "classes_", None)
    if classes is None:
        raise ValueError("A regional expert did not expose fitted class labels.")
    classes = list(classes)
    unknown = set(classes) - set(CLASS_LABELS)
    if unknown:
        raise ValueError(f"Regional expert returned unknown classes: {sorted(unknown)!r}.")
    if values.shape != (len(X), len(classes)):
        raise ValueError("Regional expert probability shape does not match its classes.")
    aligned = _np.zeros((len(X), len(CLASS_LABELS)), dtype="float64")
    for source_position, label in enumerate(classes):
        aligned[:, CLASS_LABELS.index(label)] = values[:, source_position]
    _validate_probabilities(aligned)
    return aligned


def _validate_probabilities(values: _np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != len(CLASS_LABELS):
        raise ValueError("Regional probabilities have the wrong class shape.")
    if not _np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Regional probabilities must be finite and between zero and one.")
    if not _np.allclose(values.sum(axis=1), 1.0):
        raise ValueError("Regional probability rows must sum to one.")


def _validate_arguments(
    partitioned_data: PartitionedData,
    accepted: CandidateEvaluation,
    *,
    minimum_training_rows: int,
    minimum_class_rows: int,
    regional_weight: float,
    prior_smoothing_levels: tuple[float, ...],
) -> None:
    if REGION_FEATURE not in partitioned_data.X_development:
        raise ValueError(f"Development predictors do not contain {REGION_FEATURE!r}.")
    if minimum_training_rows <= 0 or minimum_class_rows <= 0:
        raise ValueError("Regional eligibility thresholds must be positive.")
    if not 0 < regional_weight < 1:
        raise ValueError("Regional weight must be strictly between zero and one.")
    if len(prior_smoothing_levels) != len(set(prior_smoothing_levels)):
        raise ValueError("Regional-prior smoothing levels must be unique.")
    if (
        accepted.cross_validation_fingerprint
        != partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError("Accepted probabilities use a different fold design.")
    expected_columns = _pd.Index(CLASS_LABELS, name="predicted_class")
    if not accepted.out_of_fold_probabilities.columns.equals(expected_columns):
        raise ValueError("Accepted probabilities have unexpected class columns.")
    if not accepted.out_of_fold_probabilities.index.equals(
        partitioned_data.y_development.index
    ):
        raise ValueError("Accepted probabilities are not aligned to development rows.")


def _summarise_candidates(
    partitioned_data: PartitionedData,
    accepted: CandidateEvaluation,
    evaluations: dict[str, CandidateEvaluation],
    *,
    selection_key: str,
    bootstrap_replications: int,
) -> _pd.DataFrame:
    target = partitioned_data.y_development
    accepted_accuracy = accepted.fold_metrics["accuracy"]
    accepted_predictions = accepted.out_of_fold_probabilities.idxmax(axis=1)
    accepted_repair = float(
        accepted.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    accepted_log_loss, accepted_brier = _probability_quality(
        target,
        accepted.out_of_fold_probabilities,
    )
    rows = []
    for key, evaluation in evaluations.items():
        predictions = evaluation.out_of_fold_probabilities.idxmax(axis=1)
        fold_change = evaluation.fold_metrics["accuracy"] - accepted_accuracy
        accuracy_change = float(fold_change.mean())
        repair_recall = float(
            evaluation.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        repair_change = repair_recall - accepted_repair
        log_loss, brier = _probability_quality(
            target,
            evaluation.out_of_fold_probabilities,
        )
        lower, upper = _paired_bootstrap_interval(
            target,
            accepted_predictions,
            predictions,
            replications=bootstrap_replications,
        )
        selection_eligible = key == selection_key
        rows.append(
            {
                "candidate": key,
                "label": evaluation.model_name,
                "selection_eligible": selection_eligible,
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
                "paired_bootstrap_accuracy_change_lower": lower,
                "paired_bootstrap_accuracy_change_upper": upper,
                "passes_gate": (
                    selection_eligible
                    and accuracy_change >= SELECTION_GATE_ACCURACY
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


def _summarise_regions(
    partitioned_data: PartitionedData,
    accepted: CandidateEvaluation,
    evaluations: dict[str, CandidateEvaluation],
    expert_used: _np.ndarray,
) -> _pd.DataFrame:
    regions = _normalise_regions(partitioned_data.X_development[REGION_FEATURE])
    target = partitioned_data.y_development
    frames = []
    all_evaluations = {"accepted": accepted, **evaluations}
    for key, evaluation in all_evaluations.items():
        probabilities = evaluation.out_of_fold_probabilities
        predictions = probabilities.idxmax(axis=1)
        frame = _pd.DataFrame(
            {
                "region": regions,
                "target": target,
                "prediction": predictions,
                "repair_probability": probabilities[
                    "functional needs repair"
                ].to_numpy(),
                "expert_used": expert_used,
            }
        )
        rows = []
        for region, group in frame.groupby("region", sort=True):
            repair_mask = group["target"].eq("functional needs repair")
            repair_rows = int(repair_mask.sum())
            rows.append(
                {
                    "model": key,
                    "region": region,
                    "rows": len(group),
                    "observed_repair_share": float(repair_mask.mean()),
                    "mean_repair_probability": float(
                        group["repair_probability"].mean()
                    ),
                    "accuracy": float(
                        group["prediction"].eq(group["target"]).mean()
                    ),
                    "repair_recall": (
                        float(
                            group.loc[repair_mask, "prediction"]
                            .eq("functional needs repair")
                            .mean()
                        )
                        if repair_rows
                        else _np.nan
                    ),
                    "expert_coverage": float(group["expert_used"].mean()),
                }
            )
        frames.append(_pd.DataFrame(rows))
    return _pd.concat(frames, ignore_index=True).set_index(["model", "region"])


def _probability_quality(
    target: _pd.Series,
    probabilities: _pd.DataFrame,
) -> tuple[float, float]:
    values = probabilities.to_numpy(dtype="float64")
    values = values / values.sum(axis=1, keepdims=True)
    one_hot = _np.column_stack(
        [target.eq(label).to_numpy(dtype="float64") for label in CLASS_LABELS]
    )
    return (
        float(_log_loss(target, values, labels=list(CLASS_LABELS))),
        float(_np.mean(_np.square(values - one_hot).sum(axis=1))),
    )


def _paired_bootstrap_interval(
    target: _pd.Series,
    accepted_predictions: _pd.Series,
    candidate_predictions: _pd.Series,
    *,
    replications: int,
) -> tuple[float, float]:
    if replications <= 0:
        return _np.nan, _np.nan
    target_values = target.to_numpy()
    difference = (
        candidate_predictions.to_numpy() == target_values
    ).astype("float64") - (
        accepted_predictions.to_numpy() == target_values
    ).astype("float64")
    generator = _np.random.default_rng(BOOTSTRAP_SEED)
    estimates = _np.empty(replications, dtype="float64")
    for position in range(replications):
        sample = generator.integers(0, len(difference), size=len(difference))
        estimates[position] = difference[sample].mean()
    lower, upper = _np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper)


def _prior_key(smoothing: float) -> str:
    return f"prior_smoothing_{smoothing:g}".replace(".", "_")

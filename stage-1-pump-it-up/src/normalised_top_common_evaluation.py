"""Locked comparison of raw and normalised top-common identities."""

from __future__ import annotations

import numpy as _np
import pandas as _pd

from model_evaluation import CandidateEvaluation


MEAN_ACCURACY_GAIN_GATE = 0.001
FOLD_WIN_GATE = 3
WORST_FOLD_DELTA_GATE = -0.0025
REPAIR_RECALL_DELTA_GATE = -0.02
REPAIR_METRIC = "recall: functional needs repair"


def summarise_normalised_comparison(
    raw: CandidateEvaluation,
    normalised: CandidateEvaluation,
) -> tuple[_pd.DataFrame, _pd.DataFrame]:
    """Return the predeclared gate and paired fold evidence."""

    if raw.cross_validation_fingerprint != normalised.cross_validation_fingerprint:
        raise ValueError("Representations use different cross-validation folds.")
    if not raw.fold_metrics.index.equals(normalised.fold_metrics.index):
        raise ValueError("Representation fold labels do not align.")
    if len(raw.fold_metrics) != 5:
        raise ValueError("Normalised identity comparison requires five folds.")
    required = {"accuracy", REPAIR_METRIC}
    if not required.issubset(raw.fold_metrics) or not required.issubset(
        normalised.fold_metrics
    ):
        raise ValueError("Representation metrics are incomplete.")
    values = _np.concatenate(
        [
            raw.fold_metrics[list(required)].to_numpy(dtype="float64").ravel(),
            normalised.fold_metrics[list(required)]
            .to_numpy(dtype="float64")
            .ravel(),
        ]
    )
    if not _np.isfinite(values).all():
        raise ValueError("Representation metrics must be finite.")

    accuracy_delta = normalised.fold_metrics["accuracy"] - raw.fold_metrics["accuracy"]
    repair_delta = (
        normalised.fold_metrics[REPAIR_METRIC] - raw.fold_metrics[REPAIR_METRIC]
    )
    mean_accuracy_delta = float(accuracy_delta.mean())
    mean_repair_delta = float(repair_delta.mean())
    fold_wins = int(accuracy_delta.gt(0).sum())
    worst_fold_delta = float(accuracy_delta.min())
    passes = (
        _meets(mean_accuracy_delta, MEAN_ACCURACY_GAIN_GATE)
        and fold_wins >= FOLD_WIN_GATE
        and _meets(worst_fold_delta, WORST_FOLD_DELTA_GATE)
        and _meets(mean_repair_delta, REPAIR_RECALL_DELTA_GATE)
    )
    summary = _pd.DataFrame(
        [
            {
                "candidate": "raw_top_50_identities",
                "mean_accuracy": float(raw.fold_metrics["accuracy"].mean()),
                "mean_accuracy_delta": 0.0,
                "fold_wins": 0,
                "worst_fold_delta": 0.0,
                "repair_recall": float(raw.fold_metrics[REPAIR_METRIC].mean()),
                "repair_recall_delta": 0.0,
                "passes_gate": False,
            },
            {
                "candidate": "normalised_top_50_identities",
                "mean_accuracy": float(normalised.fold_metrics["accuracy"].mean()),
                "mean_accuracy_delta": mean_accuracy_delta,
                "fold_wins": fold_wins,
                "worst_fold_delta": worst_fold_delta,
                "repair_recall": float(
                    normalised.fold_metrics[REPAIR_METRIC].mean()
                ),
                "repair_recall_delta": mean_repair_delta,
                "passes_gate": passes,
            },
        ]
    ).set_index("candidate")
    folds = _pd.DataFrame(
        {
            "raw_accuracy": raw.fold_metrics["accuracy"],
            "normalised_accuracy": normalised.fold_metrics["accuracy"],
            "accuracy_delta": accuracy_delta,
            "normalised_wins": accuracy_delta.gt(0),
            "raw_repair_recall": raw.fold_metrics[REPAIR_METRIC],
            "normalised_repair_recall": normalised.fold_metrics[REPAIR_METRIC],
            "repair_recall_delta": repair_delta,
        }
    )
    folds.index.name = "validation_fold"
    return summary, folds


def _meets(value: float, threshold: float) -> bool:
    return value >= threshold or bool(
        _np.isclose(value, threshold, rtol=0.0, atol=1e-12)
    )

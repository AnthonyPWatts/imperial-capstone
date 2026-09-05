"""Locked comparison helpers for CatBoost refit-iteration calibration."""

from __future__ import annotations

from dataclasses import asdict as _asdict
from dataclasses import dataclass as _dataclass
import json as _json
from numbers import Integral as _Integral
from pathlib import Path as _Path
import time as _time

import joblib as _joblib
import numpy as _np
import pandas as _pd

from catboost_identity_evaluation import CatBoostIdentityTrial
from catboost_identity_evaluation import engineer_complete_identity_catboost_features
from final_model import validate_probabilities
from gpu_model_evaluation import _make_catboost_model
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from locked_architecture_candidates import sha256_file
from locked_architecture_candidates import validate_screen_contract
from locked_screen_evidence import recompute_and_validate_oof_evaluation
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation


INNER_FIT_SHARE = 0.90
MEAN_ACCURACY_GAIN_GATE = 0.001
FOLD_WIN_GATE = 3
WORST_FOLD_DELTA_GATE = -0.0025
REPAIR_RECALL_DELTA_GATE = -0.02
REPAIR_METRIC = "recall: functional needs repair"
BASELINE_CANDIDATE = "unscaled_complete_identity_catboost"
CORRECTED_CANDIDATE = "corrected_complete_identity_catboost"
CACHE_VERSION = 1


@_dataclass(frozen=True)
class BaselineEvidence:
    """Validated cached baseline and its fold-selected iteration counts."""

    evaluation: CandidateEvaluation
    selected_iterations: dict[int, int]
    provenance: dict[str, object]


def corrected_refit_iterations(selected_iterations: int) -> int:
    """Scale a 90%-fit stopping count to the complete outer-training fold."""

    if isinstance(selected_iterations, bool) or not isinstance(
        selected_iterations,
        _Integral,
    ):
        raise TypeError("Selected CatBoost iterations must be an integer.")
    selected = int(selected_iterations)
    if selected <= 0:
        raise ValueError("Selected CatBoost iterations must be positive.")

    # This is exactly ceil(selected / 0.9), expressed with integer arithmetic
    # so a binary floating-point boundary cannot add an accidental tree.
    return (10 * selected + 8) // 9


def summarise_refit_calibration(
    baseline: CandidateEvaluation,
    corrected: CandidateEvaluation,
) -> tuple[_pd.DataFrame, _pd.DataFrame]:
    """Return the predeclared four-part gate and paired fold evidence."""

    if baseline.cross_validation_fingerprint != (
        corrected.cross_validation_fingerprint
    ):
        raise ValueError("CatBoost refits use different cross-validation folds.")
    if not baseline.fold_metrics.index.equals(corrected.fold_metrics.index):
        raise ValueError("CatBoost refit fold labels do not align.")
    if len(baseline.fold_metrics) != 5:
        raise ValueError("CatBoost refit calibration requires exactly five folds.")
    required = ["accuracy", REPAIR_METRIC]
    if not set(required).issubset(baseline.fold_metrics) or not set(
        required
    ).issubset(corrected.fold_metrics):
        raise ValueError("CatBoost refit metrics are incomplete.")
    values = _np.concatenate(
        [
            baseline.fold_metrics.loc[:, required]
            .to_numpy(dtype="float64")
            .ravel(),
            corrected.fold_metrics.loc[:, required]
            .to_numpy(dtype="float64")
            .ravel(),
        ]
    )
    if not _np.isfinite(values).all():
        raise ValueError("CatBoost refit metrics must be finite.")

    accuracy_delta = (
        corrected.fold_metrics["accuracy"]
        - baseline.fold_metrics["accuracy"]
    )
    repair_delta = (
        corrected.fold_metrics[REPAIR_METRIC]
        - baseline.fold_metrics[REPAIR_METRIC]
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
                "candidate": BASELINE_CANDIDATE,
                "mean_accuracy": float(
                    baseline.fold_metrics["accuracy"].mean()
                ),
                "mean_accuracy_delta": 0.0,
                "fold_wins": 0,
                "worst_fold_delta": 0.0,
                "repair_recall": float(
                    baseline.fold_metrics[REPAIR_METRIC].mean()
                ),
                "repair_recall_delta": 0.0,
                "passes_gate": False,
            },
            {
                "candidate": CORRECTED_CANDIDATE,
                "mean_accuracy": float(
                    corrected.fold_metrics["accuracy"].mean()
                ),
                "mean_accuracy_delta": mean_accuracy_delta,
                "fold_wins": fold_wins,
                "worst_fold_delta": worst_fold_delta,
                "repair_recall": float(
                    corrected.fold_metrics[REPAIR_METRIC].mean()
                ),
                "repair_recall_delta": mean_repair_delta,
                "passes_gate": passes,
            },
        ]
    ).set_index("candidate")
    folds = _pd.DataFrame(
        {
            "baseline_accuracy": baseline.fold_metrics["accuracy"],
            "corrected_accuracy": corrected.fold_metrics["accuracy"],
            "accuracy_delta": accuracy_delta,
            "corrected_wins": accuracy_delta.gt(0),
            "baseline_repair_recall": baseline.fold_metrics[REPAIR_METRIC],
            "corrected_repair_recall": corrected.fold_metrics[REPAIR_METRIC],
            "repair_recall_delta": repair_delta,
        }
    )
    folds.index.name = "validation_fold"
    return summary, folds


def load_baseline_evidence(
    cache_path: _Path,
    protocol_path: _Path,
    partitioned_data,
    cross_validation,
    *,
    expected_cache_sha256: str,
    expected_protocol_sha256: str,
    expected_metadata: dict[str, object],
    expected_recipe: dict[str, object],
    expected_feature_policy: dict[str, object],
) -> BaselineEvidence:
    """Load the pinned unscaled baseline and reconstruct all OOF metrics."""

    cache_hash = sha256_file(cache_path)
    protocol_hash = sha256_file(protocol_path)
    if cache_hash != expected_cache_sha256:
        raise ValueError("The locked CatBoost baseline cache hash changed.")
    if protocol_hash != expected_protocol_sha256:
        raise ValueError("The locked CatBoost baseline protocol hash changed.")

    payload = _joblib.load(cache_path)
    if not isinstance(payload, dict) or set(payload) != {
        *expected_metadata,
        "trial",
    }:
        raise ValueError("The locked CatBoost baseline cache is malformed.")
    validate_screen_contract(
        payload,
        expected_metadata,
        screen="CatBoost refit baseline cache",
    )
    trial = payload.get("trial")
    if not isinstance(trial, CatBoostIdentityTrial):
        raise ValueError("The baseline cache has no CatBoostIdentityTrial.")

    protocol = _json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_protocol = {
        "protocol_version": 2,
        "migration": "strict v1 cache validation; no model refit",
        "cache_sha256": cache_hash,
        "validated_v1_metadata": expected_metadata,
        "model_recipe": expected_recipe,
        "feature_policy_signature": expected_feature_policy,
    }
    if protocol != expected_protocol:
        raise ValueError("The locked CatBoost baseline protocol changed.")

    evaluation = recompute_and_validate_oof_evaluation(
        trial.evaluation,
        partitioned_data,
        cross_validation,
        candidate="complete_identity_catboost",
    )
    diagnostics = trial.evaluation.diagnostics
    expected_index = _pd.Index(range(1, 6), name="validation_fold")
    if not diagnostics.index.equals(expected_index):
        raise ValueError("The baseline diagnostic folds changed.")
    selected = diagnostics["selected_iterations"]
    if selected.isna().any() or not _np.equal(selected, _np.floor(selected)).all():
        raise ValueError("The baseline selected iterations are invalid.")
    selected_iterations = {
        int(fold): int(value) for fold, value in selected.items()
    }
    if any(value <= 0 for value in selected_iterations.values()):
        raise ValueError("The baseline selected iterations must be positive.")
    return BaselineEvidence(
        evaluation=evaluation,
        selected_iterations=selected_iterations,
        provenance={
            "cache_path": cache_path.as_posix(),
            "cache_sha256": cache_hash,
            "protocol_path": protocol_path.as_posix(),
            "protocol_sha256": protocol_hash,
            "metadata": expected_metadata,
            "model_recipe": expected_recipe,
            "feature_policy_signature": expected_feature_policy,
            "derived_metrics_reconstructed_from_oof": True,
        },
    )


def make_refit_recipe(
    spec,
    selected_iterations: dict[int, int],
) -> dict[str, object]:
    """Freeze the complete fixed-iteration model and correction contract."""

    expected_folds = set(range(1, 6))
    if set(selected_iterations) != expected_folds:
        raise ValueError("The refit recipe requires selected counts for five folds.")
    corrected = {
        fold: corrected_refit_iterations(selected_iterations[fold])
        for fold in range(1, 6)
    }
    parameters = []
    for fold in range(1, 6):
        model = _make_catboost_model(
            spec,
            iterations=corrected[fold],
            early_stopping=False,
        )
        resolved = model.get_params()
        if "od_type" in resolved or "od_wait" in resolved:
            raise ValueError("Corrected CatBoost unexpectedly enables stopping.")
        parameters.append(
            {"validation_fold": fold, "parameters": resolved}
        )
    return {
        "spec": _asdict(spec),
        "feature_engineer": (
            "catboost_identity_evaluation."
            "engineer_complete_identity_catboost_features"
        ),
        "inner_fit_share": INNER_FIT_SHARE,
        "iteration_correction": "ceil(selected_iterations / 0.9)",
        "new_early_stopping": False,
        "baseline_selected_iterations_by_fold": [
            selected_iterations[fold] for fold in range(1, 6)
        ],
        "corrected_refit_iterations_by_fold": [
            corrected[fold] for fold in range(1, 6)
        ],
        "model_parameters_by_fold": parameters,
    }


def evaluate_corrected_refits(
    spec,
    partitioned_data,
    cross_validation,
    *,
    selected_iterations: dict[int, int],
    cache_directory: _Path,
    common_metadata: dict[str, object],
) -> CandidateEvaluation:
    """Fit or strictly reload the five corrected outer-fold refits."""

    cache_directory.mkdir(parents=True, exist_ok=True)
    probability_values = _np.full(
        (len(partitioned_data.y_development), 3),
        _np.nan,
    )
    diagnostics = []
    for fold, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        corrected = corrected_refit_iterations(selected_iterations[fold])
        validation_ids = partitioned_data.development_ids.iloc[
            validation_positions
        ].to_numpy()
        metadata = {
            **common_metadata,
            "validation_fold": fold,
            "baseline_selected_iterations": selected_iterations[fold],
            "corrected_refit_iterations": corrected,
            "outer_training_rows": len(training_positions),
            "validation_rows": len(validation_positions),
            "outer_training_id_sha256": _ids_sha256(
                partitioned_data.development_ids.iloc[training_positions]
            ),
            "validation_id_sha256": _ids_sha256(
                partitioned_data.development_ids.iloc[validation_positions]
            ),
        }
        path = cache_directory / f"corrected-fold-{fold}.joblib"
        if path.exists():
            payload = _joblib.load(path)
            fold_probabilities, fold_diagnostics = _validate_fold_cache(
                payload,
                metadata,
                validation_ids,
                path,
            )
            print(f"Loaded {path}.", flush=True)
        else:
            started = _time.perf_counter()
            fold_probabilities, fit_seconds = fit_gpu_candidate_probabilities(
                spec,
                partitioned_data.X_development.iloc[training_positions],
                partitioned_data.y_development.iloc[training_positions],
                partitioned_data.X_development.iloc[validation_positions],
                iterations=corrected,
                catboost_feature_engineer=(
                    engineer_complete_identity_catboost_features
                ),
            )
            fold_diagnostics = {
                "validation_fold": fold,
                "selected_iterations": corrected,
                "baseline_selected_iterations": selected_iterations[fold],
                "corrected_refit_iterations": corrected,
                "inner_fit_rows": len(training_positions),
                "inner_stop_rows": 0,
                "stopping_seconds": 0.0,
                "refit_predict_seconds": fit_seconds,
                "total_seconds": _time.perf_counter() - started,
            }
            payload = {
                "metadata": metadata,
                "validation_ids": validation_ids,
                "probabilities": fold_probabilities,
                "diagnostics": fold_diagnostics,
            }
            _validate_fold_cache(payload, metadata, validation_ids, path)
            _joblib.dump(payload, path, compress=3)
            print(
                f"Completed corrected CatBoost fold {fold}/5: "
                f"{selected_iterations[fold]} -> {corrected} trees in "
                f"{fold_diagnostics['total_seconds']:.1f}s.",
                flush=True,
            )
        probability_values[validation_positions] = fold_probabilities
        diagnostics.append(fold_diagnostics)

    validate_probabilities(probability_values, len(partitioned_data.y_development))
    return build_candidate_evaluation(
        model_name=spec.name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostics,
    )


def paired_prediction_counts(
    baseline: CandidateEvaluation,
    corrected: CandidateEvaluation,
    partitioned_data,
    cross_validation,
) -> _pd.DataFrame:
    """Return exact gained/lost correctness counts for each paired fold."""

    baseline_predictions = baseline.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    corrected_predictions = corrected.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    rows = []
    for fold, (_, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        actual = partitioned_data.y_development.iloc[
            validation_positions
        ].to_numpy()
        base = baseline_predictions[validation_positions]
        candidate = corrected_predictions[validation_positions]
        base_correct = base == actual
        candidate_correct = candidate == actual
        rows.append(
            {
                "validation_fold": fold,
                "validation_rows": len(validation_positions),
                "prediction_disagreements": int((base != candidate).sum()),
                "gained_correct": int((candidate_correct & ~base_correct).sum()),
                "lost_correct": int((~candidate_correct & base_correct).sum()),
                "net_additional_correct": int(
                    candidate_correct.sum() - base_correct.sum()
                ),
            }
        )
    return _pd.DataFrame(rows).set_index("validation_fold")


def _validate_fold_cache(payload, metadata, validation_ids, path):
    if not isinstance(payload, dict) or set(payload) != {
        "metadata",
        "validation_ids",
        "probabilities",
        "diagnostics",
    }:
        raise ValueError(f"Malformed corrected fold cache: {path}")
    if payload["metadata"] != metadata:
        raise ValueError(f"Stale corrected fold metadata: {path}")
    if not _np.array_equal(payload["validation_ids"], validation_ids):
        raise ValueError(f"Misaligned corrected validation IDs: {path}")
    validate_probabilities(payload["probabilities"], len(validation_ids))
    diagnostics = payload["diagnostics"]
    expected = {
        "validation_fold": metadata["validation_fold"],
        "selected_iterations": metadata["corrected_refit_iterations"],
        "baseline_selected_iterations": metadata[
            "baseline_selected_iterations"
        ],
        "corrected_refit_iterations": metadata[
            "corrected_refit_iterations"
        ],
        "inner_fit_rows": metadata["outer_training_rows"],
        "inner_stop_rows": 0,
        "stopping_seconds": 0.0,
    }
    if not isinstance(diagnostics, dict) or any(
        diagnostics.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(f"Incompatible corrected fold diagnostics: {path}")
    for key in ("refit_predict_seconds", "total_seconds"):
        value = diagnostics.get(key)
        if not isinstance(value, (int, float)) or not _np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid corrected fold timing: {path}")
    return payload["probabilities"], diagnostics


def _ids_sha256(ids) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _meets(value: float, threshold: float) -> bool:
    return value >= threshold or bool(
        _np.isclose(value, threshold, rtol=0.0, atol=1e-12)
    )

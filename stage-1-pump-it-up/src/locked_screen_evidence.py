"""Validate fresh-screen evidence used by locked architecture hypotheses."""

from __future__ import annotations

from dataclasses import asdict as _asdict
from dataclasses import dataclass as _dataclass
import json as _json
from pathlib import Path as _Path
import re as _re

import joblib as _joblib

from catboost_identity_evaluation import CatBoostIdentityTrial
from catboost_identity_evaluation import DEFERRED_IDENTITY_FEATURES
from catboost_identity_evaluation import make_complete_identity_catboost_spec
from data_partitioning import make_cross_validation
from final_model import validate_probabilities
from fresh_rf_family_evaluation import CURRENT_RF_VARIANT
from fresh_rf_family_evaluation import FOLD_WIN_GATE
from fresh_rf_family_evaluation import FRESH_CROSS_VALIDATION_SEED
from fresh_rf_family_evaluation import LOCKED_MODEL_SEED
from fresh_rf_family_evaluation import LOCKED_RF_VARIANTS
from fresh_rf_family_evaluation import MEAN_ACCURACY_GAIN_GATE
from fresh_rf_family_evaluation import REPAIR_RECALL_DELTA_GATE
from fresh_rf_family_evaluation import RF_SCREEN_CACHE_VERSION
from fresh_rf_family_evaluation import WORST_FOLD_DELTA_GATE
from fresh_rf_family_evaluation import make_full_labelled_partition
from fresh_rf_family_evaluation import make_locked_rf_specs
from fresh_rf_family_evaluation import paired_fold_deltas
from fresh_rf_family_evaluation import summarise_rf_screen
from fresh_rf_family_evaluation import validate_rf_cache_payload
from gpu_model_evaluation import CATBOOST_VARIANTS
from gpu_model_evaluation import GpuCandidateSpec
from gpu_model_evaluation import SKLEARN_TREE_VARIANTS
from gpu_model_evaluation import median_selected_iterations
from locked_architecture_candidates import sha256_file
from locked_architecture_candidates import validate_screen_contract
from model_evaluation import build_candidate_evaluation
from model_evaluation import CandidateEvaluation
from modelling_data import ModellingData
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_ACCURACY
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_FOLD_WINS
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_WORST_FOLD
from spatial_grid_catboost_evaluation import make_spatial_grid_catboost_spec
from spatial_grid_catboost_evaluation import spatial_grid_policy_signature
from spatial_grid_catboost_evaluation import (
    summarise_spatial_grid_catboost_comparison,
)


FRESH_FOLD_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)


@_dataclass(frozen=True)
class LockedScreenEvidence:
    """A reproduced screen decision and its fixed full-data recipe."""

    result: dict[str, object]
    spec: GpuCandidateSpec
    iterations: int
    hashes: dict[str, str]


def load_spatial_screen_evidence(
    directory: _Path,
    modelling: ModellingData,
) -> LockedScreenEvidence:
    """Reproduce the spatial gate and verify both v2 cache sidecars."""

    result = _read_json(directory / "result.json")
    identity_spec = make_complete_identity_catboost_spec()
    grid_spec = make_spatial_grid_catboost_spec()
    recipes = {
        "complete_identity_catboost": _catboost_recipe(identity_spec),
        "spatial_grid_catboost": _catboost_recipe(grid_spec),
    }
    policies = {
        "complete_identity_catboost": _identity_policy_signature(),
        "spatial_grid_catboost": spatial_grid_policy_signature(),
    }
    validate_screen_contract(
        result,
        {
            "protocol": {
                "labelled_rows": len(modelling.y_original),
                "cross_validation_folds": 5,
                "cross_validation_seed": 20260905,
                "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
                "historical_local_subset": "not created or consulted",
                "competition_predictions_generated": False,
            },
            "gate": {
                "minimum_accuracy_change": SPATIAL_GRID_GATE_ACCURACY,
                "minimum_fold_wins": SPATIAL_GRID_GATE_FOLD_WINS,
                "minimum_worst_fold_change": SPATIAL_GRID_GATE_WORST_FOLD,
            },
            "model_recipes": recipes,
            "feature_policy_signatures": policies,
        },
        screen="spatial",
    )
    partitioned = make_full_labelled_partition(modelling)
    cross_validation = make_cross_validation(partitioned)
    identity_trial = _load_spatial_trial(
        directory,
        "complete_identity_catboost",
        identity_spec,
        policies["complete_identity_catboost"],
        modelling,
    )
    grid_trial = _load_spatial_trial(
        directory,
        "spatial_grid_catboost",
        grid_spec,
        policies["spatial_grid_catboost"],
        modelling,
    )
    identity = recompute_and_validate_oof_evaluation(
        identity_trial.evaluation,
        partitioned,
        cross_validation,
        candidate="complete_identity_catboost",
    )
    grid = recompute_and_validate_oof_evaluation(
        grid_trial.evaluation,
        partitioned,
        cross_validation,
        candidate="spatial_grid_catboost",
    )
    summary, _ = summarise_spatial_grid_catboost_comparison(
        identity,
        grid,
    )
    passed = bool(summary.loc["spatial_grid_catboost", "passes_gate"])
    selected = "spatial_grid_catboost" if passed else "complete_identity_catboost"
    iterations = median_selected_iterations(grid_trial.evaluation)
    validate_screen_contract(
        result,
        {
            "passes_gate": passed,
            "selected_candidate": selected,
            "full_data_refit_iterations": {
                "spatial_grid_catboost": iterations,
            },
        },
        screen="spatial",
    )
    paths = _spatial_paths(directory)
    _require_files(paths.values())
    return LockedScreenEvidence(
        result=result,
        spec=grid_spec,
        iterations=iterations,
        hashes={name: sha256_file(path) for name, path in paths.items()},
    )


def load_rf_screen_evidence(
    directory: _Path,
    modelling: ModellingData,
) -> LockedScreenEvidence:
    """Reproduce the RF gate from strictly versioned candidate caches."""

    result = _read_json(directory / "result.json")
    partitioned = make_full_labelled_partition(modelling)
    if partitioned.cross_validation_fingerprint != FRESH_FOLD_FINGERPRINT:
        raise ValueError("RF fresh-fold fingerprint changed.")
    specs = make_locked_rf_specs()
    validate_screen_contract(
        result,
        {
            "cache_version": RF_SCREEN_CACHE_VERSION,
            "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
            "model_seed": LOCKED_MODEL_SEED,
            "folds": 5,
            "labelled_rows": len(modelling.y_original),
            "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
            "incumbent": CURRENT_RF_VARIANT,
            "locked_candidates": list(LOCKED_RF_VARIANTS),
            "locked_specs": {name: _asdict(spec) for name, spec in specs.items()},
            "gate_thresholds": {
                "mean_accuracy_gain": MEAN_ACCURACY_GAIN_GATE,
                "minimum_fold_wins": FOLD_WIN_GATE,
                "worst_fold_delta": WORST_FOLD_DELTA_GATE,
                "minimum_repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
            },
            "old_local_test_scored_separately": False,
            "competition_predictions_generated": False,
        },
        screen="RF",
    )
    paths = _rf_paths(directory)
    _require_files(paths.values())
    _validate_rf_cache_provenance(
        _read_json(paths["provenance"]),
        paths,
    )
    cached_evaluations = {
        name: validate_rf_cache_payload(
            _joblib.load(paths[f"evaluation:{name}"]),
            spec,
            partitioned,
        )
        for name, spec in specs.items()
    }
    cross_validation = make_cross_validation(partitioned)
    evaluations = {
        name: recompute_and_validate_oof_evaluation(
            evaluation,
            partitioned,
            cross_validation,
            candidate=name,
        )
        for name, evaluation in cached_evaluations.items()
    }
    paired = paired_fold_deltas(
        partitioned,
        cross_validation,
        evaluations,
    )
    summary = summarise_rf_screen(evaluations, paired)
    passing = summary.loc[summary["passes_gate"]]
    passed = not passing.empty
    selected = str(passing.index[0]) if passed else CURRENT_RF_VARIANT
    validate_screen_contract(
        result,
        {
            "passes_gate": passed,
            "selected_candidate": selected,
            "selected_candidate_is_challenger": selected != CURRENT_RF_VARIANT,
        },
        screen="RF",
    )
    return LockedScreenEvidence(
        result=result,
        spec=specs[selected],
        iterations=int(SKLEARN_TREE_VARIANTS[selected]["n_estimators"]),
        hashes={name: sha256_file(path) for name, path in paths.items()},
    )


def recompute_and_validate_oof_evaluation(
    evaluation: CandidateEvaluation,
    partitioned_data,
    cross_validation,
    *,
    candidate: str,
) -> CandidateEvaluation:
    """Rebuild every scored metric from locked OOF rows and current labels."""

    probabilities = evaluation.out_of_fold_probabilities
    expected_index = partitioned_data.y_development.index
    expected_columns = (
        "functional",
        "functional needs repair",
        "non functional",
    )
    if not probabilities.index.equals(expected_index):
        raise ValueError(f"{candidate} OOF rows are misaligned.")
    if tuple(probabilities.columns) != expected_columns:
        raise ValueError(f"{candidate} OOF class columns changed.")
    validate_probabilities(
        probabilities.to_numpy(),
        len(partitioned_data.y_development),
    )
    if evaluation.cross_validation_fingerprint != (
        partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError(f"{candidate} OOF fold fingerprint changed.")

    recomputed = build_candidate_evaluation(
        model_name=evaluation.model_name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probabilities.to_numpy(),
        diagnostic_rows=[
            {"validation_fold": fold_number}
            for fold_number in range(1, 6)
        ],
    )
    for field in (
        "fold_metrics",
        "metric_summary",
        "confusion_counts",
        "confusion_recall",
    ):
        if not getattr(evaluation, field).equals(getattr(recomputed, field)):
            raise ValueError(
                f"{candidate} cached {field} does not match its OOF evidence."
            )
    return recomputed


def _load_spatial_trial(
    directory: _Path,
    candidate: str,
    spec: GpuCandidateSpec,
    policy: dict[str, object],
    modelling: ModellingData,
) -> CatBoostIdentityTrial:
    stem = {
        "complete_identity_catboost": "complete-identity-catboost",
        "spatial_grid_catboost": "spatial-grid-catboost",
    }[candidate]
    cache_path = directory / f"{stem}.joblib"
    payload = _joblib.load(cache_path)
    legacy = {
        "cache_version": 1,
        "candidate": candidate,
        "labelled_rows": len(modelling.y_original),
        "cross_validation_folds": 5,
        "cross_validation_seed": 20260905,
        "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
        "model_name": spec.name,
    }
    if not isinstance(payload, dict):
        raise ValueError("Spatial cache is not a versioned payload.")
    validate_screen_contract(payload, legacy, screen="spatial cache")
    trial = payload.get("trial")
    if not isinstance(trial, CatBoostIdentityTrial):
        raise ValueError("Spatial cache has no CatBoostIdentityTrial.")
    _validate_evaluation(trial.evaluation, spec.name, len(modelling.y_original))
    expected_protocol = {
        "protocol_version": 2,
        "migration": "strict v1 cache validation; no model refit",
        "cache_sha256": sha256_file(cache_path),
        "validated_v1_metadata": legacy,
        "model_recipe": _catboost_recipe(spec),
        "feature_policy_signature": policy,
    }
    if _read_json(cache_path.with_suffix(".protocol-v2.json")) != expected_protocol:
        raise ValueError("Spatial cache protocol signature changed.")
    return trial


def _validate_evaluation(
    evaluation: CandidateEvaluation,
    model_name: str,
    rows: int,
) -> None:
    if evaluation.model_name != model_name:
        raise ValueError("Fresh-screen model name changed.")
    if evaluation.cross_validation_fingerprint != FRESH_FOLD_FINGERPRINT:
        raise ValueError("Fresh-screen fold fingerprint changed.")
    if len(evaluation.fold_metrics) != 5:
        raise ValueError("Fresh-screen fold count changed.")
    validate_probabilities(evaluation.out_of_fold_probabilities.to_numpy(), rows)


def _catboost_recipe(spec: GpuCandidateSpec) -> dict[str, object]:
    return {
        "spec": _asdict(spec),
        "variant_parameters": dict(CATBOOST_VARIANTS[spec.variant]),
    }


def _identity_policy_signature() -> dict[str, object]:
    return {
        "version": 1,
        "feature_engineer": "engineer_complete_identity_catboost_features",
        "deferred_identity_features": list(DEFERRED_IDENTITY_FEATURES),
    }


def _spatial_paths(directory: _Path) -> dict[str, _Path]:
    return {
        "result": directory / "result.json",
        "summary": directory / "candidate-summary.csv",
        "folds": directory / "fold-deltas.csv",
        "identity": directory / "complete-identity-catboost.joblib",
        "identity_protocol": (
            directory / "complete-identity-catboost.protocol-v2.json"
        ),
        "grid": directory / "spatial-grid-catboost.joblib",
        "grid_protocol": directory / "spatial-grid-catboost.protocol-v2.json",
    }


def _rf_paths(directory: _Path) -> dict[str, _Path]:
    paths = {
        "result": directory / "result.json",
        "summary": directory / "candidate-summary.csv",
        "folds": directory / "fold-paired-deltas.csv",
        "provenance": directory / "cache-provenance.json",
    }
    paths.update(
        {
            f"evaluation:{name}": directory / f"{_slug(name)}.joblib"
            for name in LOCKED_RF_VARIANTS
        }
    )
    paths.update(
        {
            f"bare:{name}": directory / f"{_slug(name)}.bare-v0.joblib"
            for name in LOCKED_RF_VARIANTS
        }
    )
    return paths


def _validate_rf_cache_provenance(
    provenance: dict[str, object],
    paths: dict[str, _Path],
) -> None:
    """Verify the same-run bare-to-versioned RF cache transition."""

    validate_screen_contract(
        provenance,
        {
            "transition": (
                "Same-run bare CandidateEvaluation wrapped after cache "
                "contract was hardened"
            ),
            "metadata_attached_after_training": True,
            "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
        },
        screen="RF cache provenance",
    )
    caches = provenance.get("caches")
    if not isinstance(caches, dict) or set(caches) != set(LOCKED_RF_VARIANTS):
        raise ValueError("RF cache provenance candidate set changed.")
    for candidate in LOCKED_RF_VARIANTS:
        record = caches[candidate]
        if not isinstance(record, dict):
            raise ValueError("RF cache provenance record changed.")
        versioned_path = paths[f"evaluation:{candidate}"]
        bare_path = paths[f"bare:{candidate}"]
        expected = {
            "bare_cache": bare_path.name,
            "bare_sha256": sha256_file(bare_path),
            "versioned_cache": versioned_path.name,
            "versioned_sha256": sha256_file(versioned_path),
        }
        if record != expected:
            raise ValueError(
                f"RF cache provenance changed for {candidate!r}."
            )


def _read_json(path: _Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = _json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _require_files(paths) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing evidence files: {missing!r}.")


def _slug(value: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")

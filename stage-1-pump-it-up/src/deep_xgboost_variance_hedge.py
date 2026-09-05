"""Strict contracts for the optional two-seed deep-XGBoost variance hedge."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import hashlib as _hashlib
import json as _json
from pathlib import Path as _Path
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd

from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from deep_archive_confirmation import blend_deep_archive
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from gpu_model_evaluation import XGBOOST_VARIANTS
from top_common_identity_evaluation import ARCHIVED_DEPTH_17_ITERATIONS
from top_common_identity_evaluation import TOP_COMMON_VALUES


SEED_20260824 = 20260824
SEED_20260905 = 20260905
DEEP_COMPONENT = "deep_xgboost"
DEEP_COMPONENT_WEIGHT = 0.33
CACHE_VERSION = 1
OPT_IN_TOKEN = "optional-stable-positive-near-miss-variance-hedge"
OPTIONAL_STATUS = "optional_stable_positive_near_miss_variance_hedge"

EXPECTED_ENSEMBLE_WEIGHTS = {
    "deep_xgboost": 0.33,
    "spatial_xgboost": 0.11,
    "random_forest": 0.18,
    "frequency_random_forest": 0.09,
    "spatial_random_forest": 0.09,
    "identity_catboost": 0.20,
}
EXPECTED_GATE = {
    "minimum_mean_accuracy_gain": 0.001,
    "minimum_fold_wins": 3,
    "minimum_worst_fold_delta": -0.0025,
    "minimum_repair_recall_delta": -0.02,
    "must_pass_against_both_seeds": True,
}


@_dataclass(frozen=True)
class VarianceHedgeProbabilities:
    """The exact seed average and the fixed-weight ensemble it produces."""

    deep_seed_average: _np.ndarray
    incumbent_ensemble: _np.ndarray
    candidate_ensemble: _np.ndarray


def canonical_deep_xgboost_recipe(seed: int) -> dict[str, object]:
    """Return the complete full-training recipe for one deep component."""

    if seed not in {SEED_20260824, SEED_20260905}:
        raise ValueError(f"Seed {seed} is outside the locked two-seed recipe.")
    return {
        "family": "XGBoost GPU",
        "variant": "archived depth 17",
        "variant_parameters": dict(XGBOOST_VARIANTS["archived depth 17"]),
        "model_seed": seed,
        "iterations": ARCHIVED_DEPTH_17_ITERATIONS,
        "training_scope": "all 59,400 labelled rows",
        "preprocessor": (
            "top_common_identity_evaluation."
            "make_top_common_identity_preprocessor"
        ),
        "identity_fields": list(DEFERRED_HIGH_CARDINALITY_FEATURES),
        "top_k": TOP_COMMON_VALUES,
        "normalise": False,
        "class_order": list(CLASS_LABELS),
    }


def canonical_probability_average_recipe() -> dict[str, object]:
    """Return the sole admitted two-seed combination."""

    return {
        "operation": "row-wise arithmetic mean of aligned class probabilities",
        "weights": {
            "seed_20260824": 0.5,
            "seed_20260905": 0.5,
        },
        "class_order": list(CLASS_LABELS),
        "selection": "single predeclared weight; no search",
    }


def canonical_ensemble_recipe() -> dict[str, object]:
    """Describe the one-component substitution into the incumbent archive."""

    return {
        "incumbent_weights": dict(EXPECTED_ENSEMBLE_WEIGHTS),
        "replacement": {
            "component": DEEP_COMPONENT,
            "unchanged_weight": DEEP_COMPONENT_WEIGHT,
            "probabilities": canonical_probability_average_recipe(),
        },
        "all_other_components": "byte-pinned incumbent probability caches",
        "post_hoc_row_changes": 0,
    }


def validate_near_miss_evidence(result: object) -> None:
    """Require the locked no-promotion, stable-positive near-miss decision."""

    if not isinstance(result, dict):
        raise ValueError("Two-seed evidence result must be a JSON object.")
    required_decision = {
        "passes_gate": False,
        "stable_positive_near_miss": True,
        "verdict": "stable_positive_near_miss",
        "selected_candidate": None,
        "promotion_decision": "do_not_promote",
        "competition_predictions_generated": False,
    }
    for key, expected in required_decision.items():
        if result.get(key) != expected:
            raise ValueError(
                f"Two-seed evidence no-promotion field {key!r} changed."
            )

    protocol = result.get("protocol")
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != expected
        for key, expected in {
            "cache_only": True,
            "models_fitted": 0,
            "labelled_rows": 59_400,
            "folds": 5,
            "fold_seed": SEED_20260905,
            "comparison": "only the fixed 50:50 probability average",
            "competition_predictions_generated": False,
        }.items()
    ):
        raise ValueError("Two-seed evidence protocol changed.")
    if result.get("gate") != EXPECTED_GATE:
        raise ValueError("Two-seed evidence gate changed.")

    recipes = result.get("recipes")
    expected_recipes = {
        "seed_20260824": _evidence_deep_recipe(SEED_20260824),
        "seed_20260905": _evidence_deep_recipe(SEED_20260905),
        "probability_average": canonical_probability_average_recipe(),
    }
    if recipes != expected_recipes:
        raise ValueError("Two-seed evidence recipes changed.")

    comparisons = result.get("comparison_summary")
    if not isinstance(comparisons, list) or len(comparisons) != 2:
        raise ValueError("Two-seed evidence comparisons changed.")
    by_name = {
        row.get("comparison"): row
        for row in comparisons
        if isinstance(row, dict)
    }
    expected_names = {
        "average_vs_seed_20260824",
        "average_vs_seed_20260905",
    }
    if set(by_name) != expected_names:
        raise ValueError("Two-seed evidence comparison names changed.")
    for name, row in by_name.items():
        if row.get("passes_gate") is not False:
            raise ValueError(f"{name} unexpectedly passes the promotion gate.")
        if not 0.0 < _finite_float(row.get("mean_accuracy_delta"), name) < 0.001:
            raise ValueError(f"{name} is not a positive mean-gain near-miss.")
        if int(row.get("fold_wins", -1)) < 3:
            raise ValueError(f"{name} no longer satisfies the fold-win guard.")
        if _finite_float(row.get("worst_fold_delta"), name) < -0.0025:
            raise ValueError(f"{name} no longer satisfies the worst-fold guard.")
        if _finite_float(row.get("repair_recall_delta"), name) < -0.02:
            raise ValueError(f"{name} no longer satisfies the repair guard.")

    expected_failures = {
        "average_vs_seed_20260824": ["mean_accuracy_gain"],
        "average_vs_seed_20260905": ["mean_accuracy_gain"],
    }
    if result.get("gate_failures") != expected_failures:
        raise ValueError("Two-seed evidence gate failures changed.")


def build_variance_hedge_probabilities(
    incumbent_components: _Mapping[str, _np.ndarray],
    seed_20260824_probabilities: _np.ndarray,
    seed_20260905_probabilities: _np.ndarray,
) -> VarianceHedgeProbabilities:
    """Replace only the incumbent 0.33 deep slot with the exact seed mean."""

    _validate_locked_weights()
    expected_components = set(EXPECTED_ENSEMBLE_WEIGHTS)
    actual_components = set(incumbent_components)
    if actual_components != expected_components:
        raise ValueError(
            "Incumbent component contract changed; "
            f"missing={sorted(expected_components - actual_components)!r}, "
            f"unexpected={sorted(actual_components - expected_components)!r}."
        )
    expected_rows = len(seed_20260824_probabilities)
    validate_probabilities(seed_20260824_probabilities, expected_rows)
    validate_probabilities(seed_20260905_probabilities, expected_rows)
    for name, probabilities in incumbent_components.items():
        validate_probabilities(probabilities, expected_rows)

    seed_average = 0.5 * (
        _np.asarray(seed_20260824_probabilities, dtype="float64")
        + _np.asarray(seed_20260905_probabilities, dtype="float64")
    )
    validate_probabilities(seed_average, expected_rows)
    incumbent_ensemble = blend_deep_archive(dict(incumbent_components))
    candidate_components = dict(incumbent_components)
    candidate_components[DEEP_COMPONENT] = seed_average
    candidate_ensemble = blend_deep_archive(candidate_components)
    expected_candidate = incumbent_ensemble + DEEP_COMPONENT_WEIGHT * (
        seed_average - incumbent_components[DEEP_COMPONENT]
    )
    if not _np.allclose(
        candidate_ensemble,
        expected_candidate,
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError("Variance hedge changed more than the 0.33 deep slot.")
    validate_probabilities(candidate_ensemble, expected_rows)
    return VarianceHedgeProbabilities(
        deep_seed_average=seed_average,
        incumbent_ensemble=incumbent_ensemble,
        candidate_ensemble=candidate_ensemble,
    )


def make_deep_component_cache_metadata(
    *,
    seed: int,
    training_rows: int,
    competition_rows: int,
    data_sha256: _Mapping[str, str],
    source_sha256: _Mapping[str, str],
    evidence_sha256: _Mapping[str, str],
) -> dict[str, object]:
    """Build complete metadata for a newly fitted full-training component."""

    if seed != SEED_20260905:
        raise ValueError("Only the missing seed-20260905 cache may be created.")
    if training_rows != 59_400 or competition_rows != 14_850:
        raise ValueError("Full-training or competition row counts changed.")
    return {
        "cache_version": CACHE_VERSION,
        "candidate": "deep_xgboost_seed_20260905_full_training",
        "recipe": canonical_deep_xgboost_recipe(seed),
        "training_rows": training_rows,
        "competition_rows": competition_rows,
        "data_sha256": dict(data_sha256),
        "source_sha256": dict(source_sha256),
        "evidence_sha256": dict(evidence_sha256),
    }


def validate_legacy_deep_component_cache(
    payload: object,
    expected_ids: _pd.Series,
    *,
    expected_seed: int,
) -> dict[str, object]:
    """Validate the byte-pinned legacy incumbent deep-component cache."""

    required = {
        "competition_ids",
        "probabilities",
        "seed",
        "iterations",
        "fit_and_predict_seconds",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Incumbent deep-XGBoost cache schema changed.")
    if payload["seed"] != expected_seed:
        raise ValueError("Incumbent deep-XGBoost cache seed changed.")
    if payload["iterations"] != ARCHIVED_DEPTH_17_ITERATIONS:
        raise ValueError("Incumbent deep-XGBoost iterations changed.")
    _validate_cache_common(payload, expected_ids, name="incumbent deep-XGBoost")
    return payload


def validate_versioned_deep_component_cache(
    payload: object,
    expected_metadata: _Mapping[str, object],
    expected_ids: _pd.Series,
) -> dict[str, object]:
    """Validate a reusable seed-20260905 full-training component cache."""

    required = {
        "cache_version",
        "metadata",
        "competition_ids",
        "probabilities",
        "fit_and_predict_seconds",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Seed-20260905 deep-XGBoost cache schema changed.")
    if payload["cache_version"] != CACHE_VERSION:
        raise ValueError("Seed-20260905 deep-XGBoost cache version changed.")
    if payload["metadata"] != dict(expected_metadata):
        raise ValueError("Seed-20260905 deep-XGBoost cache metadata is stale.")
    _validate_cache_common(payload, expected_ids, name="seed-20260905 deep-XGBoost")
    return payload


def validate_component_ids(
    payload: object,
    expected_ids: _pd.Series,
    *,
    name: str,
) -> None:
    """Require a cache's competition IDs to match in value and row order."""

    if not isinstance(payload, dict):
        raise ValueError(f"{name} cache is not a mapping.")
    ids = payload.get("competition_ids")
    if not isinstance(ids, _pd.Series) or not ids.reset_index(drop=True).equals(
        expected_ids.reset_index(drop=True)
    ):
        raise ValueError(f"{name} competition IDs changed or are misordered.")


def validate_submission_frame(
    submission: _pd.DataFrame,
    expected_ids: _pd.Series,
) -> None:
    """Validate exact DrivenData columns, IDs, row count and labels."""

    if list(submission.columns) != ["id", "status_group"]:
        raise ValueError("Variance-hedge submission columns changed.")
    if len(submission) != len(expected_ids):
        raise ValueError("Variance-hedge submission row count changed.")
    if submission.isna().any().any() or submission["id"].duplicated().any():
        raise ValueError("Variance-hedge submission has missing or duplicate rows.")
    if not submission["id"].reset_index(drop=True).equals(
        expected_ids.reset_index(drop=True)
    ):
        raise ValueError("Variance-hedge submission IDs changed or are misordered.")
    if not set(submission["status_group"]).issubset(CLASS_LABELS):
        raise ValueError("Variance-hedge submission contains an invalid label.")


def write_submission_without_overwrite(
    submission: _pd.DataFrame,
    expected_ids: _pd.Series,
    destination: _Path,
) -> _Path:
    """Write a validated CSV exactly once; even an identical file is refused."""

    destination = _Path(destination).resolve()
    validate_submission_frame(submission, expected_ids)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite submission: {destination}.")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Submission directory does not exist: {destination.parent}."
        )
    submission.to_csv(destination, index=False, mode="x")
    reloaded = _pd.read_csv(destination)
    validate_submission_frame(reloaded, expected_ids)
    if not reloaded.equals(submission.reset_index(drop=True)):
        raise ValueError("Reloaded variance-hedge submission changed.")
    return destination


def write_json_without_overwrite(
    value: _Mapping[str, object],
    destination: _Path,
) -> _Path:
    """Write JSON exactly once; an existing path is always an error."""

    destination = _Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite JSON: {destination}.")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"JSON directory does not exist: {destination.parent}.")
    with destination.open("x", encoding="utf-8") as stream:
        _json.dump(value, stream, indent=2)
        stream.write("\n")
    return destination


def require_explicit_opt_in(value: str | None) -> None:
    """Reject generation unless the deliberate near-miss token is supplied."""

    if value != OPT_IN_TOKEN:
        raise PermissionError(
            "This candidate failed the promotion gate. Supply the exact "
            f"opt-in token {OPT_IN_TOKEN!r} to generate the optional hedge."
        )


def require_sha256(path: _Path, expected: str) -> str:
    """Require a file to exist and match its pinned lowercase SHA-256."""

    path = _Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 changed for {path}: {actual}.")
    return actual


def sha256_file(path: _Path) -> str:
    """Return a streaming lowercase SHA-256 digest."""

    digest = _hashlib.sha256()
    with _Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence_deep_recipe(seed: int) -> dict[str, object]:
    recipe = canonical_deep_xgboost_recipe(seed)
    recipe.pop("training_scope")
    return recipe


def _validate_locked_weights() -> None:
    if dict(DEEP_ARCHIVE_WEIGHTS) != EXPECTED_ENSEMBLE_WEIGHTS:
        raise ValueError("Incumbent deep-archive weights changed.")
    if DEEP_ARCHIVE_WEIGHTS.get(DEEP_COMPONENT) != DEEP_COMPONENT_WEIGHT:
        raise ValueError("The incumbent deep-XGBoost weight is no longer 0.33.")


def _validate_cache_common(
    payload: dict[str, object],
    expected_ids: _pd.Series,
    *,
    name: str,
) -> None:
    validate_component_ids(payload, expected_ids, name=name)
    validate_probabilities(payload["probabilities"], len(expected_ids))
    seconds = payload["fit_and_predict_seconds"]
    if (
        not isinstance(seconds, (int, float))
        or not _np.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError(f"{name} fit timing is invalid.")


def _finite_float(value: object, context: str) -> float:
    if not isinstance(value, (int, float)) or not _np.isfinite(value):
        raise ValueError(f"{context} contains a non-finite metric.")
    return float(value)

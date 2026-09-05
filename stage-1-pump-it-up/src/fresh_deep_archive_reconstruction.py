"""Reconstruct and compare the locked deep archive on fresh full-data folds."""

from __future__ import annotations

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass
import hashlib as _hashlib
import json as _json
import math as _math
from numbers import Integral as _Integral
from numbers import Real as _Real
from typing import Any as _Any

import numpy as _np
import pandas as _pd

from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from deep_archive_confirmation import blend_deep_archive
from deep_xgboost_ten_seed_bag import TEN_BAG_NAME
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from model_evaluation import build_candidate_evaluation
from model_evaluation import CandidateEvaluation


FRESH_RECONSTRUCTION_CACHE_VERSION = 1
FRESH_LABELLED_ROWS = 59_400
FRESH_CROSS_VALIDATION_FOLDS = 5
FRESH_CROSS_VALIDATION_SEED = 20260905
FRESH_CROSS_VALIDATION_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)

ENSEMBLE_MEAN_ACCURACY_GAIN_GATE = 0.001
ENSEMBLE_FOLD_WIN_GATE = 3
ENSEMBLE_WORST_FOLD_DELTA_GATE = -0.0025
ENSEMBLE_REPAIR_RECALL_DELTA_GATE = -0.02
EXPLORATORY_MEAN_ACCURACY_GAIN_GUARD = 0.0
EXPLORATORY_FOLD_WIN_GUARD = 3
EXPLORATORY_WORST_FOLD_DELTA_GUARD = -0.001
EXPLORATORY_REPAIR_RECALL_DELTA_GUARD = -0.01
REPAIR_METRIC = "recall: functional needs repair"

CATBOOST_IDENTITY_GRID_BAG = "catboost_identity_grid_50_50_at_0_20"
RF_CURRENT_LEAF_1_BAG = "rf_current_features_0_3_50_50_at_0_18"
RF_CURRENT_LEAF_2_BAG = "rf_current_features_0_3_leaf_2_50_50_at_0_18"
DEEP_SEED_BAG = "deep_seed_20260824_20260905_50_50_at_0_33"
COMBINED_FALLBACK_CANDIDATE = "fresh_deep_archive_combined_fallback"
DEEP_TEN_SEED_DIRECT_REPLACEMENT = "deep_ten_seed_equal_bag_at_0_33"
LOCKED_FRESH_CANDIDATES = (
    CATBOOST_IDENTITY_GRID_BAG,
    RF_CURRENT_LEAF_1_BAG,
    RF_CURRENT_LEAF_2_BAG,
    DEEP_SEED_BAG,
)

SPATIAL_GRID_COMPONENT = "spatial_grid_catboost"
RF_FEATURES_0_3_COMPONENT = "rf_features_0_3"
RF_FEATURES_0_3_LEAF_2_COMPONENT = "rf_features_0_3_leaf_2"
DEEP_SEED_BAG_COMPONENT = "deep_seed_20260824_20260905_bag"
LOCKED_ALTERNATIVE_COMPONENT_KEYS = (
    SPATIAL_GRID_COMPONENT,
    RF_FEATURES_0_3_COMPONENT,
    RF_FEATURES_0_3_LEAF_2_COMPONENT,
    DEEP_SEED_BAG_COMPONENT,
)

_EXPECTED_DEEP_ARCHIVE_WEIGHTS = {
    "deep_xgboost": 0.33,
    "spatial_xgboost": 0.11,
    "random_forest": 0.18,
    "frequency_random_forest": 0.09,
    "spatial_random_forest": 0.09,
    "identity_catboost": 0.20,
}
_FOLD_PAYLOAD_KEYS = {
    "cache_version",
    "metadata",
    "validation_ids",
    "probabilities",
    "diagnostics",
    "content_sha256",
}
_REQUIRED_FOLD_METADATA = {
    "labelled_rows": FRESH_LABELLED_ROWS,
    "cross_validation_folds": FRESH_CROSS_VALIDATION_FOLDS,
    "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
    "cross_validation_fingerprint": FRESH_CROSS_VALIDATION_FINGERPRINT,
}


@_dataclass(frozen=True)
class _CandidateRecipe:
    candidate: str
    slot: str
    alternative: str
    slot_weight: float


_CANDIDATE_RECIPES = (
    _CandidateRecipe(
        CATBOOST_IDENTITY_GRID_BAG,
        "identity_catboost",
        SPATIAL_GRID_COMPONENT,
        0.20,
    ),
    _CandidateRecipe(
        RF_CURRENT_LEAF_1_BAG,
        "random_forest",
        RF_FEATURES_0_3_COMPONENT,
        0.18,
    ),
    _CandidateRecipe(
        RF_CURRENT_LEAF_2_BAG,
        "random_forest",
        RF_FEATURES_0_3_LEAF_2_COMPONENT,
        0.18,
    ),
    _CandidateRecipe(
        DEEP_SEED_BAG,
        "deep_xgboost",
        DEEP_SEED_BAG_COMPONENT,
        0.33,
    ),
)
LOCKED_FRESH_CANDIDATE_FAMILIES = {
    CATBOOST_IDENTITY_GRID_BAG: "catboost",
    RF_CURRENT_LEAF_1_BAG: "random_forest",
    RF_CURRENT_LEAF_2_BAG: "random_forest",
    DEEP_SEED_BAG: "deep_xgboost",
}


@_dataclass(frozen=True)
class FreshDeepArchiveComparison:
    """Incumbent and four one-factor candidates on one fresh fold plan."""

    incumbent_evaluation: CandidateEvaluation
    candidate_evaluations: dict[str, CandidateEvaluation]
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame


@_dataclass(frozen=True)
class FreshDeepArchiveFallbackComparison:
    """One combined fallback assembled from admissible distinct families."""

    selected_candidates: tuple[str, ...]
    selected_families: tuple[str, ...]
    incumbent_evaluation: CandidateEvaluation
    candidate_evaluation: CandidateEvaluation
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame


@_dataclass(frozen=True)
class FreshDeepTenSeedReplacementComparison:
    """Formal-only direct replacement of the locked 0.33 deep slot."""

    incumbent_evaluation: CandidateEvaluation
    candidate_evaluation: CandidateEvaluation
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame


def make_fold_cache_payload(
    metadata: _Mapping[str, object],
    validation_ids: object,
    probabilities: object,
    diagnostics: _Mapping[str, object],
) -> dict[str, object]:
    """Create one self-digesting, versioned fresh-fold cache payload."""

    metadata_copy = dict(metadata)
    identifiers = _normalise_ids(validation_ids)
    probability_values = _normalise_probabilities(probabilities, len(identifiers))
    if not isinstance(diagnostics, _Mapping):
        raise TypeError("Fold diagnostics must be a mapping.")
    diagnostics_copy = dict(diagnostics)
    _validate_fold_metadata(metadata_copy)
    _validate_diagnostics(diagnostics_copy, metadata_copy["validation_fold"])
    payload = {
        "cache_version": FRESH_RECONSTRUCTION_CACHE_VERSION,
        "metadata": metadata_copy,
        "validation_ids": identifiers,
        "probabilities": probability_values,
        "diagnostics": diagnostics_copy,
        "content_sha256": _content_sha256(
            metadata_copy,
            identifiers,
            probability_values,
            diagnostics_copy,
        ),
    }
    validate_fold_cache_payload(payload, metadata_copy, identifiers)
    return payload


def validate_fold_cache_payload(
    payload: object,
    expected_metadata: _Mapping[str, object],
    expected_ids: object,
) -> tuple[_np.ndarray, dict[str, object]]:
    """Validate exact metadata, row order, probabilities and content digest."""

    if not isinstance(payload, dict) or set(payload) != _FOLD_PAYLOAD_KEYS:
        raise ValueError("Fresh reconstruction fold-cache schema changed.")
    if payload["cache_version"] != FRESH_RECONSTRUCTION_CACHE_VERSION:
        raise ValueError("Fresh reconstruction fold-cache version changed.")
    expected_metadata_copy = dict(expected_metadata)
    _validate_fold_metadata(expected_metadata_copy)
    if payload["metadata"] != expected_metadata_copy:
        raise ValueError("Fresh reconstruction fold-cache metadata is stale.")

    identifiers = payload["validation_ids"]
    if not isinstance(identifiers, _np.ndarray) or identifiers.ndim != 1:
        raise ValueError("Fresh reconstruction validation IDs are malformed.")
    expected_identifiers = _normalise_ids(expected_ids)
    if not _np.array_equal(identifiers, expected_identifiers):
        raise ValueError(
            "Fresh reconstruction validation IDs changed or are misordered."
        )
    probabilities = payload["probabilities"]
    if not isinstance(probabilities, _np.ndarray):
        raise ValueError("Fresh reconstruction probabilities are malformed.")
    validate_probabilities(probabilities, len(expected_identifiers))
    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, dict):
        raise ValueError("Fresh reconstruction fold diagnostics are malformed.")
    _validate_diagnostics(
        diagnostics,
        expected_metadata_copy["validation_fold"],
    )
    expected_digest = _content_sha256(
        payload["metadata"],
        identifiers,
        probabilities,
        diagnostics,
    )
    if payload["content_sha256"] != expected_digest:
        raise ValueError("Fresh reconstruction fold-cache content was tampered with.")
    return probabilities.copy(), diagnostics.copy()


def assemble_fold_evaluation(
    partitioned: PartitionedData,
    payloads: _Mapping[int, object],
    metadata_by_fold: _Mapping[int, _Mapping[str, object]],
    model_name: str,
) -> CandidateEvaluation:
    """Stitch five strictly aligned fold caches into one replayable evaluation."""

    _validate_fresh_partition(partitioned)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Fresh reconstruction model name is empty.")
    expected_folds = set(range(1, FRESH_CROSS_VALIDATION_FOLDS + 1))
    if set(payloads) != expected_folds:
        raise ValueError("Fresh reconstruction fold payload set changed.")
    if set(metadata_by_fold) != expected_folds:
        raise ValueError("Fresh reconstruction fold metadata set changed.")

    probabilities = _np.full(
        (FRESH_LABELLED_ROWS, len(CLASS_LABELS)),
        _np.nan,
        dtype="float64",
    )
    diagnostic_rows: list[dict[str, object]] = []
    cross_validation = make_cross_validation(partitioned)
    for fold, (_, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        metadata = metadata_by_fold[fold]
        if metadata.get("validation_fold") != fold:
            raise ValueError("Fresh reconstruction metadata fold order changed.")
        expected_ids = partitioned.development_ids.iloc[
            validation_positions
        ].to_numpy()
        fold_probabilities, diagnostics = validate_fold_cache_payload(
            payloads[fold],
            metadata,
            expected_ids,
        )
        probabilities[validation_positions] = fold_probabilities
        diagnostic_rows.append({"validation_fold": fold, **diagnostics})
    validate_probabilities(probabilities, FRESH_LABELLED_ROWS)
    return build_candidate_evaluation(
        model_name=model_name,
        partitioned_data=partitioned,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=diagnostic_rows,
    )


def compare_fresh_deep_archive_candidates(
    partitioned: PartitionedData,
    incumbent_components: _Mapping[str, _np.ndarray],
    alternative_components: _Mapping[str, _np.ndarray],
    *,
    component_evidence_eligibility: _Mapping[str, bool] | None = None,
) -> FreshDeepArchiveComparison:
    """Compare exactly four 50:50 one-slot bags at unchanged archive weights."""

    _validate_fresh_partition(partitioned)
    _validate_locked_candidate_contract()
    incumbent = _validate_component_mapping(
        incumbent_components,
        tuple(_EXPECTED_DEEP_ARCHIVE_WEIGHTS),
        name="incumbent",
    )
    alternatives = _validate_component_mapping(
        alternative_components,
        LOCKED_ALTERNATIVE_COMPONENT_KEYS,
        name="alternative",
    )
    eligibility = _validate_component_evidence_eligibility(
        component_evidence_eligibility
    )
    cross_validation = make_cross_validation(partitioned)
    incumbent_probabilities = blend_deep_archive(incumbent)
    incumbent_evaluation = _build_evaluation(
        "fresh seed-20260824 deep archive",
        partitioned,
        cross_validation,
        incumbent_probabilities,
    )

    evaluations: dict[str, CandidateEvaluation] = {}
    fold_frames = []
    summary_rows = []
    for recipe in _CANDIDATE_RECIPES:
        if recipe.alternative == DEEP_SEED_BAG_COMPONENT:
            replacement = alternatives[recipe.alternative]
        else:
            replacement = (
                0.5 * incumbent[recipe.slot]
                + 0.5 * alternatives[recipe.alternative]
            )
            validate_probabilities(replacement, FRESH_LABELLED_ROWS)
        candidate_components = {**incumbent, recipe.slot: replacement}
        candidate_probabilities = blend_deep_archive(candidate_components)
        evaluation = _build_evaluation(
            recipe.candidate,
            partitioned,
            cross_validation,
            candidate_probabilities,
        )
        evaluations[recipe.candidate] = evaluation
        folds = _paired_fold_deltas(
            partitioned,
            incumbent_evaluation,
            evaluation,
            recipe.candidate,
        )
        fold_frames.append(folds)
        summary_rows.append(
            _summarise_candidate(
                incumbent_evaluation,
                evaluation,
                folds,
                recipe,
                eligibility[recipe.candidate],
            )
        )
    if tuple(evaluations) != LOCKED_FRESH_CANDIDATES:
        raise ValueError("Fresh reconstruction candidate order changed.")
    return FreshDeepArchiveComparison(
        incumbent_evaluation=incumbent_evaluation,
        candidate_evaluations=evaluations,
        summary=_pd.DataFrame(summary_rows).set_index("candidate"),
        fold_deltas=_pd.concat(fold_frames, ignore_index=True),
    )


def combine_fresh_deep_archive_fallback(
    partitioned: PartitionedData,
    comparison: FreshDeepArchiveComparison,
    selected_candidates: object,
) -> FreshDeepArchiveFallbackComparison:
    """Build one additive fallback from two or three admissible families."""

    _validate_fresh_partition(partitioned)
    _validate_locked_candidate_contract()
    admissible, eligibility = _validate_comparison_for_fallback(
        partitioned,
        comparison,
    )
    if isinstance(selected_candidates, str):
        raise TypeError("Fallback selections must be a sequence of candidates.")
    try:
        supplied = tuple(selected_candidates)
    except TypeError as error:
        raise TypeError(
            "Fallback selections must be a sequence of candidates."
        ) from error
    if len(supplied) not in {2, 3}:
        raise ValueError("Fallback requires exactly two or three candidates.")
    if len(set(supplied)) != len(supplied):
        raise ValueError("Fallback selections contain a duplicate candidate.")
    unknown = set(supplied).difference(LOCKED_FRESH_CANDIDATES)
    if unknown:
        raise ValueError(f"Fallback selection is unknown: {sorted(unknown)!r}.")
    selected = tuple(
        candidate for candidate in LOCKED_FRESH_CANDIDATES
        if candidate in supplied
    )
    families = tuple(
        LOCKED_FRESH_CANDIDATE_FAMILIES[candidate]
        for candidate in selected
    )
    if len(set(families)) != len(families):
        raise ValueError("Fallback selections contain a duplicate model family.")
    rejected = [candidate for candidate in selected if not admissible[candidate]]
    if rejected:
        raise ValueError(
            f"Fallback selections are not admissible: {rejected!r}."
        )

    incumbent = comparison.incumbent_evaluation
    incumbent_probabilities = incumbent.out_of_fold_probabilities.to_numpy(
        dtype="float64"
    )
    combined_probabilities = incumbent_probabilities.copy()
    for candidate in selected:
        candidate_probabilities = comparison.candidate_evaluations[
            candidate
        ].out_of_fold_probabilities.to_numpy(dtype="float64")
        combined_probabilities += (
            candidate_probabilities - incumbent_probabilities
        )
    validate_probabilities(combined_probabilities, FRESH_LABELLED_ROWS)
    cross_validation = make_cross_validation(partitioned)
    combined = _build_evaluation(
        COMBINED_FALLBACK_CANDIDATE,
        partitioned,
        cross_validation,
        combined_probabilities,
    )
    folds = _paired_fold_deltas(
        partitioned,
        incumbent,
        combined,
        COMBINED_FALLBACK_CANDIDATE,
    )
    summary = _summarise_combined_fallback(
        incumbent,
        combined,
        folds,
        selected,
        families,
        all(eligibility[candidate] for candidate in selected),
    )
    return FreshDeepArchiveFallbackComparison(
        selected_candidates=selected,
        selected_families=families,
        incumbent_evaluation=incumbent,
        candidate_evaluation=combined,
        summary=_pd.DataFrame([summary]).set_index("candidate"),
        fold_deltas=folds,
    )


def compare_fresh_deep_ten_seed_direct_replacement(
    partitioned: PartitionedData,
    incumbent_components: _Mapping[str, _np.ndarray],
    ten_seed_bag: CandidateEvaluation,
) -> FreshDeepTenSeedReplacementComparison:
    """Replace only the 0.33 deep slot with a validated equal ten-seed bag."""

    _validate_fresh_partition(partitioned)
    _validate_locked_candidate_contract()
    incumbent_components_copy = _validate_component_mapping(
        incumbent_components,
        tuple(_EXPECTED_DEEP_ARCHIVE_WEIGHTS),
        name="incumbent",
    )
    ten_seed = _recompute_and_validate_evaluation(
        partitioned,
        ten_seed_bag,
        name="ten-seed deep bag",
    )
    if ten_seed.model_name != TEN_BAG_NAME:
        raise ValueError("Ten-seed deep-bag model identity changed.")
    ten_seed_probabilities = ten_seed.out_of_fold_probabilities.to_numpy(
        dtype="float64"
    )
    incumbent_probabilities = blend_deep_archive(
        incumbent_components_copy
    )
    replacement_probabilities = blend_deep_archive(
        {
            **incumbent_components_copy,
            "deep_xgboost": ten_seed_probabilities,
        }
    )
    cross_validation = make_cross_validation(partitioned)
    incumbent = _build_evaluation(
        "fresh seed-20260824 deep archive",
        partitioned,
        cross_validation,
        incumbent_probabilities,
    )
    candidate = _build_evaluation(
        DEEP_TEN_SEED_DIRECT_REPLACEMENT,
        partitioned,
        cross_validation,
        replacement_probabilities,
    )
    folds = _paired_fold_deltas(
        partitioned,
        incumbent,
        candidate,
        DEEP_TEN_SEED_DIRECT_REPLACEMENT,
    )
    summary = _summarise_ten_seed_replacement(
        incumbent,
        candidate,
        folds,
    )
    return FreshDeepTenSeedReplacementComparison(
        incumbent_evaluation=incumbent,
        candidate_evaluation=candidate,
        summary=_pd.DataFrame([summary]).set_index("candidate"),
        fold_deltas=folds,
    )


def passes_fresh_deep_archive_gate(
    *,
    mean_accuracy_delta: float,
    fold_wins: int,
    worst_fold_delta: float,
    repair_recall_delta: float,
) -> bool:
    """Apply the predeclared exact-ensemble gate to a fresh-fold candidate."""

    values = (
        mean_accuracy_delta,
        worst_fold_delta,
        repair_recall_delta,
    )
    if any(
        not isinstance(value, _Real) or not _math.isfinite(value)
        for value in values
    ):
        raise ValueError("Fresh reconstruction gate inputs must be finite numbers.")
    if isinstance(fold_wins, bool) or not isinstance(fold_wins, _Integral):
        raise TypeError("Fresh reconstruction fold wins must be an integer.")
    return (
        _at_least(mean_accuracy_delta, ENSEMBLE_MEAN_ACCURACY_GAIN_GATE)
        and fold_wins >= ENSEMBLE_FOLD_WIN_GATE
        and _at_least(worst_fold_delta, ENSEMBLE_WORST_FOLD_DELTA_GATE)
        and _at_least(
            repair_recall_delta,
            ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
        )
    )


def passes_fresh_deep_archive_exploratory_guard(
    *,
    mean_accuracy_delta: float,
    fold_wins: int,
    worst_fold_delta: float,
    repair_recall_delta: float,
    component_evidence_eligible: bool,
) -> bool:
    """Apply the stricter stability guard to any positive exploratory gain."""

    values = (
        mean_accuracy_delta,
        worst_fold_delta,
        repair_recall_delta,
    )
    if any(
        not isinstance(value, _Real) or not _math.isfinite(value)
        for value in values
    ):
        raise ValueError("Fresh reconstruction guard inputs must be finite numbers.")
    if isinstance(fold_wins, bool) or not isinstance(fold_wins, _Integral):
        raise TypeError("Fresh reconstruction fold wins must be an integer.")
    if not isinstance(component_evidence_eligible, bool):
        raise TypeError("Component evidence eligibility must be Boolean.")
    return (
        component_evidence_eligible
        and mean_accuracy_delta > EXPLORATORY_MEAN_ACCURACY_GAIN_GUARD
        and fold_wins >= EXPLORATORY_FOLD_WIN_GUARD
        and _at_least(
            worst_fold_delta,
            EXPLORATORY_WORST_FOLD_DELTA_GUARD,
        )
        and _at_least(
            repair_recall_delta,
            EXPLORATORY_REPAIR_RECALL_DELTA_GUARD,
        )
    )


def _validate_fold_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise TypeError("Fresh reconstruction fold metadata must be a dictionary.")
    for key, expected in _REQUIRED_FOLD_METADATA.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Fresh reconstruction fold metadata {key!r} changed."
            )
    candidate = metadata.get("candidate")
    if not isinstance(candidate, str) or not candidate:
        raise ValueError("Fresh reconstruction fold candidate is missing.")
    recipe = metadata.get("recipe")
    if not isinstance(recipe, dict) or not recipe:
        raise ValueError("Fresh reconstruction fold recipe is missing.")
    source_hashes = metadata.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Fresh reconstruction source hashes are missing.")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for name, digest in source_hashes.items()
    ):
        raise ValueError("Fresh reconstruction source hashes are malformed.")
    fold = metadata.get("validation_fold")
    if (
        isinstance(fold, bool)
        or not isinstance(fold, _Integral)
        or fold not in range(1, FRESH_CROSS_VALIDATION_FOLDS + 1)
    ):
        raise ValueError("Fresh reconstruction validation fold is invalid.")
    _canonical_json(metadata)


def _validate_diagnostics(diagnostics: dict[str, object], fold: int) -> None:
    if "validation_fold" in diagnostics and diagnostics["validation_fold"] != fold:
        raise ValueError("Fresh reconstruction diagnostic fold changed.")
    _validate_finite_values(diagnostics, path="diagnostics")
    _canonical_json(diagnostics)


def _validate_finite_values(value: object, *, path: str) -> None:
    if isinstance(value, _Mapping):
        for key, nested in value.items():
            _validate_finite_values(nested, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _validate_finite_values(nested, path=f"{path}[{index}]")
    elif isinstance(value, _Real) and not _math.isfinite(value):
        raise ValueError(f"Fresh reconstruction {path} is not finite.")


def _validate_fresh_partition(partitioned: PartitionedData) -> None:
    if len(partitioned.y_development) != FRESH_LABELLED_ROWS:
        raise ValueError("Fresh reconstruction labelled-row count changed.")
    if not (
        len(partitioned.development_ids)
        == len(partitioned.X_development)
        == len(partitioned.y_development)
        == len(partitioned.validation_folds)
    ):
        raise ValueError("Fresh reconstruction partition rows are misaligned.")
    if partitioned.development_ids.duplicated().any():
        raise ValueError("Fresh reconstruction development IDs are duplicated.")
    if (
        partitioned.cross_validation_fingerprint
        != FRESH_CROSS_VALIDATION_FINGERPRINT
    ):
        raise ValueError("Fresh reconstruction fold fingerprint changed.")
    if set(partitioned.validation_folds) != set(
        range(1, FRESH_CROSS_VALIDATION_FOLDS + 1)
    ):
        raise ValueError("Fresh reconstruction fold assignment changed.")
    if set(partitioned.y_development) != set(CLASS_LABELS):
        raise ValueError("Fresh reconstruction target classes changed.")


def _validate_locked_candidate_contract() -> None:
    if DEEP_ARCHIVE_WEIGHTS != _EXPECTED_DEEP_ARCHIVE_WEIGHTS:
        raise ValueError("Locked deep-archive weights changed.")
    if tuple(recipe.candidate for recipe in _CANDIDATE_RECIPES) != (
        LOCKED_FRESH_CANDIDATES
    ):
        raise ValueError("Locked fresh candidate set changed.")
    if tuple(recipe.alternative for recipe in _CANDIDATE_RECIPES) != (
        LOCKED_ALTERNATIVE_COMPONENT_KEYS
    ):
        raise ValueError("Locked alternative component set changed.")
    for recipe in _CANDIDATE_RECIPES:
        if recipe.slot_weight != _EXPECTED_DEEP_ARCHIVE_WEIGHTS[recipe.slot]:
            raise ValueError("Locked fresh candidate slot weight changed.")


def _validate_component_mapping(
    components: _Mapping[str, _np.ndarray],
    expected_keys: tuple[str, ...],
    *,
    name: str,
) -> dict[str, _np.ndarray]:
    if not isinstance(components, _Mapping) or set(components) != set(expected_keys):
        actual = set(components) if isinstance(components, _Mapping) else set()
        expected = set(expected_keys)
        raise ValueError(
            f"Fresh {name} component set changed; "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}."
        )
    validated = {}
    for key in expected_keys:
        probabilities = components[key]
        if not isinstance(probabilities, _np.ndarray):
            raise ValueError(f"Fresh {name} component {key!r} is not an array.")
        validate_probabilities(probabilities, FRESH_LABELLED_ROWS)
        validated[key] = probabilities.astype("float64", copy=False)
    return validated


def _validate_component_evidence_eligibility(
    eligibility: _Mapping[str, bool] | None,
) -> dict[str, bool]:
    if eligibility is None:
        return {candidate: True for candidate in LOCKED_FRESH_CANDIDATES}
    if not isinstance(eligibility, _Mapping) or set(eligibility) != set(
        LOCKED_FRESH_CANDIDATES
    ):
        raise ValueError("Fresh component evidence eligibility set changed.")
    if any(not isinstance(value, bool) for value in eligibility.values()):
        raise TypeError("Fresh component evidence eligibility must be Boolean.")
    return {
        candidate: eligibility[candidate]
        for candidate in LOCKED_FRESH_CANDIDATES
    }


def _validate_comparison_for_fallback(
    partitioned: PartitionedData,
    comparison: object,
) -> tuple[dict[str, bool], dict[str, bool]]:
    if not isinstance(comparison, FreshDeepArchiveComparison):
        raise TypeError("Fallback requires a FreshDeepArchiveComparison.")
    if tuple(comparison.candidate_evaluations) != LOCKED_FRESH_CANDIDATES:
        raise ValueError("Fallback comparison candidate set changed.")
    if comparison.summary.index.tolist() != list(LOCKED_FRESH_CANDIDATES):
        raise ValueError("Fallback comparison summary candidate set changed.")
    required_summary_columns = {
        "mean_accuracy_delta",
        "fold_wins",
        "worst_fold_delta",
        "repair_recall_delta",
        "component_evidence_eligible",
        "passes_formal_gate",
        "passes_exploratory_guard",
    }
    missing_columns = required_summary_columns.difference(
        comparison.summary.columns
    )
    if missing_columns:
        raise ValueError(
            "Fallback comparison summary schema changed; "
            f"missing={sorted(missing_columns)!r}."
        )
    incumbent = _recompute_and_validate_evaluation(
        partitioned,
        comparison.incumbent_evaluation,
        name="fallback incumbent",
    )
    recipe_by_candidate = {
        recipe.candidate: recipe for recipe in _CANDIDATE_RECIPES
    }
    admissible = {}
    eligibility = {}
    for candidate in LOCKED_FRESH_CANDIDATES:
        evaluation = _recompute_and_validate_evaluation(
            partitioned,
            comparison.candidate_evaluations[candidate],
            name=f"fallback {candidate}",
        )
        evidence_eligible = comparison.summary.loc[
            candidate, "component_evidence_eligible"
        ]
        if not isinstance(evidence_eligible, (bool, _np.bool_)):
            raise TypeError("Fallback component evidence eligibility changed.")
        evidence_eligible = bool(evidence_eligible)
        folds = _paired_fold_deltas(
            partitioned,
            incumbent,
            evaluation,
            candidate,
        )
        expected = _summarise_candidate(
            incumbent,
            evaluation,
            folds,
            recipe_by_candidate[candidate],
            evidence_eligible,
        )
        for field in (
            "mean_accuracy_delta",
            "worst_fold_delta",
            "repair_recall_delta",
        ):
            if not _np.isclose(
                comparison.summary.loc[candidate, field],
                expected[field],
                rtol=0.0,
                atol=1e-15,
            ):
                raise ValueError(
                    f"Fallback comparison metric changed for {candidate!r}."
                )
        for field in (
            "fold_wins",
            "passes_formal_gate",
            "passes_exploratory_guard",
        ):
            if comparison.summary.loc[candidate, field] != expected[field]:
                raise ValueError(
                    f"Fallback comparison gate changed for {candidate!r}."
                )
        admissible[candidate] = bool(
            expected["passes_formal_gate"]
            or expected["passes_exploratory_guard"]
        )
        eligibility[candidate] = evidence_eligible
    return admissible, eligibility


def _recompute_and_validate_evaluation(
    partitioned: PartitionedData,
    evaluation: object,
    *,
    name: str,
) -> CandidateEvaluation:
    if not isinstance(evaluation, CandidateEvaluation):
        raise TypeError(f"{name} is not a CandidateEvaluation.")
    if evaluation.cross_validation_fingerprint != FRESH_CROSS_VALIDATION_FINGERPRINT:
        raise ValueError(f"{name} fold fingerprint changed.")
    probabilities = evaluation.out_of_fold_probabilities
    if not probabilities.index.equals(partitioned.y_development.index):
        raise ValueError(f"{name} OOF row order changed.")
    if tuple(probabilities.columns) != tuple(CLASS_LABELS):
        raise ValueError(f"{name} OOF class order changed.")
    values = probabilities.to_numpy(dtype="float64")
    validate_probabilities(values, FRESH_LABELLED_ROWS)
    recomputed = _build_evaluation(
        evaluation.model_name,
        partitioned,
        make_cross_validation(partitioned),
        values,
    )
    for field in (
        "fold_metrics",
        "metric_summary",
        "confusion_counts",
        "confusion_recall",
    ):
        if not getattr(evaluation, field).equals(getattr(recomputed, field)):
            raise ValueError(f"{name} cached metrics changed.")
    return recomputed


def _build_evaluation(
    name: str,
    partitioned: PartitionedData,
    cross_validation: object,
    probabilities: _np.ndarray,
) -> CandidateEvaluation:
    validate_probabilities(probabilities, FRESH_LABELLED_ROWS)
    return build_candidate_evaluation(
        model_name=name,
        partitioned_data=partitioned,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=[
            {"validation_fold": fold}
            for fold in range(1, FRESH_CROSS_VALIDATION_FOLDS + 1)
        ],
    )


def _paired_fold_deltas(
    partitioned: PartitionedData,
    incumbent: CandidateEvaluation,
    candidate: CandidateEvaluation,
    candidate_name: str,
) -> _pd.DataFrame:
    incumbent_predictions = incumbent.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    candidate_predictions = candidate.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()
    rows = []
    for fold, (_, positions) in enumerate(
        make_cross_validation(partitioned).split(),
        start=1,
    ):
        actual = partitioned.y_development.iloc[positions].to_numpy()
        incumbent_correct = incumbent_predictions[positions] == actual
        candidate_correct = candidate_predictions[positions] == actual
        rows.append(
            {
                "candidate": candidate_name,
                "validation_fold": fold,
                "rows": len(positions),
                "incumbent_accuracy": float(incumbent_correct.mean()),
                "candidate_accuracy": float(candidate_correct.mean()),
                "accuracy_delta": float(
                    candidate_correct.mean() - incumbent_correct.mean()
                ),
                "incumbent_repair_recall": float(
                    incumbent.fold_metrics.loc[fold, REPAIR_METRIC]
                ),
                "candidate_repair_recall": float(
                    candidate.fold_metrics.loc[fold, REPAIR_METRIC]
                ),
                "repair_recall_delta": float(
                    candidate.fold_metrics.loc[fold, REPAIR_METRIC]
                    - incumbent.fold_metrics.loc[fold, REPAIR_METRIC]
                ),
                "prediction_disagreements": int(
                    (candidate_predictions[positions]
                     != incumbent_predictions[positions]).sum()
                ),
                "gained_correct": int(
                    (candidate_correct & ~incumbent_correct).sum()
                ),
                "lost_correct": int(
                    (~candidate_correct & incumbent_correct).sum()
                ),
                "net_additional_correct": int(
                    candidate_correct.sum() - incumbent_correct.sum()
                ),
            }
        )
    return _pd.DataFrame(rows)


def _summarise_candidate(
    incumbent: CandidateEvaluation,
    candidate: CandidateEvaluation,
    folds: _pd.DataFrame,
    recipe: _CandidateRecipe,
    component_evidence_eligible: bool,
) -> dict[str, object]:
    incumbent_accuracy = float(
        incumbent.metric_summary.loc["accuracy", "mean"]
    )
    candidate_accuracy = float(
        candidate.metric_summary.loc["accuracy", "mean"]
    )
    incumbent_repair = float(
        incumbent.metric_summary.loc[REPAIR_METRIC, "mean"]
    )
    candidate_repair = float(
        candidate.metric_summary.loc[REPAIR_METRIC, "mean"]
    )
    accuracy_delta = candidate_accuracy - incumbent_accuracy
    repair_delta = candidate_repair - incumbent_repair
    fold_wins = int(folds["accuracy_delta"].gt(0).sum())
    worst_fold = float(folds["accuracy_delta"].min())
    return {
        "candidate": recipe.candidate,
        "replaced_slot": recipe.slot,
        "alternative_component": recipe.alternative,
        "slot_weight": recipe.slot_weight,
        "within_slot_incumbent_weight": 0.5,
        "within_slot_alternative_weight": 0.5,
        "effective_incumbent_component_weight": 0.5 * recipe.slot_weight,
        "effective_alternative_component_weight": 0.5 * recipe.slot_weight,
        "incumbent_mean_accuracy": incumbent_accuracy,
        "candidate_mean_accuracy": candidate_accuracy,
        "mean_accuracy_delta": accuracy_delta,
        "fold_wins": fold_wins,
        "fold_losses": int(folds["accuracy_delta"].lt(0).sum()),
        "worst_fold_delta": worst_fold,
        "best_fold_delta": float(folds["accuracy_delta"].max()),
        "incumbent_repair_recall": incumbent_repair,
        "candidate_repair_recall": candidate_repair,
        "repair_recall_delta": repair_delta,
        "prediction_disagreements": int(
            folds["prediction_disagreements"].sum()
        ),
        "gained_correct": int(folds["gained_correct"].sum()),
        "lost_correct": int(folds["lost_correct"].sum()),
        "net_additional_correct": int(
            folds["net_additional_correct"].sum()
        ),
        "component_evidence_eligible": component_evidence_eligible,
        "passes_formal_gate": passes_fresh_deep_archive_gate(
            mean_accuracy_delta=accuracy_delta,
            fold_wins=fold_wins,
            worst_fold_delta=worst_fold,
            repair_recall_delta=repair_delta,
        ),
        "passes_exploratory_guard": (
            passes_fresh_deep_archive_exploratory_guard(
                mean_accuracy_delta=accuracy_delta,
                fold_wins=fold_wins,
                worst_fold_delta=worst_fold,
                repair_recall_delta=repair_delta,
                component_evidence_eligible=component_evidence_eligible,
            )
        ),
    }


def _summarise_combined_fallback(
    incumbent: CandidateEvaluation,
    candidate: CandidateEvaluation,
    folds: _pd.DataFrame,
    selected_candidates: tuple[str, ...],
    selected_families: tuple[str, ...],
    component_evidence_eligible: bool,
) -> dict[str, object]:
    incumbent_accuracy = float(
        incumbent.metric_summary.loc["accuracy", "mean"]
    )
    candidate_accuracy = float(
        candidate.metric_summary.loc["accuracy", "mean"]
    )
    incumbent_repair = float(
        incumbent.metric_summary.loc[REPAIR_METRIC, "mean"]
    )
    candidate_repair = float(
        candidate.metric_summary.loc[REPAIR_METRIC, "mean"]
    )
    accuracy_delta = candidate_accuracy - incumbent_accuracy
    repair_delta = candidate_repair - incumbent_repair
    fold_wins = int(folds["accuracy_delta"].gt(0).sum())
    worst_fold = float(folds["accuracy_delta"].min())
    return {
        "candidate": COMBINED_FALLBACK_CANDIDATE,
        "selected_candidates": list(selected_candidates),
        "selected_families": list(selected_families),
        "selection_count": len(selected_candidates),
        "component_evidence_eligible": component_evidence_eligible,
        "incumbent_mean_accuracy": incumbent_accuracy,
        "candidate_mean_accuracy": candidate_accuracy,
        "mean_accuracy_delta": accuracy_delta,
        "fold_wins": fold_wins,
        "fold_losses": int(folds["accuracy_delta"].lt(0).sum()),
        "worst_fold_delta": worst_fold,
        "best_fold_delta": float(folds["accuracy_delta"].max()),
        "incumbent_repair_recall": incumbent_repair,
        "candidate_repair_recall": candidate_repair,
        "repair_recall_delta": repair_delta,
        "prediction_disagreements": int(
            folds["prediction_disagreements"].sum()
        ),
        "gained_correct": int(folds["gained_correct"].sum()),
        "lost_correct": int(folds["lost_correct"].sum()),
        "net_additional_correct": int(
            folds["net_additional_correct"].sum()
        ),
        "passes_formal_gate": passes_fresh_deep_archive_gate(
            mean_accuracy_delta=accuracy_delta,
            fold_wins=fold_wins,
            worst_fold_delta=worst_fold,
            repair_recall_delta=repair_delta,
        ),
        "passes_exploratory_guard": (
            passes_fresh_deep_archive_exploratory_guard(
                mean_accuracy_delta=accuracy_delta,
                fold_wins=fold_wins,
                worst_fold_delta=worst_fold,
                repair_recall_delta=repair_delta,
                component_evidence_eligible=component_evidence_eligible,
            )
        ),
    }


def _summarise_ten_seed_replacement(
    incumbent: CandidateEvaluation,
    candidate: CandidateEvaluation,
    folds: _pd.DataFrame,
) -> dict[str, object]:
    incumbent_accuracy = float(
        incumbent.metric_summary.loc["accuracy", "mean"]
    )
    candidate_accuracy = float(
        candidate.metric_summary.loc["accuracy", "mean"]
    )
    incumbent_repair = float(
        incumbent.metric_summary.loc[REPAIR_METRIC, "mean"]
    )
    candidate_repair = float(
        candidate.metric_summary.loc[REPAIR_METRIC, "mean"]
    )
    accuracy_delta = candidate_accuracy - incumbent_accuracy
    repair_delta = candidate_repair - incumbent_repair
    fold_wins = int(folds["accuracy_delta"].gt(0).sum())
    worst_fold = float(folds["accuracy_delta"].min())
    return {
        "candidate": DEEP_TEN_SEED_DIRECT_REPLACEMENT,
        "replaced_slot": "deep_xgboost",
        "slot_weight": 0.33,
        "replacement_mode": "direct validated equal ten-seed bag",
        "component_evidence_eligible": False,
        "incumbent_mean_accuracy": incumbent_accuracy,
        "candidate_mean_accuracy": candidate_accuracy,
        "mean_accuracy_delta": accuracy_delta,
        "fold_wins": fold_wins,
        "fold_losses": int(folds["accuracy_delta"].lt(0).sum()),
        "worst_fold_delta": worst_fold,
        "best_fold_delta": float(folds["accuracy_delta"].max()),
        "incumbent_repair_recall": incumbent_repair,
        "candidate_repair_recall": candidate_repair,
        "repair_recall_delta": repair_delta,
        "prediction_disagreements": int(
            folds["prediction_disagreements"].sum()
        ),
        "gained_correct": int(folds["gained_correct"].sum()),
        "lost_correct": int(folds["lost_correct"].sum()),
        "net_additional_correct": int(
            folds["net_additional_correct"].sum()
        ),
        "passes_formal_gate": passes_fresh_deep_archive_gate(
            mean_accuracy_delta=accuracy_delta,
            fold_wins=fold_wins,
            worst_fold_delta=worst_fold,
            repair_recall_delta=repair_delta,
        ),
        "passes_exploratory_guard": False,
    }


def _normalise_ids(values: object) -> _np.ndarray:
    if isinstance(values, _pd.Series):
        identifiers = values.to_numpy(copy=True)
    else:
        identifiers = _np.asarray(values).copy()
    if identifiers.ndim != 1 or len(identifiers) == 0:
        raise ValueError("Fresh reconstruction validation IDs must be one-dimensional.")
    if _pd.Index(identifiers).has_duplicates:
        raise ValueError("Fresh reconstruction validation IDs are duplicated.")
    return identifiers


def _normalise_probabilities(values: object, expected_rows: int) -> _np.ndarray:
    probabilities = _np.asarray(values, dtype="float64").copy()
    validate_probabilities(probabilities, expected_rows)
    return probabilities


def _content_sha256(
    metadata: dict[str, object],
    identifiers: _np.ndarray,
    probabilities: _np.ndarray,
    diagnostics: dict[str, object],
) -> str:
    digest = _hashlib.sha256()
    for value in (
        _canonical_json(metadata),
        _canonical_json(
            {
                "dtype": identifiers.dtype.str,
                "shape": list(identifiers.shape),
                "values": identifiers.tolist(),
            }
        ),
        _canonical_json(diagnostics),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    contiguous = _np.ascontiguousarray(probabilities, dtype="float64")
    digest.update(_canonical_json(list(contiguous.shape)).encode("utf-8"))
    digest.update(b"\n")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return _json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_value(value: object) -> _Any:
    if isinstance(value, _np.generic):
        return value.item()
    if isinstance(value, _Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Fresh reconstruction cache mappings require string keys.")
        return {key: _json_value(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(nested) for nested in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Fresh reconstruction cache value is not JSON-compatible: {type(value)!r}."
    )


def _at_least(actual: float, threshold: float) -> bool:
    return bool(
        actual > threshold
        or _np.isclose(actual, threshold, rtol=0.0, atol=1e-12)
    )

"""Strict contracts for the fresh-reconstruction competition portfolio.

The portfolio contains only the three candidates admitted before competition
predictions are generated: the CatBoost within-slot bag, the Random Forest
within-slot bag, and their one predeclared combined candidate.  This module is
deliberately data-agnostic; importing and testing it never opens competition
data or fits a model.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import hashlib as _hashlib
import json as _json
import math as _math
from numbers import Integral as _Integral
from numbers import Real as _Real
from pathlib import Path as _Path
import struct as _struct
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd

from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from deep_archive_confirmation import blend_deep_archive
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from fresh_deep_archive_reconstruction import CATBOOST_IDENTITY_GRID_BAG
from fresh_deep_archive_reconstruction import COMBINED_FALLBACK_CANDIDATE
from fresh_deep_archive_reconstruction import DEEP_SEED_BAG
from fresh_deep_archive_reconstruction import DEEP_TEN_SEED_DIRECT_REPLACEMENT
from fresh_deep_archive_reconstruction import LOCKED_FRESH_CANDIDATES
from fresh_deep_archive_reconstruction import RF_CURRENT_LEAF_1_BAG
from fresh_deep_archive_reconstruction import RF_CURRENT_LEAF_2_BAG
from fresh_deep_archive_reconstruction import (
    passes_fresh_deep_archive_exploratory_guard,
)
from fresh_deep_archive_reconstruction import passes_fresh_deep_archive_gate


COMPETITION_ROWS = 14_850
FRESH_LABELLED_ROWS = 59_400
FRESH_CROSS_VALIDATION_SEED = 20260905
FRESH_CROSS_VALIDATION_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)
OPT_IN_TOKEN = "generate-admissible-fresh-reconstruction-portfolio"
PORTFOLIO_STATUS = "fresh_reconstruction_exploratory_portfolio"

CATBOOST_FAMILY = "catboost"
RF_FAMILY = "random_forest"
DEEP_FAMILY = "deep_xgboost"
CATBOOST_SLOT = "identity_catboost"
RF_SLOT = "random_forest"
CATBOOST_ALTERNATIVE = "spatial_grid_catboost"
RF_ALTERNATIVE = "rf_features_0_3"

PORTFOLIO_CANDIDATES = (
    COMBINED_FALLBACK_CANDIDATE,
    CATBOOST_IDENTITY_GRID_BAG,
    RF_CURRENT_LEAF_1_BAG,
)
PORTFOLIO_FILENAMES = {
    COMBINED_FALLBACK_CANDIDATE: (
        "01-fresh-catboost-rf-combined-50-50-within-slot-bags.csv"
    ),
    CATBOOST_IDENTITY_GRID_BAG: (
        "02-fresh-catboost-identity-grid-50-50-within-slot-bag.csv"
    ),
    RF_CURRENT_LEAF_1_BAG: (
        "03-fresh-rf-current-features-0-3-50-50-within-slot-bag.csv"
    ),
}

_EXPECTED_WEIGHTS = {
    "deep_xgboost": 0.33,
    "spatial_xgboost": 0.11,
    "random_forest": 0.18,
    "frequency_random_forest": 0.09,
    "spatial_random_forest": 0.09,
    "identity_catboost": 0.20,
}
_BASE_CANDIDATE_CONTRACTS = {
    CATBOOST_IDENTITY_GRID_BAG: {
        "replaced_slot": CATBOOST_SLOT,
        "alternative_component": CATBOOST_ALTERNATIVE,
        "slot_weight": 0.20,
        "within_slot_incumbent_weight": 0.5,
        "within_slot_alternative_weight": 0.5,
        "effective_incumbent_component_weight": 0.10,
        "effective_alternative_component_weight": 0.10,
        "component_evidence_eligible": True,
    },
    RF_CURRENT_LEAF_1_BAG: {
        "replaced_slot": RF_SLOT,
        "alternative_component": RF_ALTERNATIVE,
        "slot_weight": 0.18,
        "within_slot_incumbent_weight": 0.5,
        "within_slot_alternative_weight": 0.5,
        "effective_incumbent_component_weight": 0.09,
        "effective_alternative_component_weight": 0.09,
        "component_evidence_eligible": True,
    },
    RF_CURRENT_LEAF_2_BAG: {
        "replaced_slot": RF_SLOT,
        "alternative_component": "rf_features_0_3_leaf_2",
        "slot_weight": 0.18,
        "within_slot_incumbent_weight": 0.5,
        "within_slot_alternative_weight": 0.5,
        "effective_incumbent_component_weight": 0.09,
        "effective_alternative_component_weight": 0.09,
        "component_evidence_eligible": True,
    },
    DEEP_SEED_BAG: {
        "replaced_slot": "deep_xgboost",
        "alternative_component": "deep_seed_20260824_20260905_bag",
        "slot_weight": 0.33,
        "within_slot_incumbent_weight": 0.5,
        "within_slot_alternative_weight": 0.5,
        "effective_incumbent_component_weight": 0.165,
        "effective_alternative_component_weight": 0.165,
        "component_evidence_eligible": True,
    },
}
_METRIC_FIELDS = (
    "incumbent_mean_accuracy",
    "candidate_mean_accuracy",
    "mean_accuracy_delta",
    "worst_fold_delta",
    "best_fold_delta",
    "incumbent_repair_recall",
    "candidate_repair_recall",
    "repair_recall_delta",
)
_OUTPUT_FILES = {
    "candidate-summary.csv",
    "fold-paired-deltas.csv",
    "combined-catboost-rf-leaf-1-summary.csv",
    "combined-catboost-rf-leaf-1-fold-paired-deltas.csv",
    "ten-seed-direct-replacement-summary.csv",
    "ten-seed-direct-replacement-fold-paired-deltas.csv",
    "all-candidate-summary.csv",
    "all-fold-paired-deltas.csv",
}


@_dataclass(frozen=True)
class FreshPortfolioEvidence:
    """The exact three-candidate decision authorised by fresh OOF evidence."""

    base_rows: dict[str, dict[str, object]]
    combined_row: dict[str, object]
    evidence_tier: dict[str, str]


@_dataclass(frozen=True)
class FreshPortfolioProbabilities:
    """Incumbent and exact ordered submission-candidate probabilities."""

    incumbent: _np.ndarray
    candidates: dict[str, _np.ndarray]


@_dataclass(frozen=True)
class PortfolioDistinctness:
    """Hard-label disagreement counts for the fixed candidate slate."""

    vs_incumbent: dict[str, int]
    pairwise: dict[str, int]
    redundant_candidates: tuple[str, ...]
    redundant_pairs: tuple[str, ...]


def require_explicit_opt_in(value: str | None) -> None:
    """Require the exact deliberate token before competition work starts."""

    if value != OPT_IN_TOKEN:
        raise PermissionError(
            "Competition portfolio generation is disabled unless the exact "
            f"opt-in token {OPT_IN_TOKEN!r} is supplied."
        )


def validate_fresh_portfolio_evidence(
    result: object,
) -> FreshPortfolioEvidence:
    """Validate the fixed fresh-fold evidence and expose only the locked slate."""

    if not isinstance(result, dict):
        raise ValueError("Fresh reconstruction result must be a JSON object.")
    _require_exact_fields(
        result,
        {
            "screen",
            "status",
            "formal_passing_candidates",
            "exploratory_admissible_candidates",
            "admissible_candidates",
            "ten_seed_formal_only_append",
            "factorial_combination",
            "deep_archive_weights",
            "source_sha256",
            "evidence_sha256",
            "component_evidence",
            "protocol",
            "results",
            "output_sha256",
        },
        "fresh reconstruction result",
    )
    if result["screen"] != "fresh exact deep-archive one-factor reconstruction":
        raise ValueError("Fresh reconstruction screen name changed.")
    if result["status"] != "comparison_complete":
        raise ValueError("Fresh reconstruction is not complete.")
    if result["deep_archive_weights"] != _EXPECTED_WEIGHTS:
        raise ValueError("Fresh reconstruction archive weights changed.")
    _validate_protocol(result["protocol"])
    if result["source_sha256"] != {
        "TrainingSetValues.csv": (
            "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
        ),
        "TrainingSetLabels.csv": (
            "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
        ),
    }:
        raise ValueError("Fresh reconstruction labelled-data hashes changed.")
    _validate_digest_mapping(result["source_sha256"], "source evidence")
    _validate_nested_digests(result["evidence_sha256"], "component evidence")
    _validate_digest_mapping(result["output_sha256"], "reconstruction output")
    if set(result["output_sha256"]) != _OUTPUT_FILES:
        raise ValueError("Fresh reconstruction output file set changed.")
    _validate_component_evidence(result["component_evidence"])

    rows = result["results"]
    expected_result_candidates = (
        *LOCKED_FRESH_CANDIDATES,
        DEEP_TEN_SEED_DIRECT_REPLACEMENT,
        COMBINED_FALLBACK_CANDIDATE,
    )
    if not isinstance(rows, list) or len(rows) != len(expected_result_candidates):
        raise ValueError("Fresh reconstruction candidate count changed.")
    by_candidate: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Fresh reconstruction candidate row is malformed.")
        candidate = row.get("candidate")
        if not isinstance(candidate, str) or candidate in by_candidate:
            raise ValueError("Fresh reconstruction candidate names are malformed.")
        by_candidate[candidate] = row
    if tuple(by_candidate) != expected_result_candidates:
        raise ValueError("Fresh reconstruction candidate order or set changed.")

    formal = []
    exploratory = []
    for candidate in LOCKED_FRESH_CANDIDATES:
        row = by_candidate[candidate]
        _validate_base_candidate_row(
            row,
            candidate,
            _BASE_CANDIDATE_CONTRACTS[candidate],
        )
        if row["passes_formal_gate"]:
            formal.append(candidate)
        if row["passes_exploratory_guard"]:
            exploratory.append(candidate)
    ten_seed_row = _validate_ten_seed_result(
        result["ten_seed_formal_only_append"]
    )
    _validate_result_row_copy(
        by_candidate[DEEP_TEN_SEED_DIRECT_REPLACEMENT],
        ten_seed_row,
        DEEP_TEN_SEED_DIRECT_REPLACEMENT,
    )
    if ten_seed_row["passes_formal_gate"]:
        formal.append(DEEP_TEN_SEED_DIRECT_REPLACEMENT)
    if result["formal_passing_candidates"] != formal:
        raise ValueError("Fresh reconstruction formal-pass list changed.")
    if result["exploratory_admissible_candidates"] != exploratory:
        raise ValueError("Fresh reconstruction exploratory-pass list changed.")
    if formal:
        raise ValueError("The locked exploratory portfolio unexpectedly has a formal pass.")
    if exploratory != [CATBOOST_IDENTITY_GRID_BAG, RF_CURRENT_LEAF_1_BAG]:
        raise ValueError("The locked one-factor exploratory slate changed.")

    combined_row = _validate_combined_result(result["factorial_combination"])
    _validate_result_row_copy(
        by_candidate[COMBINED_FALLBACK_CANDIDATE],
        combined_row,
        COMBINED_FALLBACK_CANDIDATE,
    )
    if result["admissible_candidates"] != [
        CATBOOST_IDENTITY_GRID_BAG,
        RF_CURRENT_LEAF_1_BAG,
        COMBINED_FALLBACK_CANDIDATE,
    ]:
        raise ValueError("Fresh reconstruction final admissible slate changed.")
    return FreshPortfolioEvidence(
        base_rows={candidate: by_candidate[candidate] for candidate in exploratory},
        combined_row=combined_row,
        evidence_tier={candidate: "exploratory" for candidate in PORTFOLIO_CANDIDATES},
    )


def build_fresh_portfolio_probabilities(
    incumbent_components: _Mapping[str, _np.ndarray],
    *,
    spatial_grid_catboost: _np.ndarray,
    rf_features_0_3: _np.ndarray,
) -> FreshPortfolioProbabilities:
    """Build exactly two 50:50 slot bags and their additive combination."""

    if dict(DEEP_ARCHIVE_WEIGHTS) != _EXPECTED_WEIGHTS:
        raise ValueError("Incumbent deep-archive weights changed.")
    actual = set(incumbent_components)
    expected = set(_EXPECTED_WEIGHTS)
    if actual != expected:
        raise ValueError(
            "Incumbent component contract changed; "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}."
        )
    rows = len(spatial_grid_catboost)
    if rows <= 0 or len(rf_features_0_3) != rows:
        raise ValueError("Replacement component row counts do not align.")
    for name, probabilities in {
        **dict(incumbent_components),
        CATBOOST_ALTERNATIVE: spatial_grid_catboost,
        RF_ALTERNATIVE: rf_features_0_3,
    }.items():
        try:
            validate_probabilities(probabilities, rows)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid probabilities for {name!r}.") from error

    incumbent = blend_deep_archive(dict(incumbent_components))
    cat_slot = 0.5 * (
        _np.asarray(incumbent_components[CATBOOST_SLOT], dtype="float64")
        + _np.asarray(spatial_grid_catboost, dtype="float64")
    )
    rf_slot = 0.5 * (
        _np.asarray(incumbent_components[RF_SLOT], dtype="float64")
        + _np.asarray(rf_features_0_3, dtype="float64")
    )
    cat = blend_deep_archive(
        {**dict(incumbent_components), CATBOOST_SLOT: cat_slot}
    )
    rf = blend_deep_archive(
        {**dict(incumbent_components), RF_SLOT: rf_slot}
    )
    combined = blend_deep_archive(
        {
            **dict(incumbent_components),
            CATBOOST_SLOT: cat_slot,
            RF_SLOT: rf_slot,
        }
    )
    expected_cat = incumbent + 0.10 * (
        spatial_grid_catboost - incumbent_components[CATBOOST_SLOT]
    )
    expected_rf = incumbent + 0.09 * (
        rf_features_0_3 - incumbent_components[RF_SLOT]
    )
    expected_combined = expected_cat + expected_rf - incumbent
    for name, actual_values, expected_values in (
        (CATBOOST_IDENTITY_GRID_BAG, cat, expected_cat),
        (RF_CURRENT_LEAF_1_BAG, rf, expected_rf),
        (COMBINED_FALLBACK_CANDIDATE, combined, expected_combined),
    ):
        if not _np.allclose(actual_values, expected_values, rtol=0.0, atol=1e-14):
            raise ValueError(f"{name!r} changed a locked slot weight.")
        validate_probabilities(actual_values, rows)
    candidates = {
        COMBINED_FALLBACK_CANDIDATE: combined,
        CATBOOST_IDENTITY_GRID_BAG: cat,
        RF_CURRENT_LEAF_1_BAG: rf,
    }
    if tuple(candidates) != PORTFOLIO_CANDIDATES:
        raise ValueError("Fresh portfolio candidate order changed.")
    return FreshPortfolioProbabilities(incumbent=incumbent, candidates=candidates)


def analyse_portfolio_label_distinctness(
    incumbent_labels: object,
    candidate_labels: _Mapping[str, object],
) -> PortfolioDistinctness:
    """Measure material hard-label changes without using outcome labels."""

    if tuple(candidate_labels) != PORTFOLIO_CANDIDATES:
        raise ValueError("Fresh portfolio label-vector order or set changed.")
    incumbent = _normalise_label_vector(incumbent_labels, "incumbent")
    candidates = {
        candidate: _normalise_label_vector(candidate_labels[candidate], candidate)
        for candidate in PORTFOLIO_CANDIDATES
    }
    vs_incumbent = {
        candidate: int(_np.count_nonzero(labels != incumbent))
        for candidate, labels in candidates.items()
    }
    pairwise = {}
    for left_index, left in enumerate(PORTFOLIO_CANDIDATES):
        for right in PORTFOLIO_CANDIDATES[left_index + 1 :]:
            key = f"{left}__vs__{right}"
            pairwise[key] = int(
                _np.count_nonzero(candidates[left] != candidates[right])
            )
    return PortfolioDistinctness(
        vs_incumbent=vs_incumbent,
        pairwise=pairwise,
        redundant_candidates=tuple(
            candidate
            for candidate in PORTFOLIO_CANDIDATES
            if vs_incumbent[candidate] == 0
        ),
        redundant_pairs=tuple(
            pair for pair, disagreements in pairwise.items() if disagreements == 0
        ),
    )


def require_materially_distinct_portfolio(
    distinctness: PortfolioDistinctness,
) -> None:
    """Fail closed when a candidate is immaterial or duplicates another."""

    if not isinstance(distinctness, PortfolioDistinctness):
        raise TypeError("Fresh portfolio distinctness evidence is malformed.")
    if distinctness.redundant_candidates or distinctness.redundant_pairs:
        raise ValueError(
            "Fresh portfolio contains redundant hard-label output; "
            f"vs_incumbent={list(distinctness.redundant_candidates)!r}, "
            f"pairwise={list(distinctness.redundant_pairs)!r}."
        )


def make_component_cache_metadata(
    *,
    candidate: str,
    recipe: _Mapping[str, object],
    data_sha256: _Mapping[str, str],
    source_sha256: _Mapping[str, str],
    evidence_sha256: _Mapping[str, str],
) -> dict[str, object]:
    """Return exact metadata for one optional full-data replacement cache."""

    if candidate not in {CATBOOST_ALTERNATIVE, RF_ALTERNATIVE}:
        raise ValueError(f"Unsupported replacement component: {candidate!r}.")
    if not isinstance(recipe, _Mapping) or not recipe:
        raise ValueError("Replacement recipe is empty.")
    for name, values in (
        ("data", data_sha256),
        ("source", source_sha256),
        ("evidence", evidence_sha256),
    ):
        _validate_digest_mapping(values, name)
    return {
        "cache_version": 1,
        "candidate": candidate,
        "recipe": dict(recipe),
        "training_rows": FRESH_LABELLED_ROWS,
        "competition_rows": COMPETITION_ROWS,
        "data_sha256": dict(data_sha256),
        "source_sha256": dict(source_sha256),
        "evidence_sha256": dict(evidence_sha256),
        "within_slot_use": {
            "incumbent_weight": 0.5,
            "alternative_weight": 0.5,
            "searched": False,
        },
    }


def validate_component_cache(
    payload: object,
    expected_metadata: _Mapping[str, object],
    expected_ids: _pd.Series,
) -> dict[str, object]:
    """Validate exact cache metadata, ID order, probabilities and timing."""

    required = {
        "cache_version",
        "metadata",
        "competition_ids",
        "probabilities",
        "fit_and_predict_seconds",
        "content_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Fresh portfolio component-cache schema changed.")
    if payload["cache_version"] != 1:
        raise ValueError("Fresh portfolio component-cache version changed.")
    if payload["metadata"] != dict(expected_metadata):
        raise ValueError("Fresh portfolio component-cache metadata is stale.")
    _validate_expected_ids(expected_ids)
    identifiers = payload["competition_ids"]
    if not isinstance(identifiers, _pd.Series) or not identifiers.reset_index(
        drop=True
    ).equals(expected_ids.reset_index(drop=True)):
        raise ValueError("Fresh portfolio component IDs changed or are misordered.")
    validate_probabilities(payload["probabilities"], COMPETITION_ROWS)
    seconds = payload["fit_and_predict_seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, _Real)
        or not _math.isfinite(seconds)
        or seconds <= 0
    ):
        raise ValueError("Fresh portfolio component timing is invalid.")
    expected_digest = component_cache_content_sha256(
        payload["metadata"],
        identifiers,
        payload["probabilities"],
        seconds,
    )
    if payload["content_sha256"] != expected_digest:
        raise ValueError("Fresh portfolio component-cache content was tampered with.")
    return payload


def make_component_cache_payload(
    metadata: _Mapping[str, object],
    competition_ids: _pd.Series,
    probabilities: _np.ndarray,
    fit_and_predict_seconds: float,
) -> dict[str, object]:
    """Build a self-digesting replacement-component cache payload."""

    payload = {
        "cache_version": 1,
        "metadata": dict(metadata),
        "competition_ids": competition_ids.copy(),
        "probabilities": _np.asarray(probabilities, dtype="float64").copy(),
        "fit_and_predict_seconds": fit_and_predict_seconds,
    }
    payload["content_sha256"] = component_cache_content_sha256(
        payload["metadata"],
        payload["competition_ids"],
        payload["probabilities"],
        fit_and_predict_seconds,
    )
    return payload


def component_cache_content_sha256(
    metadata: _Mapping[str, object],
    competition_ids: _pd.Series,
    probabilities: _np.ndarray,
    fit_and_predict_seconds: float,
) -> str:
    """Hash exact metadata, ID dtype/order, probability bytes and timing."""

    if not isinstance(metadata, _Mapping):
        raise TypeError("Fresh portfolio component metadata must be a mapping.")
    if not isinstance(competition_ids, _pd.Series):
        raise TypeError("Fresh portfolio component IDs must be a pandas Series.")
    identifier_values = competition_ids.to_numpy(copy=True)
    if identifier_values.dtype.hasobject:
        raise ValueError("Fresh portfolio component IDs must have a fixed dtype.")
    probability_values = _np.asarray(probabilities)
    if probability_values.dtype != _np.dtype("float64"):
        raise ValueError("Fresh portfolio probabilities must use exact float64 bytes.")
    if (
        isinstance(fit_and_predict_seconds, bool)
        or not isinstance(fit_and_predict_seconds, _Real)
        or not _math.isfinite(fit_and_predict_seconds)
        or fit_and_predict_seconds <= 0
    ):
        raise ValueError("Fresh portfolio component timing is invalid.")
    canonical_metadata = _json.dumps(
        dict(metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = _hashlib.sha256()
    for label, values in (
        (b"ids", identifier_values),
        (b"probabilities", probability_values),
    ):
        contiguous = _np.ascontiguousarray(values)
        digest.update(label)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(
            _json.dumps(list(contiguous.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(contiguous.tobytes(order="C"))
    digest.update(b"metadata")
    digest.update(canonical_metadata)
    digest.update(b"fit_and_predict_seconds")
    digest.update(_struct.pack("!d", float(fit_and_predict_seconds)))
    return digest.hexdigest()


def validate_submission_frame(
    submission: _pd.DataFrame,
    expected_ids: _pd.Series,
) -> None:
    """Require the exact 14,850 ordered IDs and all three legal labels."""

    _validate_expected_ids(expected_ids)
    if list(submission.columns) != ["id", "status_group"]:
        raise ValueError("Fresh portfolio submission columns changed.")
    if len(submission) != COMPETITION_ROWS:
        raise ValueError("Fresh portfolio submission row count changed.")
    if submission.isna().any().any() or submission["id"].duplicated().any():
        raise ValueError("Fresh portfolio submission has missing or duplicate rows.")
    if not submission["id"].reset_index(drop=True).equals(
        expected_ids.reset_index(drop=True)
    ):
        raise ValueError("Fresh portfolio submission IDs changed or are misordered.")
    if set(submission["status_group"]) != set(CLASS_LABELS):
        raise ValueError("Fresh portfolio submission must contain exactly three labels.")


def write_submission_without_overwrite(
    submission: _pd.DataFrame,
    expected_ids: _pd.Series,
    destination: _Path,
) -> _Path:
    """Write and reload one submission, refusing every existing path."""

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
        raise ValueError("Reloaded fresh portfolio submission changed.")
    return destination


def write_json_without_overwrite(
    value: _Mapping[str, object],
    destination: _Path,
) -> _Path:
    """Write JSON exactly once; even identical existing output is refused."""

    destination = _Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite JSON: {destination}.")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"JSON directory does not exist: {destination.parent}.")
    with destination.open("x", encoding="utf-8") as stream:
        _json.dump(value, stream, indent=2)
        stream.write("\n")
    return destination


def sha256_file(path: _Path) -> str:
    """Return the lowercase streaming SHA-256 of a file."""

    digest = _hashlib.sha256()
    with _Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_base_candidate_row(
    row: dict[str, object],
    candidate: str,
    contract: dict[str, object],
) -> None:
    if row.get("candidate") != candidate:
        raise ValueError(f"Fresh reconstruction candidate {candidate!r} changed.")
    for field, expected in contract.items():
        if row.get(field) != expected:
            raise ValueError(f"{candidate!r} recipe field {field!r} changed.")
    _validate_metric_fields(row, candidate)
    _validate_count_fields(row, candidate)
    expected_delta = row["candidate_mean_accuracy"] - row["incumbent_mean_accuracy"]
    expected_repair = row["candidate_repair_recall"] - row["incumbent_repair_recall"]
    _require_close(row["mean_accuracy_delta"], expected_delta, candidate)
    _require_close(row["repair_recall_delta"], expected_repair, candidate)
    if row["net_additional_correct"] != row["gained_correct"] - row["lost_correct"]:
        raise ValueError(f"{candidate!r} net-correct count is inconsistent.")
    if row["prediction_disagreements"] < (
        row["gained_correct"] + row["lost_correct"]
    ):
        raise ValueError(f"{candidate!r} disagreement count is inconsistent.")
    formal = passes_fresh_deep_archive_gate(
        mean_accuracy_delta=row["mean_accuracy_delta"],
        fold_wins=row["fold_wins"],
        worst_fold_delta=row["worst_fold_delta"],
        repair_recall_delta=row["repair_recall_delta"],
    )
    exploratory = passes_fresh_deep_archive_exploratory_guard(
        mean_accuracy_delta=row["mean_accuracy_delta"],
        fold_wins=row["fold_wins"],
        worst_fold_delta=row["worst_fold_delta"],
        repair_recall_delta=row["repair_recall_delta"],
        component_evidence_eligible=row["component_evidence_eligible"],
    )
    if row.get("passes_formal_gate") is not formal:
        raise ValueError(f"{candidate!r} formal gate flag changed.")
    if row.get("passes_exploratory_guard") is not exploratory:
        raise ValueError(f"{candidate!r} exploratory gate flag changed.")


def _validate_combined_result(combined_result: object) -> dict[str, object]:
    if not isinstance(combined_result, dict):
        raise ValueError("Combined-candidate result must be a JSON object.")
    _require_exact_fields(
        combined_result,
        {
            "candidate",
            "selected_candidates",
            "selected_families",
            "result",
            "fold_deltas",
        },
        "combined-candidate result",
    )
    if combined_result.get("candidate") != COMBINED_FALLBACK_CANDIDATE:
        raise ValueError("Combined evidence candidate name changed.")
    selected = combined_result.get("selected_candidates")
    if selected != [CATBOOST_IDENTITY_GRID_BAG, RF_CURRENT_LEAF_1_BAG]:
        raise ValueError("Combined evidence selected a different candidate set.")
    families = combined_result.get("selected_families")
    if families != [CATBOOST_FAMILY, RF_FAMILY]:
        raise ValueError("Combined evidence selected different model families.")
    if len(set(families)) != 2:
        raise ValueError("Combined evidence does not contain two orthogonal families.")
    row = combined_result.get("result")
    if not isinstance(row, dict):
        raise ValueError("Combined evidence result row changed.")
    if row.get("candidate") != COMBINED_FALLBACK_CANDIDATE:
        raise ValueError("Combined evidence candidate name changed.")
    if row.get("selected_candidates") != [
        CATBOOST_IDENTITY_GRID_BAG,
        RF_CURRENT_LEAF_1_BAG,
    ]:
        raise ValueError("Combined result row selected different candidates.")
    if row.get("selected_families") != [CATBOOST_FAMILY, RF_FAMILY]:
        raise ValueError("Combined result row selected different families.")
    if row.get("selection_count") != 2:
        raise ValueError("Combined result row selection count changed.")
    _validate_metric_fields(row, COMBINED_FALLBACK_CANDIDATE)
    _validate_count_fields(row, COMBINED_FALLBACK_CANDIDATE)
    _require_close(
        row["mean_accuracy_delta"],
        row["candidate_mean_accuracy"] - row["incumbent_mean_accuracy"],
        COMBINED_FALLBACK_CANDIDATE,
    )
    _require_close(
        row["repair_recall_delta"],
        row["candidate_repair_recall"] - row["incumbent_repair_recall"],
        COMBINED_FALLBACK_CANDIDATE,
    )
    if row["net_additional_correct"] != row["gained_correct"] - row["lost_correct"]:
        raise ValueError("Combined net-correct count is inconsistent.")
    if row.get("component_evidence_eligible") is not True:
        raise ValueError("Combined component evidence is ineligible.")
    formal = passes_fresh_deep_archive_gate(
        mean_accuracy_delta=row["mean_accuracy_delta"],
        fold_wins=row["fold_wins"],
        worst_fold_delta=row["worst_fold_delta"],
        repair_recall_delta=row["repair_recall_delta"],
    )
    exploratory = passes_fresh_deep_archive_exploratory_guard(
        mean_accuracy_delta=row["mean_accuracy_delta"],
        fold_wins=row["fold_wins"],
        worst_fold_delta=row["worst_fold_delta"],
        repair_recall_delta=row["repair_recall_delta"],
        component_evidence_eligible=True,
    )
    if row.get("passes_formal_gate") is not formal:
        raise ValueError("Combined formal gate flag changed.")
    if row.get("passes_exploratory_guard") is not exploratory:
        raise ValueError("Combined exploratory gate flag changed.")
    if not (formal or exploratory):
        raise ValueError("Combined candidate does not clear an admissibility gate.")
    _validate_fold_deltas(
        combined_result.get("fold_deltas"),
        row,
        COMBINED_FALLBACK_CANDIDATE,
    )
    return row


def _validate_ten_seed_result(value: object) -> dict[str, object]:
    """Enforce the ten-seed append's formal-only, non-promotable outcome."""

    if not isinstance(value, dict):
        raise ValueError("Ten-seed formal-only evidence is malformed.")
    _require_exact_fields(
        value,
        {
            "candidate",
            "component_evidence_eligible",
            "exploratory_consideration_permitted",
            "result",
            "fold_deltas",
        },
        "ten-seed formal-only result",
    )
    if value["candidate"] != DEEP_TEN_SEED_DIRECT_REPLACEMENT:
        raise ValueError("Ten-seed candidate name changed.")
    if value["component_evidence_eligible"] is not False:
        raise ValueError("Ten-seed component evidence unexpectedly became eligible.")
    if value["exploratory_consideration_permitted"] is not False:
        raise ValueError("Ten-seed exploratory consideration is forbidden.")
    row = value.get("result")
    if not isinstance(row, dict):
        raise ValueError("Ten-seed result row is malformed.")
    if row.get("candidate") != DEEP_TEN_SEED_DIRECT_REPLACEMENT:
        raise ValueError("Ten-seed result candidate changed.")
    if row.get("replaced_slot") != "deep_xgboost" or row.get("slot_weight") != 0.33:
        raise ValueError("Ten-seed direct-replacement recipe changed.")
    if row.get("replacement_mode") != "direct validated equal ten-seed bag":
        raise ValueError("Ten-seed replacement mode changed.")
    if row.get("component_evidence_eligible") is not False:
        raise ValueError("Ten-seed result became component-evidence eligible.")
    _validate_metric_fields(row, DEEP_TEN_SEED_DIRECT_REPLACEMENT)
    _validate_count_fields(row, DEEP_TEN_SEED_DIRECT_REPLACEMENT)
    _require_close(
        row["mean_accuracy_delta"],
        row["candidate_mean_accuracy"] - row["incumbent_mean_accuracy"],
        DEEP_TEN_SEED_DIRECT_REPLACEMENT,
    )
    _require_close(
        row["repair_recall_delta"],
        row["candidate_repair_recall"] - row["incumbent_repair_recall"],
        DEEP_TEN_SEED_DIRECT_REPLACEMENT,
    )
    formal = passes_fresh_deep_archive_gate(
        mean_accuracy_delta=row["mean_accuracy_delta"],
        fold_wins=row["fold_wins"],
        worst_fold_delta=row["worst_fold_delta"],
        repair_recall_delta=row["repair_recall_delta"],
    )
    if row.get("passes_formal_gate") is not formal:
        raise ValueError("Ten-seed formal gate flag changed.")
    if row.get("passes_exploratory_guard") is not False:
        raise ValueError("Ten-seed entered exploratory selection.")
    if formal:
        raise ValueError("The locked ten-seed candidate unexpectedly passed formally.")
    _validate_fold_deltas(
        value.get("fold_deltas"),
        row,
        DEEP_TEN_SEED_DIRECT_REPLACEMENT,
    )
    return row


def _validate_component_evidence(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "catboost_identity_grid_bag",
        "rf_probability_bags",
        "deep_two_seed_bag",
        "eligibility",
    }:
        raise ValueError("Fresh reconstruction component evidence changed.")
    if value["eligibility"] != {
        CATBOOST_IDENTITY_GRID_BAG: True,
        RF_CURRENT_LEAF_1_BAG: True,
        RF_CURRENT_LEAF_2_BAG: True,
        DEEP_SEED_BAG: True,
    }:
        raise ValueError("Fresh reconstruction component eligibility changed.")

    cat_rows = value["catboost_identity_grid_bag"]
    if not isinstance(cat_rows, list) or len(cat_rows) != 1:
        raise ValueError("CatBoost bag component evidence changed.")
    cat = cat_rows[0]
    if (
        not isinstance(cat, dict)
        or cat.get("candidate") != "complete_identity_spatial_grid_50_50_bag"
        or cat.get("passes_gate") is not True
        or _finite(cat.get("mean_accuracy_delta"), "CatBoost component") < 0.001
        or _integer(cat.get("fold_wins"), "CatBoost fold wins") < 3
        or _finite(cat.get("worst_fold_delta"), "CatBoost component") < -0.0025
        or _finite(cat.get("repair_recall_delta"), "CatBoost component") < -0.02
    ):
        raise ValueError("CatBoost bag component evidence is ineligible.")

    rf_rows = value["rf_probability_bags"]
    if not isinstance(rf_rows, list) or len(rf_rows) != 3:
        raise ValueError("RF bag component evidence changed.")
    rf_by_candidate = {
        row.get("candidate"): row for row in rf_rows if isinstance(row, dict)
    }
    if tuple(rf_by_candidate) != (
        "Random Forest features 0.3 leaf 2",
        "Random Forest features 0.3",
        "Current Random Forest",
    ):
        raise ValueError("RF bag component candidate set changed.")
    leaf_1 = rf_by_candidate["Random Forest features 0.3"]
    if (
        leaf_1.get("passes_gate") is not False
        or not 0.0 < _finite(leaf_1.get("mean_accuracy_delta"), "RF leaf-1") < 0.001
        or _integer(leaf_1.get("fold_wins_vs_incumbent"), "RF fold wins") < 3
        or _finite(leaf_1.get("worst_fold_delta"), "RF leaf-1") < -0.001
        or _finite(leaf_1.get("repair_recall_delta"), "RF leaf-1") < -0.01
    ):
        raise ValueError("RF leaf-1 stable-positive component evidence changed.")
    if rf_by_candidate["Random Forest features 0.3 leaf 2"].get(
        "passes_gate"
    ) is not True:
        raise ValueError("RF leaf-2 bag component gate changed.")
    if rf_by_candidate["Current Random Forest"].get("passes_gate") is not False:
        raise ValueError("RF incumbent component gate changed.")

    deep = value["deep_two_seed_bag"]
    if not isinstance(deep, dict) or {
        "passes_gate_against_both": deep.get("passes_gate_against_both"),
        "stable_positive_near_miss": deep.get("stable_positive_near_miss"),
        "verdict": deep.get("verdict"),
    } != {
        "passes_gate_against_both": False,
        "stable_positive_near_miss": True,
        "verdict": "stable_positive_near_miss",
    }:
        raise ValueError("Deep two-seed component evidence changed.")
    comparisons = deep.get("comparisons")
    if not isinstance(comparisons, list) or [
        row.get("comparison") for row in comparisons if isinstance(row, dict)
    ] != [
        "average_vs_seed_20260824",
        "average_vs_seed_20260905",
    ]:
        raise ValueError("Deep two-seed comparisons changed.")


def _validate_fold_deltas(
    value: object,
    summary: dict[str, object],
    candidate: str,
) -> None:
    if not isinstance(value, list) or len(value) != 5:
        raise ValueError(f"{candidate!r} fold evidence changed.")
    aggregate = {
        "prediction_disagreements": 0,
        "gained_correct": 0,
        "lost_correct": 0,
        "net_additional_correct": 0,
    }
    deltas = []
    for expected_fold, row in enumerate(value, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"{candidate!r} fold evidence is malformed.")
        if row.get("candidate") != candidate:
            raise ValueError(f"{candidate!r} fold candidate changed.")
        if row.get("validation_fold") != expected_fold or row.get("rows") != 11_880:
            raise ValueError(f"{candidate!r} fold membership changed.")
        for field in (
            "incumbent_accuracy",
            "candidate_accuracy",
            "accuracy_delta",
            "incumbent_repair_recall",
            "candidate_repair_recall",
            "repair_recall_delta",
        ):
            _finite(row.get(field), f"{candidate} fold {expected_fold}")
        if not _np.isclose(
            row["accuracy_delta"],
            row["candidate_accuracy"] - row["incumbent_accuracy"],
            rtol=0.0,
            atol=2e-10,
        ):
            raise ValueError(f"{candidate!r} fold accuracy delta is inconsistent.")
        if not _np.isclose(
            row["repair_recall_delta"],
            row["candidate_repair_recall"] - row["incumbent_repair_recall"],
            rtol=0.0,
            atol=2e-10,
        ):
            raise ValueError(f"{candidate!r} fold repair delta is inconsistent.")
        for field in aggregate:
            aggregate[field] += _integer(
                row.get(field),
                f"{candidate} fold {expected_fold} {field}",
            )
        if row["net_additional_correct"] != row["gained_correct"] - row["lost_correct"]:
            raise ValueError(f"{candidate!r} fold net count is inconsistent.")
        deltas.append(float(row["accuracy_delta"]))
    if sum(row["rows"] for row in value) != FRESH_LABELLED_ROWS:
        raise ValueError(f"{candidate!r} fold rows do not cover the labelled data.")
    for field, total in aggregate.items():
        if summary.get(field) != total:
            raise ValueError(f"{candidate!r} fold total {field!r} is inconsistent.")
    if summary["fold_wins"] != sum(delta > 0.0 for delta in deltas):
        raise ValueError(f"{candidate!r} fold-win total is inconsistent.")
    if summary["fold_losses"] != sum(delta < 0.0 for delta in deltas):
        raise ValueError(f"{candidate!r} fold-loss total is inconsistent.")
    if not _np.isclose(summary["worst_fold_delta"], min(deltas), atol=1e-10):
        raise ValueError(f"{candidate!r} worst fold is inconsistent.")
    if not _np.isclose(summary["best_fold_delta"], max(deltas), atol=1e-10):
        raise ValueError(f"{candidate!r} best fold is inconsistent.")


def _validate_result_row_copy(
    expanded: dict[str, object],
    nested: dict[str, object],
    candidate: str,
) -> None:
    for field, expected in nested.items():
        if field not in expanded:
            raise ValueError(f"{candidate!r} is missing from the combined result table.")
        actual = expanded[field]
        if isinstance(expected, _Real) and not isinstance(expected, bool):
            if not _np.isclose(actual, expected, rtol=0.0, atol=1e-12):
                raise ValueError(f"{candidate!r} result-table metric changed.")
        elif actual != expected:
            raise ValueError(f"{candidate!r} result-table field changed.")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, _Real) or not _math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return float(value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, _Integral):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def _validate_protocol(value: object) -> None:
    expected = {
        "labelled_rows": FRESH_LABELLED_ROWS,
        "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": FRESH_CROSS_VALIDATION_FINGERPRINT,
        "historical_local_subset": "not reconstructed or consulted",
        "competition_data_opened": False,
        "competition_predictions_generated": False,
    }
    if not isinstance(value, dict) or value != expected:
        raise ValueError("Fresh reconstruction protocol changed.")


def _validate_metric_fields(row: dict[str, object], candidate: str) -> None:
    for field in _METRIC_FIELDS:
        value = row.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, _Real)
            or not _math.isfinite(value)
        ):
            raise ValueError(f"{candidate!r} metric {field!r} is not finite.")


def _validate_count_fields(row: dict[str, object], candidate: str) -> None:
    for field in (
        "fold_wins",
        "fold_losses",
        "prediction_disagreements",
        "gained_correct",
        "lost_correct",
        "net_additional_correct",
    ):
        value = row.get(field)
        if isinstance(value, bool) or not isinstance(value, _Integral):
            raise ValueError(f"{candidate!r} count {field!r} is not an integer.")
    if row["fold_wins"] + row["fold_losses"] > 5:
        raise ValueError(f"{candidate!r} fold counts are inconsistent.")


def _validate_expected_ids(expected_ids: _pd.Series) -> None:
    if not isinstance(expected_ids, _pd.Series):
        raise TypeError("Expected competition IDs must be a pandas Series.")
    if len(expected_ids) != COMPETITION_ROWS:
        raise ValueError("Expected competition ID count changed.")
    if expected_ids.isna().any() or expected_ids.duplicated().any():
        raise ValueError("Expected competition IDs are missing or duplicated.")


def _normalise_label_vector(value: object, name: str) -> _np.ndarray:
    labels = _np.asarray(value, dtype=object)
    if labels.ndim != 1 or len(labels) != COMPETITION_ROWS:
        raise ValueError(f"{name!r} label vector has the wrong shape.")
    if any(label not in CLASS_LABELS for label in labels):
        raise ValueError(f"{name!r} label vector contains an invalid label.")
    return labels.copy()


def _validate_digest_mapping(value: object, name: str) -> None:
    if not isinstance(value, _Mapping) or not value:
        raise ValueError(f"{name} hashes are missing.")
    for path, digest in value.items():
        if (
            not isinstance(path, str)
            or not path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} hashes are malformed.")


def _validate_nested_digests(value: object, name: str) -> None:
    if not isinstance(value, _Mapping) or not value:
        raise ValueError(f"{name} hashes are missing.")
    for key, nested in value.items():
        if isinstance(nested, _Mapping):
            _validate_nested_digests(nested, f"{name}:{key}")
        elif (
            not isinstance(key, str)
            or not isinstance(nested, str)
            or len(nested) != 64
            or any(character not in "0123456789abcdef" for character in nested)
        ):
            raise ValueError(f"{name} hashes are malformed.")


def _require_close(actual: object, expected: float, candidate: str) -> None:
    # The evidence producer serialises metrics with pandas' default 10 decimal
    # digits, so derived differences can move by just under 2e-10 on reload.
    if not _np.isclose(actual, expected, rtol=0.0, atol=2e-10):
        raise ValueError(f"{candidate!r} derived metric is inconsistent.")


def _require_exact_fields(value: dict[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} schema changed.")

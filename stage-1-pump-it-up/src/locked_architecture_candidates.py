"""Build only pre-gated substitutions into the locked deep archive."""

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
from final_model import validate_probabilities


COMPLETE_IDENTITY_CANDIDATE = "complete_identity_catboost"
SPATIAL_GRID_CANDIDATE = "spatial_grid_catboost"
CURRENT_RF_VARIANT = "Current Random Forest"
SPATIAL_SUBSTITUTION = "spatial_grid_catboost_substitution"
RF_SUBSTITUTION = "rf_family_substitution"
COMBINED_SUBSTITUTION = "spatial_grid_catboost_and_rf_family_substitution"


@_dataclass(frozen=True)
class LockedReplacementDecision:
    """Architecture substitutions admitted by both locked fresh screens."""

    use_spatial_grid_catboost: bool
    rf_variant: str | None


def decide_locked_replacements(
    spatial_result: _Mapping[str, object],
    rf_result: _Mapping[str, object],
    *,
    allowed_rf_variants: tuple[str, ...],
) -> LockedReplacementDecision:
    """Validate screen decisions and expose only passing challengers."""

    spatial_passes = _required_bool(spatial_result, "passes_gate", "spatial")
    spatial_selected = _required_string(
        spatial_result,
        "selected_candidate",
        "spatial",
    )
    expected_spatial = (
        SPATIAL_GRID_CANDIDATE
        if spatial_passes
        else COMPLETE_IDENTITY_CANDIDATE
    )
    if spatial_selected != expected_spatial:
        raise ValueError(
            "Spatial screen selection is inconsistent with its pass gate."
        )

    rf_passes = _required_bool(rf_result, "passes_gate", "RF")
    rf_selected = _required_string(rf_result, "selected_candidate", "RF")
    if rf_selected not in allowed_rf_variants:
        raise ValueError(f"RF screen selected an unlocked variant: {rf_selected!r}.")
    if rf_passes == (rf_selected == CURRENT_RF_VARIANT):
        raise ValueError("RF screen selection is inconsistent with its pass gate.")

    return LockedReplacementDecision(
        use_spatial_grid_catboost=spatial_passes,
        rf_variant=rf_selected if rf_passes else None,
    )


def build_locked_substitution_probabilities(
    incumbent_components: _Mapping[str, _np.ndarray],
    decision: LockedReplacementDecision,
    *,
    spatial_grid_catboost: _np.ndarray | None = None,
    rf_family: _np.ndarray | None = None,
) -> dict[str, _np.ndarray]:
    """Blend each admitted component substitution without changing rows."""

    expected = set(DEEP_ARCHIVE_WEIGHTS)
    actual = set(incumbent_components)
    if actual != expected:
        raise ValueError(
            "Incumbent component contract changed; "
            f"missing={sorted(expected - actual)!r}, "
            f"unexpected={sorted(actual - expected)!r}."
        )
    if DEEP_ARCHIVE_WEIGHTS["identity_catboost"] != 0.20:
        raise ValueError("Locked CatBoost weight changed.")
    if DEEP_ARCHIVE_WEIGHTS["random_forest"] != 0.18:
        raise ValueError("Locked Random Forest weight changed.")

    if decision.use_spatial_grid_catboost != (spatial_grid_catboost is not None):
        raise ValueError(
            "Spatial replacement probabilities do not match the gate decision."
        )
    if (decision.rf_variant is not None) != (rf_family is not None):
        raise ValueError("RF replacement probabilities do not match the gate decision.")

    candidates: dict[str, _np.ndarray] = {}
    if spatial_grid_catboost is not None:
        components = {
            **incumbent_components,
            "identity_catboost": spatial_grid_catboost,
        }
        candidates[SPATIAL_SUBSTITUTION] = blend_deep_archive(components)
    if rf_family is not None:
        components = {
            **incumbent_components,
            "random_forest": rf_family,
        }
        candidates[RF_SUBSTITUTION] = blend_deep_archive(components)
    if spatial_grid_catboost is not None and rf_family is not None:
        components = {
            **incumbent_components,
            "identity_catboost": spatial_grid_catboost,
            "random_forest": rf_family,
        }
        candidates[COMBINED_SUBSTITUTION] = blend_deep_archive(components)
    return candidates


def validate_screen_contract(
    actual: _Mapping[str, object],
    expected: _Mapping[str, object],
    *,
    screen: str,
) -> None:
    """Reject a missing or stale nested screen protocol field."""

    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                raise ValueError(f"{screen} screen field {key!r} is missing.")
            validate_screen_contract(
                actual_value,
                expected_value,
                screen=screen,
            )
        elif actual_value != expected_value:
            raise ValueError(f"{screen} screen field {key!r} changed.")


def validate_competition_ids(
    payload: _Mapping[str, object],
    expected_ids: _pd.Series,
    *,
    name: str,
) -> None:
    """Require cached competition IDs to match in value and row order."""

    actual = payload.get("competition_ids")
    if not isinstance(actual, _pd.Series) or not actual.reset_index(
        drop=True
    ).equals(expected_ids.reset_index(drop=True)):
        raise ValueError(f"{name} competition IDs changed or are misordered.")


def validate_replacement_cache(
    payload: object,
    expected_metadata: _Mapping[str, object],
    expected_ids: _pd.Series,
    *,
    expected_rows: int,
    name: str,
) -> dict[str, object]:
    """Validate a versioned replacement-probability cache."""

    required = {
        "metadata",
        "competition_ids",
        "probabilities",
        "fit_and_predict_seconds",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"{name} replacement cache schema changed.")
    if payload["metadata"] != dict(expected_metadata):
        raise ValueError(f"{name} replacement cache metadata is stale.")
    validate_competition_ids(payload, expected_ids, name=name)
    validate_probabilities(payload["probabilities"], expected_rows)
    seconds = payload["fit_and_predict_seconds"]
    if (
        not isinstance(seconds, (int, float))
        or not _np.isfinite(seconds)
        or seconds < 0
    ):
        raise ValueError(f"{name} replacement cache timing is invalid.")
    return payload


def write_json_without_overwrite(
    destination: _Path,
    value: _Mapping[str, object],
) -> None:
    """Write deterministic JSON or refuse to replace different content."""

    text = _json.dumps(value, indent=2) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"Refusing to overwrite different JSON: {destination}."
            )
        return
    destination.write_text(text, encoding="utf-8")


def sha256_file(path: _Path) -> str:
    """Return a streaming lowercase SHA-256 digest."""

    digest = _hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_bool(
    result: _Mapping[str, object],
    key: str,
    screen: str,
) -> bool:
    value = result.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{screen} screen has no explicit Boolean {key!r}.")
    return value


def _required_string(
    result: _Mapping[str, object],
    key: str,
    screen: str,
) -> str:
    value = result.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{screen} screen has no explicit {key!r}.")
    return value

"""Selectively restore archive decisions beside the deep-archive incumbent."""

from __future__ import annotations

from collections.abc import Mapping as _Mapping
from dataclasses import dataclass as _dataclass

import numpy as _np
import pandas as _pd
from sklearn.compose import ColumnTransformer as _ColumnTransformer
from sklearn.linear_model import LogisticRegression as _LogisticRegression
from sklearn.pipeline import make_pipeline as _make_pipeline
from sklearn.preprocessing import OneHotEncoder as _OneHotEncoder
from sklearn.preprocessing import StandardScaler as _StandardScaler

from archive_synthesis_confirmation import blend_archive_synthesis
from deep_archive_confirmation import blend_deep_archive
from final_model import CLASS_LABELS
from final_model import validate_probabilities


ARCHIVE_GATE_COMPONENTS = (
    "accepted_xgboost",
    "deep_xgboost",
    "spatial_xgboost",
    "random_forest",
    "frequency_random_forest",
    "spatial_random_forest",
    "identity_catboost",
)
ARCHIVE_GATE_SHARED_COMPONENTS = (
    "spatial_xgboost",
    "random_forest",
    "frequency_random_forest",
    "spatial_random_forest",
    "identity_catboost",
)
ARCHIVE_GATE_NUMERIC_FEATURES = (
    "shared_gap",
    "accepted_gap",
    "deep_gap",
    "identity_gap",
    "shared_vote_delta",
    "archive_margin",
    "deep_margin",
)
ARCHIVE_GATE_C = 1.0
ARCHIVE_GATE_THRESHOLD = 0.50

_REPAIR_POSITION = CLASS_LABELS.index("functional needs repair")
_ARCHIVE_PRIMARY_WEIGHT = 0.33
_SHARED_WEIGHT = 1.0 - _ARCHIVE_PRIMARY_WEIGHT


@_dataclass(frozen=True)
class ArchiveGateFeatures:
    """Features and class choices for archive/deep hard disagreements."""

    values: _pd.DataFrame
    row_positions: _np.ndarray
    archive_positions: _np.ndarray
    deep_positions: _np.ndarray


def build_archive_and_deep_probabilities(
    components: _Mapping[str, _np.ndarray],
) -> tuple[_np.ndarray, _np.ndarray]:
    """Recreate the old archive and current deep recipes from aligned parts."""

    missing = set(ARCHIVE_GATE_COMPONENTS).difference(components)
    if missing:
        raise KeyError(f"Missing archive-gate components: {sorted(missing)!r}.")
    expected_rows = len(components["accepted_xgboost"])
    for name in ARCHIVE_GATE_COMPONENTS:
        validate_probabilities(
            _np.asarray(components[name], dtype="float64"),
            expected_rows,
        )
    shared = {
        name: _np.asarray(components[name], dtype="float64")
        for name in ARCHIVE_GATE_SHARED_COMPONENTS
    }
    archive = blend_archive_synthesis(
        {
            "xgboost": _np.asarray(
                components["accepted_xgboost"],
                dtype="float64",
            ),
            **shared,
        }
    )
    deep = blend_deep_archive(
        {
            "deep_xgboost": _np.asarray(
                components["deep_xgboost"],
                dtype="float64",
            ),
            **shared,
        }
    )
    return archive, deep


def build_archive_gate_features(
    components: _Mapping[str, _np.ndarray],
    archive_probabilities: _np.ndarray,
    deep_probabilities: _np.ndarray,
) -> ArchiveGateFeatures:
    """Describe only rows where the archive and deep decisions differ."""

    missing = set(ARCHIVE_GATE_COMPONENTS).difference(components)
    if missing:
        raise KeyError(f"Missing archive-gate components: {sorted(missing)!r}.")
    archive = _np.asarray(archive_probabilities, dtype="float64")
    deep = _np.asarray(deep_probabilities, dtype="float64")
    validate_probabilities(archive, len(archive))
    validate_probabilities(deep, len(archive))
    for name in ARCHIVE_GATE_COMPONENTS:
        validate_probabilities(
            _np.asarray(components[name], dtype="float64"),
            len(archive),
        )

    archive_decisions = archive.argmax(axis=1)
    deep_decisions = deep.argmax(axis=1)
    row_positions = _np.flatnonzero(archive_decisions != deep_decisions)
    archive_positions = archive_decisions[row_positions]
    deep_positions = deep_decisions[row_positions]

    accepted = _np.asarray(components["accepted_xgboost"], dtype="float64")
    deep_xgboost = _np.asarray(components["deep_xgboost"], dtype="float64")
    identity = _np.asarray(components["identity_catboost"], dtype="float64")
    shared = (archive - _ARCHIVE_PRIMARY_WEIGHT * accepted) / _SHARED_WEIGHT
    shared_vote_delta = _np.zeros(len(row_positions), dtype="int8")
    for name in ARCHIVE_GATE_SHARED_COMPONENTS:
        decisions = _np.asarray(components[name]).argmax(axis=1)[row_positions]
        shared_vote_delta += (decisions == archive_positions).astype("int8")
        shared_vote_delta -= (decisions == deep_positions).astype("int8")

    values = _pd.DataFrame(
        {
            "transition": [
                f"{archive_position}>{deep_position}"
                for archive_position, deep_position in zip(
                    archive_positions,
                    deep_positions,
                    strict=True,
                )
            ],
            "shared_gap": (
                shared[row_positions, archive_positions]
                - shared[row_positions, deep_positions]
            ),
            "accepted_gap": (
                accepted[row_positions, archive_positions]
                - accepted[row_positions, deep_positions]
            ),
            "deep_gap": (
                deep_xgboost[row_positions, deep_positions]
                - deep_xgboost[row_positions, archive_positions]
            ),
            "identity_gap": (
                identity[row_positions, archive_positions]
                - identity[row_positions, deep_positions]
            ),
            "shared_vote_delta": shared_vote_delta,
            "archive_margin": (
                archive[row_positions, archive_positions]
                - archive[row_positions, deep_positions]
            ),
            "deep_margin": (
                deep[row_positions, deep_positions]
                - deep[row_positions, archive_positions]
            ),
        }
    )
    if tuple(values.columns) != (
        "transition",
        *ARCHIVE_GATE_NUMERIC_FEATURES,
    ):
        raise ValueError("Archive-gate feature columns changed.")
    if not _np.isfinite(
        values.loc[:, ARCHIVE_GATE_NUMERIC_FEATURES].to_numpy()
    ).all():
        raise ValueError("Archive-gate numeric features must all be finite.")
    return ArchiveGateFeatures(
        values=values,
        row_positions=row_positions,
        archive_positions=archive_positions,
        deep_positions=deep_positions,
    )


def make_archive_gate_classifier():
    """Return the fixed regularised logistic archive-choice gate."""

    preprocessor = _ColumnTransformer(
        [
            (
                "transition",
                _OneHotEncoder(handle_unknown="ignore"),
                ["transition"],
            ),
            (
                "numeric",
                _StandardScaler(),
                list(ARCHIVE_GATE_NUMERIC_FEATURES),
            ),
        ]
    )
    return _make_pipeline(
        preprocessor,
        _LogisticRegression(C=ARCHIVE_GATE_C, max_iter=2_000),
    )


def archive_choice_target(
    target: _pd.Series | _np.ndarray,
    features: ArchiveGateFeatures,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Return rows with a consequential choice and one for archive-correct."""

    values = _np.asarray(target, dtype=object)
    if len(values) <= int(features.row_positions.max(initial=-1)):
        raise ValueError("Archive-gate target is shorter than its row positions.")
    selected_target = values[features.row_positions]
    labels = _np.asarray(CLASS_LABELS, dtype=object)
    archive_correct = selected_target == labels[features.archive_positions]
    deep_correct = selected_target == labels[features.deep_positions]
    relevant = archive_correct | deep_correct
    return relevant, archive_correct.astype("int8")


def apply_repair_preserving_archive_gate(
    archive_probabilities: _np.ndarray,
    deep_probabilities: _np.ndarray,
    features: ArchiveGateFeatures,
    archive_choice_probabilities: _np.ndarray,
    *,
    threshold: float = ARCHIVE_GATE_THRESHOLD,
) -> tuple[_np.ndarray, _np.ndarray]:
    """Restore selected archive rows without removing a deep repair decision."""

    archive = _np.asarray(archive_probabilities, dtype="float64")
    deep = _np.asarray(deep_probabilities, dtype="float64")
    validate_probabilities(archive, len(archive))
    validate_probabilities(deep, len(archive))
    choice = _np.asarray(archive_choice_probabilities, dtype="float64")
    if choice.shape != (len(features.row_positions),):
        raise ValueError("Archive-choice probabilities have the wrong shape.")
    if not _np.isfinite(choice).all() or (choice < 0).any() or (choice > 1).any():
        raise ValueError("Archive-choice probabilities must lie in [0, 1].")
    if not 0 < threshold < 1:
        raise ValueError("Archive-gate threshold must lie in (0, 1).")

    selected_disagreements = (choice >= threshold) & (
        features.deep_positions != _REPAIR_POSITION
    )
    selected = _np.zeros(len(deep), dtype=bool)
    selected[features.row_positions[selected_disagreements]] = True
    gated = deep.copy()
    gated[selected] = archive[selected]
    validate_probabilities(gated, len(deep))
    if selected.any():
        if _np.any(deep[selected].argmax(axis=1) == _REPAIR_POSITION):
            raise ValueError("Archive gate removed a deep repair decision.")
        if not _np.array_equal(
            gated[selected].argmax(axis=1),
            archive[selected].argmax(axis=1),
        ):
            raise ValueError("Selected gate rows did not take archive decisions.")
    return gated, selected

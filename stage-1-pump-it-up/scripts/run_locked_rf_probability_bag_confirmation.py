"""Replay two fixed RF probability bags without fitting any model."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
RF_SCREEN_DIR = (
    PROJECT_DIR / ".runtime" / "fresh-rf-family-screen-seed-20260905"
)
ENSEMBLE_DIR = (
    PROJECT_DIR / ".runtime" / "locked-rf-ensemble-confirmation-20260905"
)
OUTPUT_DIR = (
    PROJECT_DIR / ".runtime" / "locked-rf-probability-bag-20260905"
)
SOURCE_HASHES = {
    "TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    "TrainingSetLabels.csv": (
        "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
    ),
}
FRESH_FOLD_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from fresh_rf_family_evaluation import CURRENT_RF_VARIANT
from fresh_rf_family_evaluation import FOLD_WIN_GATE
from fresh_rf_family_evaluation import FRESH_CROSS_VALIDATION_SEED
from fresh_rf_family_evaluation import MEAN_ACCURACY_GAIN_GATE
from fresh_rf_family_evaluation import REPAIR_RECALL_DELTA_GATE
from fresh_rf_family_evaluation import WORST_FOLD_DELTA_GATE
from fresh_rf_family_evaluation import validate_rf_cache_payload
from locked_architecture_candidates import sha256_file
from locked_architecture_candidates import write_json_without_overwrite
from locked_rf_ensemble_confirmation import CONFIRMATION_CANDIDATES
from locked_rf_ensemble_confirmation import ENSEMBLE_FOLD_WIN_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_MEAN_ACCURACY_GAIN_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_REPAIR_RECALL_DELTA_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_WORST_FOLD_DELTA_GATE
from locked_rf_ensemble_confirmation import compare_fresh_rf_probability_bags
from locked_rf_ensemble_confirmation import (
    compare_locked_rf_ensemble_substitutions,
)
from locked_rf_ensemble_confirmation import load_locked_rf_screen_candidates
from locked_rf_ensemble_confirmation import make_equal_rf_probability_bags
from locked_rf_ensemble_confirmation import validate_confirmation_cache_payload
from run_locked_rf_ensemble_confirmation import _cache_metadata
from run_locked_rf_ensemble_confirmation import _load_frozen_development_partition
from run_locked_rf_ensemble_confirmation import _load_incumbent_components


def main() -> None:
    screen = load_locked_rf_screen_candidates(RF_SCREEN_DIR)
    fresh_partition = _load_fresh_partition()
    fresh_evaluations, fresh_cache_hashes = _load_fresh_evaluations(
        screen,
        fresh_partition,
    )
    fresh = compare_fresh_rf_probability_bags(
        fresh_partition,
        fresh_evaluations,
    )

    frozen_partition = _load_frozen_development_partition()
    incumbent_components, component_hashes = _load_incumbent_components(
        frozen_partition
    )
    old_challengers, old_cache_hashes = _load_old_challengers(
        screen,
        frozen_partition,
        component_hashes,
    )
    bags = make_equal_rf_probability_bags(
        incumbent_components["random_forest"],
        old_challengers,
    )
    ensemble = compare_locked_rf_ensemble_substitutions(
        frozen_partition,
        incumbent_components,
        bags,
    )
    summary = _combine_summaries(fresh.summary, ensemble.summary)
    folds = pd.concat(
        [
            fresh.fold_deltas.assign(evidence_layer="fresh RF bag"),
            ensemble.fold_deltas.assign(evidence_layer="exact ensemble bag"),
        ],
        ignore_index=True,
    )
    passing = summary.index[summary["passes_both_gates"]].tolist()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv_without_overwrite(OUTPUT_DIR / "candidate-summary.csv", summary)
    _write_csv_without_overwrite(
        OUTPUT_DIR / "fold-paired-deltas.csv",
        folds,
        index=False,
    )
    result = {
        "screen": "fixed 50:50 incumbent/challenger RF probability-bag replay",
        "status": "cache_only_replay_complete",
        "weights_searched": False,
        "model_fitting_performed": False,
        "rf_bag_weights": {"incumbent": 0.5, "challenger": 0.5},
        "deep_archive_rf_slot_weight": 0.18,
        "candidates": list(CONFIRMATION_CANDIDATES),
        "passing_candidates": passing,
        "fresh_gate_thresholds": {
            "mean_accuracy_gain": MEAN_ACCURACY_GAIN_GATE,
            "minimum_fold_wins": FOLD_WIN_GATE,
            "worst_fold_delta": WORST_FOLD_DELTA_GATE,
            "minimum_repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
        },
        "ensemble_gate_thresholds": {
            "mean_accuracy_gain": ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
            "minimum_fold_wins": ENSEMBLE_FOLD_WIN_GATE,
            "worst_fold_delta": ENSEMBLE_WORST_FOLD_DELTA_GATE,
            "minimum_repair_recall_delta": (
                ENSEMBLE_REPAIR_RECALL_DELTA_GATE
            ),
        },
        "protocol": {
            "fresh_rows": len(fresh_partition.y_development),
            "fresh_cross_validation_fingerprint": (
                fresh_partition.cross_validation_fingerprint
            ),
            "frozen_development_rows": len(
                frozen_partition.y_development
            ),
            "frozen_cross_validation_fingerprint": (
                frozen_partition.cross_validation_fingerprint
            ),
            "historical_local_membership_reconstructed_for_fresh_layer": False,
            "historical_local_test_scored_separately": False,
            "competition_data_opened": False,
            "competition_predictions_generated": False,
        },
        "source_sha256": SOURCE_HASHES,
        "fresh_screen_sha256": screen.hashes,
        "fresh_cache_sha256": fresh_cache_hashes,
        "direct_ensemble_cache_sha256": old_cache_hashes,
        "incumbent_component_sha256": component_hashes,
        "results": json.loads(
            summary.reset_index().to_json(orient="records")
        ),
    }
    write_json_without_overwrite(OUTPUT_DIR / "result.json", result)
    print(_format_summary(summary), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def _load_fresh_partition() -> PartitionedData:
    values_path = DATA_DIR / "TrainingSetValues.csv"
    labels_path = DATA_DIR / "TrainingSetLabels.csv"
    _require_hash(values_path, SOURCE_HASHES[values_path.name])
    _require_hash(labels_path, SOURCE_HASHES[labels_path.name])
    ids = pd.read_csv(values_path, usecols=["id"])["id"]
    labels = pd.read_csv(labels_path)
    if list(labels.columns) != ["id", "status_group"]:
        raise ValueError("Full-label source schema changed.")
    if ids.duplicated().any() or labels["id"].duplicated().any():
        raise ValueError("Full-label identifiers are not unique.")
    target = (
        labels.set_index("id").loc[ids, "status_group"].reset_index(drop=True)
    )
    folds = _make_fresh_folds(target)
    fingerprint = _fingerprint_folds(ids, folds)
    if fingerprint != FRESH_FOLD_FINGERPRINT:
        raise ValueError("Fresh all-label fold assignment changed.")
    X = pd.DataFrame(index=target.index)
    return PartitionedData(
        development_ids=ids.reset_index(drop=True),
        X_development=X,
        y_development=target,
        local_test_ids=ids.iloc[0:0].copy(),
        X_local_test=X.iloc[0:0].copy(),
        y_local_test=target.iloc[0:0].copy(),
        validation_folds=folds,
        development_fingerprint="all-supplied-labels",
        local_test_fingerprint="not-reconstructed-for-fresh-replay",
        cross_validation_fingerprint=fingerprint,
    )


def _make_fresh_folds(target: pd.Series) -> pd.Series:
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=FRESH_CROSS_VALIDATION_SEED,
    )
    assignments = np.zeros(len(target), dtype=np.int8)
    dummy = np.zeros((len(target), 1))
    for fold, (_, positions) in enumerate(
        splitter.split(dummy, target),
        start=1,
    ):
        assignments[positions] = fold
    return pd.Series(assignments, name="validation_fold")


def _load_fresh_evaluations(screen, partitioned):
    evaluations = {}
    hashes = {}
    for candidate in (CURRENT_RF_VARIANT, *CONFIRMATION_CANDIDATES):
        path = RF_SCREEN_DIR / f"{_slug(candidate)}.joblib"
        payload = joblib.load(path)
        evaluations[candidate] = validate_rf_cache_payload(
            payload,
            (
                screen.specs[candidate]
                if candidate in screen.specs
                else _current_spec()
            ),
            partitioned,
        )
        hashes[candidate] = sha256_file(path)
    return evaluations, hashes


def _current_spec():
    from fresh_rf_family_evaluation import make_locked_rf_specs

    return make_locked_rf_specs()[CURRENT_RF_VARIANT]


def _load_old_challengers(screen, partitioned, component_hashes):
    evaluations = {}
    hashes = {}
    for candidate in CONFIRMATION_CANDIDATES:
        path = ENSEMBLE_DIR / f"{_slug(candidate)}.joblib"
        metadata = _cache_metadata(
            candidate,
            screen.specs[candidate],
            screen.hashes,
            component_hashes,
        )
        evaluation = validate_confirmation_cache_payload(
            joblib.load(path),
            metadata,
            partitioned,
            candidate=candidate,
        )
        evaluations[candidate] = (
            evaluation.out_of_fold_probabilities.to_numpy(dtype="float64")
        )
        hashes[candidate] = sha256_file(path)
    return evaluations, hashes


def _combine_summaries(fresh: pd.DataFrame, ensemble: pd.DataFrame):
    rows = []
    for candidate in CONFIRMATION_CANDIDATES:
        fresh_row = fresh.loc[candidate]
        ensemble_row = ensemble.loc[candidate]
        rows.append(
            {
                "candidate": candidate,
                "fresh_mean_accuracy_delta": fresh_row[
                    "mean_accuracy_delta"
                ],
                "fresh_fold_wins": fresh_row["fold_wins_vs_incumbent"],
                "fresh_worst_fold_delta": fresh_row["worst_fold_delta"],
                "fresh_repair_recall_delta": fresh_row[
                    "repair_recall_delta"
                ],
                "passes_fresh_gate": bool(fresh_row["passes_gate"]),
                "ensemble_mean_accuracy_delta": ensemble_row[
                    "mean_accuracy_delta"
                ],
                "ensemble_fold_wins": ensemble_row["fold_wins"],
                "ensemble_worst_fold_delta": ensemble_row[
                    "worst_fold_delta"
                ],
                "ensemble_repair_recall_delta": ensemble_row[
                    "repair_recall_delta"
                ],
                "ensemble_net_additional_correct": ensemble_row[
                    "net_additional_correct"
                ],
                "passes_ensemble_gate": bool(ensemble_row["passes_gate"]),
                "passes_both_gates": bool(
                    fresh_row["passes_gate"] and ensemble_row["passes_gate"]
                ),
            }
        )
    return pd.DataFrame(rows).set_index("candidate")


def _write_csv_without_overwrite(path, frame, *, index=True):
    expected = frame.to_csv(index=index)
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(f"Refusing to overwrite different CSV: {path}.")
        return
    path.write_text(expected, encoding="utf-8")


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 changed for {path}: {actual}.")


def _fingerprint_folds(ids: pd.Series, folds: pd.Series) -> str:
    import hashlib

    digest = hashlib.sha256()
    memberships = sorted(
        zip((str(value) for value in ids), folds, strict=True),
        key=lambda item: item[0],
    )
    for identifier, fold in memberships:
        digest.update(f"{identifier}:{fold}\n".encode("utf-8"))
    return digest.hexdigest()


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in display.columns:
        if "delta" in column:
            display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


def _slug(value: str) -> str:
    return "-".join(value.casefold().split()).replace(".", "-")


if __name__ == "__main__":
    main()

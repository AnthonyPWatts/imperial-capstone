"""Confirm the fixed identity/grid CatBoost bag in the locked archive.

This runner deliberately reads only the frozen 47,520-row development labels
and the already-created fresh-screen evidence.  It never opens the historical
local-test labels or any competition data.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import joblib
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
SRC_DIR = STAGE_DIR / "src"
SPATIAL_SCREEN_DIR = (
    PROJECT_DIR / ".runtime" / "fresh-spatial-grid-catboost"
)
FRESH_BAG_DIR = (
    PROJECT_DIR / ".runtime" / "locked-catboost-probability-bag-20260905"
)
OUTPUT_DIR = (
    PROJECT_DIR
    / ".runtime"
    / "locked-catboost-bag-ensemble-confirmation-20260905"
)
SPATIAL_CACHE_PATH = OUTPUT_DIR / "spatial-grid-catboost-old-folds.joblib"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import CROSS_VALIDATION_FOLDS
from data_partitioning import CROSS_VALIDATION_SEED
from data_partitioning import make_cross_validation
from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from gpu_model_evaluation import CATBOOST_VARIANTS
from locked_architecture_candidates import sha256_file
from locked_architecture_candidates import validate_screen_contract
from locked_architecture_candidates import write_json_without_overwrite
from locked_catboost_probability_bag import CATBOOST_BAG_CANDIDATE
from locked_catboost_probability_bag import CATBOOST_OLD_FOLD_CACHE_VERSION
from locked_catboost_probability_bag import compare_locked_catboost_bag_ensemble
from locked_catboost_probability_bag import passes_fresh_catboost_bag_gate
from locked_catboost_probability_bag import validate_old_fold_spatial_grid_cache
from locked_rf_ensemble_confirmation import ENSEMBLE_FOLD_WIN_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_MEAN_ACCURACY_GAIN_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_REPAIR_RECALL_DELTA_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_WORST_FOLD_DELTA_GATE
from locked_screen_evidence import FRESH_FOLD_FINGERPRINT
from run_locked_rf_ensemble_confirmation import COMPONENT_PATHS
from run_locked_rf_ensemble_confirmation import CROSS_VALIDATION_FINGERPRINT
from run_locked_rf_ensemble_confirmation import DEVELOPMENT_FINGERPRINT
from run_locked_rf_ensemble_confirmation import SOURCE_HASHES
from run_locked_rf_ensemble_confirmation import _load_frozen_development_partition
from run_locked_rf_ensemble_confirmation import _load_incumbent_components
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_ACCURACY
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_FOLD_WINS
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_WORST_FOLD
from spatial_grid_catboost_evaluation import evaluate_spatial_grid_catboost
from spatial_grid_catboost_evaluation import make_spatial_grid_catboost_spec
from spatial_grid_catboost_evaluation import spatial_grid_policy_signature


def main() -> None:
    fresh_evidence = _load_fresh_bag_authorisation()
    partitioned = _load_frozen_development_partition()
    components, component_hashes = _load_incumbent_components(partitioned)
    metadata = _cache_metadata(fresh_evidence, component_hashes)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SPATIAL_CACHE_PATH.exists():
        payload = joblib.load(SPATIAL_CACHE_PATH)
        print(f"Loaded {SPATIAL_CACHE_PATH}.", flush=True)
    else:
        print(
            "Fresh bag passed; fitting the sole missing old-fold spatial-grid "
            "CatBoost component on 47,520 frozen development rows.",
            flush=True,
        )
        trial = evaluate_spatial_grid_catboost(
            partitioned,
            make_cross_validation(partitioned),
        )
        payload = {
            "cache_version": CATBOOST_OLD_FOLD_CACHE_VERSION,
            "metadata": metadata,
            "trial": trial,
        }
        joblib.dump(payload, SPATIAL_CACHE_PATH, compress=3)
        print(f"Cached {SPATIAL_CACHE_PATH}.", flush=True)

    spatial = validate_old_fold_spatial_grid_cache(
        payload,
        metadata,
        partitioned,
    )
    comparison = compare_locked_catboost_bag_ensemble(
        partitioned,
        components,
        spatial.out_of_fold_probabilities.to_numpy(dtype="float64"),
    )
    summary = comparison.summary
    passing = bool(summary.loc[CATBOOST_BAG_CANDIDATE, "passes_gate"])
    _write_csv_without_overwrite(OUTPUT_DIR / "candidate-summary.csv", summary)
    _write_csv_without_overwrite(
        OUTPUT_DIR / "fold-paired-deltas.csv",
        comparison.fold_deltas,
    )

    result = {
        "screen": "fixed CatBoost bag exact-ensemble confirmation",
        "status": (
            "candidate_passed_requires_review"
            if passing
            else "comparison_complete_candidate_failed"
        ),
        "candidate": CATBOOST_BAG_CANDIDATE,
        "passes_gate": passing,
        "competition_refit_authorised": passing,
        "fresh_bag_authorisation": fresh_evidence,
        "bag_weights": {
            "complete_identity_catboost": 0.5,
            "spatial_grid_catboost": 0.5,
        },
        "deep_archive_catboost_slot_weight": (
            DEEP_ARCHIVE_WEIGHTS["identity_catboost"]
        ),
        "weights_searched": False,
        "gate_thresholds": {
            "mean_accuracy_gain": ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
            "minimum_fold_wins": ENSEMBLE_FOLD_WIN_GATE,
            "worst_fold_delta": ENSEMBLE_WORST_FOLD_DELTA_GATE,
            "minimum_repair_recall_delta": (
                ENSEMBLE_REPAIR_RECALL_DELTA_GATE
            ),
        },
        "protocol": {
            "development_rows": len(partitioned.y_development),
            "development_fingerprint": partitioned.development_fingerprint,
            "cross_validation_folds": CROSS_VALIDATION_FOLDS,
            "cross_validation_seed": CROSS_VALIDATION_SEED,
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "historical_local_test_labels_opened": False,
            "historical_local_test_scored": False,
            "competition_data_opened": False,
            "competition_predictions_generated": False,
        },
        "source_sha256": SOURCE_HASHES,
        "incumbent_component_sha256": component_hashes,
        "spatial_grid_cache": {
            "file": SPATIAL_CACHE_PATH.name,
            "sha256": sha256_file(SPATIAL_CACHE_PATH),
            "metadata": metadata,
        },
        "incumbent_mean_accuracy": float(
            comparison.incumbent_evaluation.metric_summary.loc[
                "accuracy", "mean"
            ]
        ),
        "result": json.loads(
            summary.reset_index().to_json(orient="records")
        )[0],
    }
    write_json_without_overwrite(OUTPUT_DIR / "result.json", result)
    print(_format_summary(summary), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def _load_fresh_bag_authorisation() -> dict[str, object]:
    """Validate the completed cache-only fresh decision without reopening labels."""

    result_path = FRESH_BAG_DIR / "result.json"
    summary_path = FRESH_BAG_DIR / "candidate-summary.csv"
    folds_path = FRESH_BAG_DIR / "fold-paired-deltas.csv"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_screen_contract(
        result,
        {
            "screen": (
                "fixed 50:50 complete-identity/spatial-grid CatBoost bag"
            ),
            "status": "fresh_cache_only_replay_complete",
            "candidate": CATBOOST_BAG_CANDIDATE,
            "weights": {
                "complete_identity_catboost": 0.5,
                "spatial_grid_catboost": 0.5,
            },
            "weights_searched": False,
            "model_fitting_performed": False,
            "passes_gate": True,
            "old_fold_confirmation_run": False,
            "gate_thresholds": {
                "mean_accuracy_gain": SPATIAL_GRID_GATE_ACCURACY,
                "minimum_fold_wins": SPATIAL_GRID_GATE_FOLD_WINS,
                "worst_fold_delta": SPATIAL_GRID_GATE_WORST_FOLD,
                "repair_recall_gate": None,
            },
            "protocol": {
                "labelled_rows": 59_400,
                "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
                "historical_local_membership_reconstructed": False,
                "historical_local_test_scored_separately": False,
                "competition_data_opened": False,
                "competition_predictions_generated": False,
            },
        },
        screen="fresh CatBoost bag",
    )
    summary = pd.read_csv(summary_path, index_col="candidate")
    if list(summary.index) != [CATBOOST_BAG_CANDIDATE]:
        raise ValueError("Fresh CatBoost bag summary candidate changed.")
    row = summary.loc[CATBOOST_BAG_CANDIDATE]
    result_row = result.get("result")
    if not isinstance(result_row, dict):
        raise ValueError("Fresh CatBoost bag result row is missing.")
    for column in summary.columns:
        actual = row[column]
        expected = result_row.get(column)
        if isinstance(actual, float):
            if abs(float(actual) - float(expected)) > 1e-10:
                raise ValueError(f"Fresh CatBoost bag {column!r} changed.")
        elif actual != expected:
            raise ValueError(f"Fresh CatBoost bag {column!r} changed.")
    if not passes_fresh_catboost_bag_gate(
        mean_accuracy_delta=float(row["mean_accuracy_delta"]),
        fold_wins=int(row["fold_wins"]),
        worst_fold_delta=float(row["worst_fold_delta"]),
    ):
        raise ValueError("Fresh CatBoost bag no longer passes its locked gate.")

    spatial_hashes = result.get("spatial_screen_sha256")
    if not isinstance(spatial_hashes, dict):
        raise ValueError("Fresh CatBoost bag has no spatial-screen hashes.")
    current_spatial_hashes = {
        "result": sha256_file(SPATIAL_SCREEN_DIR / "result.json"),
        "summary": sha256_file(SPATIAL_SCREEN_DIR / "candidate-summary.csv"),
        "folds": sha256_file(SPATIAL_SCREEN_DIR / "fold-deltas.csv"),
        "identity": sha256_file(
            SPATIAL_SCREEN_DIR / "complete-identity-catboost.joblib"
        ),
        "identity_protocol": sha256_file(
            SPATIAL_SCREEN_DIR
            / "complete-identity-catboost.protocol-v2.json"
        ),
        "grid": sha256_file(
            SPATIAL_SCREEN_DIR / "spatial-grid-catboost.joblib"
        ),
        "grid_protocol": sha256_file(
            SPATIAL_SCREEN_DIR / "spatial-grid-catboost.protocol-v2.json"
        ),
    }
    if spatial_hashes != current_spatial_hashes:
        raise ValueError("Fresh CatBoost bag spatial-screen evidence changed.")
    return {
        "result_file": result_path.name,
        "result_sha256": sha256_file(result_path),
        "summary_sha256": sha256_file(summary_path),
        "folds_sha256": sha256_file(folds_path),
        "spatial_screen_sha256": current_spatial_hashes,
        "mean_accuracy_delta": float(row["mean_accuracy_delta"]),
        "fold_wins": int(row["fold_wins"]),
        "worst_fold_delta": float(row["worst_fold_delta"]),
        "repair_recall_delta": float(row["repair_recall_delta"]),
        "passes_gate": True,
    }


def _cache_metadata(
    fresh_evidence: dict[str, object],
    component_hashes: dict[str, str],
) -> dict[str, object]:
    spec = make_spatial_grid_catboost_spec()
    return {
        "candidate": "spatial_grid_catboost",
        "model_recipe": {
            "spec": asdict(spec),
            "variant_parameters": dict(CATBOOST_VARIANTS[spec.variant]),
        },
        "feature_policy_signature": spatial_grid_policy_signature(),
        "development_rows": 47_520,
        "development_fingerprint": DEVELOPMENT_FINGERPRINT,
        "cross_validation_folds": CROSS_VALIDATION_FOLDS,
        "cross_validation_seed": CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": CROSS_VALIDATION_FINGERPRINT,
        "source_sha256": SOURCE_HASHES,
        "fresh_bag_authorisation": fresh_evidence,
        "incumbent_component_sha256": component_hashes,
        "deep_archive_weights": DEEP_ARCHIVE_WEIGHTS,
        "historical_local_test_labels_opened": False,
        "competition_data_opened": False,
    }


def _write_csv_without_overwrite(path: Path, frame: pd.DataFrame) -> None:
    expected = frame.to_csv(lineterminator="\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8")
        # The first completed run used pandas' CRLF plus text-mode newline
        # translation, yielding one harmless blank line after every row.
        legacy_double_spaced = expected.replace("\n", "\n\n")
        if actual not in (expected, legacy_double_spaced):
            raise FileExistsError(f"Refusing to overwrite different CSV: {path}.")
        return
    path.write_text(expected, encoding="utf-8")


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in display.columns:
        if "accuracy" in column or "recall" in column or "delta" in column:
            display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


if __name__ == "__main__":
    main()

"""Replay the one fixed identity/grid CatBoost bag on fresh OOF caches."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
SRC_DIR = STAGE_DIR / "src"
SPATIAL_DIR = PROJECT_DIR / ".runtime" / "fresh-spatial-grid-catboost"
OUTPUT_DIR = (
    PROJECT_DIR / ".runtime" / "locked-catboost-probability-bag-20260905"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from locked_architecture_candidates import write_json_without_overwrite
from locked_catboost_probability_bag import CATBOOST_BAG_CANDIDATE
from locked_catboost_probability_bag import compare_fresh_catboost_probability_bag
from locked_screen_evidence import load_spatial_screen_evidence
from modelling_data import ModellingData
from run_locked_rf_probability_bag_confirmation import SOURCE_HASHES
from run_locked_rf_probability_bag_confirmation import _load_fresh_partition
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_ACCURACY
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_FOLD_WINS
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_WORST_FOLD


def main() -> None:
    partitioned = _load_fresh_partition()
    modelling = ModellingData(
        original_ids=partitioned.development_ids,
        X_original=partitioned.X_development,
        y_original=partitioned.y_development,
        competition_ids=partitioned.development_ids.iloc[0:0].copy(),
        X_competition=partitioned.X_development.iloc[0:0].copy(),
    )
    screen = load_spatial_screen_evidence(SPATIAL_DIR, modelling)
    identity = joblib.load(
        SPATIAL_DIR / "complete-identity-catboost.joblib"
    )["trial"].evaluation
    grid = joblib.load(
        SPATIAL_DIR / "spatial-grid-catboost.joblib"
    )["trial"].evaluation
    comparison = compare_fresh_catboost_probability_bag(
        partitioned,
        identity,
        grid,
    )
    summary = comparison.summary
    passing = bool(summary.loc[CATBOOST_BAG_CANDIDATE, "passes_gate"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv_without_overwrite(OUTPUT_DIR / "candidate-summary.csv", summary)
    _write_csv_without_overwrite(
        OUTPUT_DIR / "fold-paired-deltas.csv",
        comparison.fold_deltas,
    )
    result = {
        "screen": "fixed 50:50 complete-identity/spatial-grid CatBoost bag",
        "status": "fresh_cache_only_replay_complete",
        "candidate": CATBOOST_BAG_CANDIDATE,
        "weights": {
            "complete_identity_catboost": 0.5,
            "spatial_grid_catboost": 0.5,
        },
        "weights_searched": False,
        "model_fitting_performed": False,
        "passes_gate": passing,
        "old_fold_confirmation_run": False,
        "gate_thresholds": {
            "mean_accuracy_gain": SPATIAL_GRID_GATE_ACCURACY,
            "minimum_fold_wins": SPATIAL_GRID_GATE_FOLD_WINS,
            "worst_fold_delta": SPATIAL_GRID_GATE_WORST_FOLD,
            "repair_recall_gate": None,
        },
        "protocol": {
            "labelled_rows": len(partitioned.y_development),
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "historical_local_membership_reconstructed": False,
            "historical_local_test_scored_separately": False,
            "competition_data_opened": False,
            "competition_predictions_generated": False,
        },
        "source_sha256": SOURCE_HASHES,
        "spatial_screen_sha256": screen.hashes,
        "result": json.loads(summary.reset_index().to_json(orient="records"))[0],
    }
    write_json_without_overwrite(OUTPUT_DIR / "result.json", result)
    print(_format_summary(summary), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def _write_csv_without_overwrite(path, frame) -> None:
    expected = frame.to_csv()
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(f"Refusing to overwrite different CSV: {path}.")
        return
    path.write_text(expected, encoding="utf-8")


def _format_summary(summary) -> str:
    display = summary.copy()
    for column in display.columns:
        if "accuracy" in column or "recall" in column or "delta" in column:
            display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


if __name__ == "__main__":
    main()

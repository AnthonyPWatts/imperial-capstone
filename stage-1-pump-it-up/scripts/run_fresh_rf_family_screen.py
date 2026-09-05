"""Run the locked RF-family screen on one fresh full-labelled fold plan."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import joblib
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from fresh_rf_family_evaluation import CURRENT_RF_VARIANT
from fresh_rf_family_evaluation import FRESH_CROSS_VALIDATION_SEED
from fresh_rf_family_evaluation import LOCKED_MODEL_SEED
from fresh_rf_family_evaluation import FOLD_WIN_GATE
from fresh_rf_family_evaluation import MEAN_ACCURACY_GAIN_GATE
from fresh_rf_family_evaluation import REPAIR_RECALL_DELTA_GATE
from fresh_rf_family_evaluation import RF_SCREEN_CACHE_VERSION
from fresh_rf_family_evaluation import WORST_FOLD_DELTA_GATE
from fresh_rf_family_evaluation import make_full_labelled_partition
from fresh_rf_family_evaluation import make_locked_rf_specs
from fresh_rf_family_evaluation import make_rf_cache_payload
from fresh_rf_family_evaluation import paired_fold_deltas
from fresh_rf_family_evaluation import summarise_rf_screen
from fresh_rf_family_evaluation import validate_rf_cache_payload
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import resolve_sklearn_tree_parameters
from modelling_data import prepare_modelling_data


OUTPUT_DIR = (
    PROJECT_DIR
    / ".runtime"
    / f"fresh-rf-family-screen-seed-{FRESH_CROSS_VALIDATION_SEED}"
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = make_full_labelled_partition(modelling_data)
    cross_validation = make_cross_validation(partitioned)
    specs = make_locked_rf_specs()

    evaluations = {}
    for variant, spec in specs.items():
        cache_path = OUTPUT_DIR / f"{_slug(variant)}.joblib"
        if cache_path.exists():
            evaluation = validate_rf_cache_payload(
                joblib.load(cache_path),
                spec,
                partitioned,
            )
            print(f"Loaded {cache_path}.", flush=True)
        else:
            evaluation = evaluate_gpu_candidate(
                spec,
                partitioned,
                cross_validation,
            )
            joblib.dump(
                make_rf_cache_payload(spec, partitioned, evaluation),
                cache_path,
            )
        evaluations[variant] = evaluation

    paired = paired_fold_deltas(
        partitioned,
        cross_validation,
        evaluations,
    )
    summary = summarise_rf_screen(evaluations, paired)
    passing = summary.loc[summary["passes_gate"]]
    screen_winner = str(summary.index[0])
    passes_gate = not passing.empty
    selected_candidate = (
        str(passing.index[0]) if passes_gate else CURRENT_RF_VARIANT
    )
    paired.to_csv(OUTPUT_DIR / "fold-paired-deltas.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    for variant, evaluation in evaluations.items():
        evaluation.fold_metrics.to_csv(
            OUTPUT_DIR / f"{_slug(variant)}-fold-metrics.csv"
        )

    result = {
        "screen": "locked fresh-seed RF family screen",
        "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
        "model_seed": LOCKED_MODEL_SEED,
        "folds": 5,
        "labelled_rows": len(partitioned.y_development),
        "cross_validation_fingerprint": (
            partitioned.cross_validation_fingerprint
        ),
        "cache_version": RF_SCREEN_CACHE_VERSION,
        "incumbent": CURRENT_RF_VARIANT,
        "locked_candidates": list(specs),
        "locked_specs": {
            variant: make_rf_cache_payload(
                spec,
                partitioned,
                evaluations[variant],
            )["candidate_spec"]
            for variant, spec in specs.items()
        },
        "locked_model_parameters": {
            variant: resolve_sklearn_tree_parameters(spec)
            for variant, spec in specs.items()
        },
        "gate_thresholds": {
            "mean_accuracy_gain": MEAN_ACCURACY_GAIN_GATE,
            "minimum_fold_wins": FOLD_WIN_GATE,
            "worst_fold_delta": WORST_FOLD_DELTA_GATE,
            "minimum_repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
        },
        "screen_winner": screen_winner,
        "best_candidate": screen_winner,
        "best_mean_accuracy": float(summary.iloc[0]["mean_accuracy"]),
        "best_mean_accuracy_delta": float(
            summary.iloc[0]["mean_accuracy_delta"]
        ),
        "passes_gate": passes_gate,
        "passing_candidates": [str(name) for name in passing.index],
        "selected_candidate": selected_candidate,
        "selected_candidate_is_challenger": (
            selected_candidate != CURRENT_RF_VARIANT
        ),
        "selected_mean_accuracy": float(
            summary.loc[selected_candidate, "mean_accuracy"]
        ),
        "selected_mean_accuracy_delta": float(
            summary.loc[selected_candidate, "mean_accuracy_delta"]
        ),
        "selected_repair_recall_delta": float(
            summary.loc[selected_candidate, "repair_recall_delta"]
        ),
        "old_local_test_membership_reconstructed": False,
        "old_local_test_scored_separately": False,
        "competition_predictions_generated": False,
        "external_data_used": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nCandidate summary:\n", _format_percentages(summary), flush=True)
    print("\nFold-wise paired deltas:\n", _format_percentages(paired), flush=True)
    print(f"\nCached evidence in {OUTPUT_DIR}.", flush=True)


def _format_percentages(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if "accuracy" in column or "delta" in column or "recall" in column:
            display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string(index=not isinstance(display.index, pd.RangeIndex))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


if __name__ == "__main__":
    main()

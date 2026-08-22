"""Run and persist the fold-safe regional-specialisation screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "regional-specialisation-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from modelling_data import prepare_modelling_data
from regional_specialisation_evaluation import evaluate_regional_specialisation


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "regional-specialisation-screen.joblib"
    partitioned_data = partition_modelling_data(_load_modelling_data())
    if output_path.exists() and not args.force:
        result = joblib.load(output_path)
        for evaluation in (
            result.hard_routed_expert,
            result.partially_pooled_expert,
            *result.prior_layers.values(),
        ):
            if (
                evaluation.cross_validation_fingerprint
                != partitioned_data.cross_validation_fingerprint
            ):
                raise ValueError("Cached regional evidence uses different folds.")
        print(f"Loaded {output_path}.", flush=True)
    else:
        baseline = joblib.load(BASELINE_PATH)
        cross_validation = make_cross_validation(partitioned_data)
        result = evaluate_regional_specialisation(
            partitioned_data,
            cross_validation,
            baseline.blend,
        )
        joblib.dump(result, output_path)

    result.candidate_summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result.routing_summary.to_csv(OUTPUT_DIR / "routing-summary.csv")
    result.regional_summary.to_csv(OUTPUT_DIR / "regional-summary.csv")
    payload = {
        "candidate_count": len(result.candidate_summary),
        "selection_gate_passers": int(
            result.candidate_summary["passes_gate"].sum()
        ),
        "expert_coverage": float(
            result.partially_pooled_expert.diagnostics["expert_coverage"].mean()
        ),
        "eligible_regions_per_fold": [
            int(value)
            for value in result.partially_pooled_expert.diagnostics[
                "eligible_regions"
            ]
        ],
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(result.candidate_summary), flush=True)
    print(
        "\nOnly the fixed 80:20 partial-pooling candidate is selection-eligible; "
        "hard routing and regional-prior layers are stress diagnostics.",
        flush=True,
    )


def _load_modelling_data():
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.loc[
        :,
        [
            "selection_eligible",
            "mean_accuracy",
            "accuracy_change",
            "fold_wins",
            "worst_fold_change",
            "repair_recall_change",
            "log_loss_change",
            "passes_gate",
        ],
    ].copy()
    for column in (
        "mean_accuracy",
        "accuracy_change",
        "worst_fold_change",
        "repair_recall_change",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

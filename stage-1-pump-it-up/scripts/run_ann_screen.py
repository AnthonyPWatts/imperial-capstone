"""Run and persist the bounded frozen-fold ANN screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "ann-screen"
GPU_SCREEN_PATH = (
    PROJECT_DIR / ".runtime" / "gpu-model-screen" / "combination-screen.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ann_evaluation import ACCEPTED_BASELINE_NAME
from ann_evaluation import ANN_CANDIDATES
from ann_evaluation import compare_ann_diversity
from ann_evaluation import evaluate_ann_blends
from ann_evaluation import evaluate_ann_candidate
from ann_evaluation import summarise_ann_blends
from ann_evaluation import summarise_ann_candidates
from data_partitioning import make_cross_validation, partition_modelling_data
from modelling_data import prepare_modelling_data


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_original = pd.read_csv(DATA_DIR / "TrainingSetValues.csv")
    labels_original = pd.read_csv(DATA_DIR / "TrainingSetLabels.csv")
    raw_competition = pd.read_csv(DATA_DIR / "TestSetValues.csv")
    modelling_data = prepare_modelling_data(
        raw_original,
        labels_original,
        raw_competition,
    )
    partitioned_data = partition_modelling_data(modelling_data)
    cross_validation = make_cross_validation(partitioned_data)
    baseline = _load_accepted_baseline(partitioned_data.cross_validation_fingerprint)

    candidate_keys = args.candidate or list(ANN_CANDIDATES)
    candidates = {}
    for key in candidate_keys:
        cache_path = OUTPUT_DIR / f"{key}.joblib"
        if cache_path.exists() and not args.force:
            evaluation = joblib.load(cache_path)
            if (
                evaluation.cross_validation_fingerprint
                != partitioned_data.cross_validation_fingerprint
            ):
                raise ValueError(f"Cached ANN trial uses different folds: {cache_path}")
            print(f"Loaded cached ANN trial: {key}")
        else:
            evaluation = evaluate_ann_candidate(
                ANN_CANDIDATES[key],
                partitioned_data,
                cross_validation,
            )
            joblib.dump(evaluation, cache_path)
        candidates[key] = evaluation

    standalone_summary = summarise_ann_candidates(baseline, candidates)
    standalone_summary.to_csv(OUTPUT_DIR / "standalone-summary.csv")
    diversity = compare_ann_diversity(partitioned_data, baseline, candidates)
    diversity.to_csv(OUTPUT_DIR / "diversity.csv")

    blends = evaluate_ann_blends(
        partitioned_data,
        cross_validation,
        baseline,
        candidates,
    )
    blend_summary = summarise_ann_blends(baseline, blends)
    blend_summary.to_csv(OUTPUT_DIR / "blend-summary.csv")
    joblib.dump(
        {
            "baseline": baseline,
            "candidates": candidates,
            "blends": blends,
            "standalone_summary": standalone_summary,
            "blend_summary": blend_summary,
            "diversity": diversity,
        },
        OUTPUT_DIR / "ann-screen.joblib",
    )

    best_standalone = standalone_summary.iloc[0]
    best_blend = blend_summary.iloc[0]
    result = {
        "baseline_accuracy": float(
            baseline.metric_summary.loc["accuracy", "mean"]
        ),
        "best_standalone": {
            "candidate": standalone_summary.index[0],
            "mean_accuracy": float(best_standalone["mean_accuracy"]),
            "accuracy_change": float(best_standalone["accuracy_change"]),
        },
        "best_blend": {
            "candidate": blend_summary.index[0][0],
            "ann_weight": float(blend_summary.index[0][1]),
            "mean_accuracy": float(best_blend["mean_accuracy"]),
            "accuracy_change": float(best_blend["accuracy_change"]),
            "fold_wins": int(best_blend["fold_wins"]),
            "worst_fold_change": float(best_blend["worst_fold_change"]),
            "passes_gate": bool(best_blend["passes_gate"]),
        },
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nStandalone candidates")
    print(standalone_summary.to_string())
    print("\nBest blends")
    print(blend_summary.head(12).to_string())
    print("\n" + json.dumps(result, indent=2))


def _load_accepted_baseline(expected_fingerprint: str):
    if not GPU_SCREEN_PATH.exists():
        raise FileNotFoundError(
            "The committed model-family evidence cache is required: "
            f"{GPU_SCREEN_PATH}"
        )
    screen = joblib.load(GPU_SCREEN_PATH)
    baseline = screen["evaluations"][ACCEPTED_BASELINE_NAME]
    if baseline.cross_validation_fingerprint != expected_fingerprint:
        raise ValueError("Accepted baseline and ANN screen use different frozen folds.")
    return baseline


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        choices=tuple(ANN_CANDIDATES),
        help="Run one candidate; repeat to select several. Default: all.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit and replace any selected cached ANN trials.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

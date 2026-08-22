"""Run the bounded no-refit third-voter screen."""

from __future__ import annotations

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "expanded-ensemble-screen"
GEOGRAPHY_BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
GPU_MODEL_DIR = PROJECT_DIR / ".runtime" / "gpu-model-screen"
ANN_MODEL_PATH = PROJECT_DIR / ".runtime" / "ann-screen" / "relu_128_64.joblib"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from expanded_ensemble_evaluation import evaluate_expanded_ensembles
from modelling_data import prepare_modelling_data


def main() -> None:
    """Load persisted OOF components, evaluate fixed recipes and save evidence."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned_data = partition_modelling_data(_load_modelling_data())
    cross_validation = make_cross_validation(partitioned_data)
    baseline = joblib.load(GEOGRAPHY_BASELINE_PATH)
    voters = {
        evaluation.model_name: evaluation
        for evaluation in (
            _load_gpu_evaluation("catboost-d8-current-features.joblib"),
            _load_gpu_evaluation(
                "lightgbm-leaves-63-bagged-current-one-hot.joblib"
            ),
            joblib.load(ANN_MODEL_PATH),
        )
    }
    result = evaluate_expanded_ensembles(
        partitioned_data,
        cross_validation,
        accepted=baseline.blend,
        xgboost=baseline.xgboost,
        random_forest=baseline.random_forest,
        voters=voters,
    )
    result.candidate_summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    result.component_summary.to_csv(OUTPUT_DIR / "component-summary.csv")
    joblib.dump(result, OUTPUT_DIR / "expanded-ensemble-screen.joblib")
    payload = {
        "baseline_accuracy": float(
            baseline.blend.metric_summary.loc["accuracy", "mean"]
        ),
        "candidate_count": len(result.candidate_summary),
        "gate_passers": int(result.candidate_summary["passes_gate"].sum()),
        "best_candidate": str(result.candidate_summary.index[0]),
        "best_accuracy": float(result.candidate_summary.iloc[0]["mean_accuracy"]),
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(result.candidate_summary), flush=True)
    print(
        "\nGate: mean gain >= 0.10 pp, at least 3/5 fold wins, "
        "worst fold >= -0.25 pp, repair recall loss <= 2 pp.",
        flush=True,
    )


def _load_modelling_data():
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _load_gpu_evaluation(filename: str):
    return joblib.load(GPU_MODEL_DIR / filename)["evaluation"]


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.loc[
        :,
        [
            "mean_accuracy",
            "accuracy_change",
            "fold_wins",
            "worst_fold_change",
            "repair_recall_change",
            "log_loss_change",
            "brier_score_change",
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


if __name__ == "__main__":
    main()

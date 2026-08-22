"""Run and persist the bounded training-only outlier-filtering screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "outlier-filtering-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from outlier_filtering_evaluation import OUTLIER_FILTER_POLICIES
from outlier_filtering_evaluation import evaluate_outlier_filter
from outlier_filtering_evaluation import summarise_outlier_trials


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned_data = partition_modelling_data(_load_modelling_data())
    baseline = joblib.load(BASELINE_PATH)
    _validate_baseline(partitioned_data, baseline)
    selected_keys = args.policy or list(OUTLIER_FILTER_POLICIES)
    trials = []
    for position, key in enumerate(selected_keys, start=1):
        path = OUTPUT_DIR / f"frozen-{key}.joblib"
        if path.exists() and not args.force:
            trial = joblib.load(path)
            if (
                trial.blend.cross_validation_fingerprint
                != partitioned_data.cross_validation_fingerprint
            ):
                raise ValueError(f"Cached outlier trial uses different folds: {path}")
            print(
                f"[{position}/{len(selected_keys)}] Loaded completed policy {key}.",
                flush=True,
            )
        else:
            print(
                f"[{position}/{len(selected_keys)}] Evaluating policy {key}.",
                flush=True,
            )
            trial = evaluate_outlier_filter(
                OUTLIER_FILTER_POLICIES[key],
                partitioned_data,
                make_cross_validation(partitioned_data),
                baseline.xgboost,
            )
            joblib.dump(trial, path)
        trials.append(trial)
        print(
            f"{key}: accuracy="
            f"{trial.blend.metric_summary.loc['accuracy', 'mean']:.4%}; "
            f"mean rows removed="
            f"{trial.removal_diagnostics['removed_rows'].mean():.1f}.",
            flush=True,
        )

    summary = summarise_outlier_trials(
        baseline.blend,
        partitioned_data.y_development,
        trials,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    removal_summary = pd.concat(
        {trial.policy.key: trial.removal_diagnostics for trial in trials},
        names=["policy", "validation_fold"],
    )
    removal_summary.to_csv(OUTPUT_DIR / "removal-diagnostics.csv")
    component_crosses = _evaluate_component_crosses(
        partitioned_data,
        baseline,
        trials,
    )
    cross_summary = _summarise_component_crosses(
        baseline.blend,
        component_crosses,
    )
    cross_summary.to_csv(OUTPUT_DIR / "component-cross-summary.csv")
    result = {
        "candidate_count": len(trials),
        "selection_gate_passers": int(summary["passes_gate"].sum()),
        "best_policy": str(summary.index[0]),
        "best_accuracy": float(summary.iloc[0]["mean_accuracy"]),
        "baseline_accuracy": float(
            baseline.blend.metric_summary.loc["accuracy", "mean"]
        ),
        "best_component_cross": str(cross_summary.index[0]),
        "best_component_cross_accuracy": float(
            cross_summary.iloc[0]["mean_accuracy"]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)
    print("\nFixed component crosses:", flush=True)
    print(_format_cross_summary(cross_summary), flush=True)
    print(
        "\nGate: mean gain >= 0.10 pp, at least 3/5 fold wins, "
        "worst fold >= -0.25 pp, repair recall loss <= 2 pp.",
        flush=True,
    )
    print(
        "The local test remains closed and no competition prediction was made.",
        flush=True,
    )


def _load_modelling_data():
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _validate_baseline(partitioned_data, baseline) -> None:
    for evaluation in (baseline.xgboost, baseline.random_forest, baseline.blend):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError("Accepted baseline uses different frozen folds.")


def _evaluate_component_crosses(partitioned_data, baseline, trials):
    cross_validation = make_cross_validation(partitioned_data)
    crosses = {}
    for trial in trials:
        key = f"accepted_xgboost__{trial.policy.key}_random_forest"
        crosses[key] = evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            baseline.xgboost,
            trial.random_forest,
            weights=[0.55, 0.45],
            model_name=(
                f"Accepted XGBoost + {trial.policy.label} Random Forest"
            ),
        )
        key = f"{trial.policy.key}_xgboost__accepted_random_forest"
        crosses[key] = evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            trial.xgboost,
            baseline.random_forest,
            weights=[0.55, 0.45],
            model_name=(
                f"{trial.policy.label} XGBoost + accepted Random Forest"
            ),
        )
    return crosses


def _summarise_component_crosses(accepted, crosses) -> pd.DataFrame:
    baseline_accuracy = float(accepted.metric_summary.loc["accuracy", "mean"])
    baseline_repair = float(
        accepted.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = []
    for key, candidate in crosses.items():
        fold_change = (
            candidate.fold_metrics["accuracy"]
            - accepted.fold_metrics["accuracy"]
        )
        accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        accuracy_change = accuracy - baseline_accuracy
        repair_change = repair - baseline_repair
        rows.append(
            {
                "component_cross": key,
                "model_name": candidate.model_name,
                "mean_accuracy": accuracy,
                "accuracy_change": accuracy_change,
                "fold_wins": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall_change": repair_change,
                "passes_gate": (
                    accuracy_change >= 0.001
                    and int(fold_change.gt(0).sum()) >= 3
                    and float(fold_change.min()) >= -0.0025
                    and repair_change >= -0.02
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("component_cross")
        .sort_values(
            ["mean_accuracy", "component_cross"],
            ascending=[False, True],
        )
    )


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.loc[
        :,
        [
            "mean_accuracy",
            "accuracy_change",
            "fold_wins",
            "worst_fold_change",
            "repair_recall_change",
            "mean_removed_share",
            "log_loss_change",
            "passes_gate",
        ],
    ].copy()
    for column in (
        "mean_accuracy",
        "accuracy_change",
        "worst_fold_change",
        "repair_recall_change",
        "mean_removed_share",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    display["log_loss_change"] = display["log_loss_change"].map(
        lambda value: f"{value:+.6f}"
    )
    return display.to_string()


def _format_cross_summary(summary: pd.DataFrame) -> str:
    display = summary.loc[
        :,
        [
            "mean_accuracy",
            "accuracy_change",
            "fold_wins",
            "worst_fold_change",
            "repair_recall_change",
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
    parser.add_argument(
        "--policy",
        action="append",
        choices=tuple(OUTLIER_FILTER_POLICIES),
        help="Run one or more fixed policies; default is the complete screen.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

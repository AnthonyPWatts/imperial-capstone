"""Run and persist the cross-fitted multiclass target-encoding screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "target-encoding-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data
from target_encoding_evaluation import evaluate_target_encoding_policy
from target_encoding_evaluation import summarise_target_encoding_trials
from target_encoding_features import TARGET_ENCODING_POLICIES
from target_encoding_features import summarise_target_encoding_policies


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned_data = partition_modelling_data(_load_modelling_data())
    baseline = joblib.load(BASELINE_PATH)
    _validate_baseline(partitioned_data, baseline)
    selected_keys = args.policy or list(TARGET_ENCODING_POLICIES)
    trials = []
    for position, key in enumerate(selected_keys, start=1):
        path = OUTPUT_DIR / f"frozen-{key}.joblib"
        if path.exists() and not args.force:
            trial = joblib.load(path)
            if (
                trial.blend.cross_validation_fingerprint
                != partitioned_data.cross_validation_fingerprint
            ):
                raise ValueError(
                    f"Cached target-encoding trial uses different folds: {path}"
                )
            print(
                f"[{position}/{len(selected_keys)}] Loaded completed policy {key}.",
                flush=True,
            )
        else:
            print(
                f"[{position}/{len(selected_keys)}] Evaluating policy {key}.",
                flush=True,
            )
            trial = evaluate_target_encoding_policy(
                TARGET_ENCODING_POLICIES[key],
                partitioned_data,
                make_cross_validation(partitioned_data),
            )
            joblib.dump(trial, path)
        trials.append(trial)
        print(
            f"{key}: accuracy="
            f"{trial.blend.metric_summary.loc['accuracy', 'mean']:.4%}.",
            flush=True,
        )

    summary = summarise_target_encoding_trials(baseline.blend, trials)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    summarise_target_encoding_policies().to_csv(
        OUTPUT_DIR / "policy-register.csv"
    )
    crosses = _evaluate_component_crosses(partitioned_data, baseline, trials)
    cross_summary = _summarise_component_crosses(baseline.blend, crosses)
    cross_summary.to_csv(OUTPUT_DIR / "component-cross-summary.csv")
    contributions = _evaluate_bounded_contributions(
        partitioned_data,
        baseline,
        trials,
    )
    contribution_summary = (
        _summarise_component_crosses(baseline.blend, contributions)
        if contributions
        else None
    )
    if contribution_summary is not None:
        contribution_summary.to_csv(
            OUTPUT_DIR / "bounded-contribution-summary.csv"
        )
    result = {
        "candidate_count": len(trials),
        "selection_gate_passers": int(summary["passes_gate"].sum()),
        "best_policy": str(summary.index[0]),
        "best_accuracy": float(summary.iloc[0]["mean_accuracy"]),
        "best_component_cross": str(cross_summary.index[0]),
        "best_component_cross_accuracy": float(
            cross_summary.iloc[0]["mean_accuracy"]
        ),
        "best_bounded_contribution": (
            str(contribution_summary.index[0])
            if contribution_summary is not None
            else None
        ),
        "best_bounded_contribution_accuracy": (
            float(contribution_summary.iloc[0]["mean_accuracy"])
            if contribution_summary is not None
            else None
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
    if contribution_summary is not None:
        print("\nBounded organisation-XGBoost contributions:", flush=True)
        print(_format_cross_summary(contribution_summary), flush=True)
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


def _evaluate_bounded_contributions(partitioned_data, baseline, trials):
    by_key = {trial.policy.key: trial for trial in trials}
    if "organisation_identity" not in by_key:
        return {}
    organisation_xgboost = by_key["organisation_identity"].xgboost
    cross_validation = make_cross_validation(partitioned_data)
    contributions = {}
    for contribution in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        key = f"organisation_xgboost_{contribution:.0%}"
        contributions[key] = evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            baseline.xgboost,
            baseline.random_forest,
            organisation_xgboost,
            weights=[
                0.55 * (1.0 - contribution),
                0.45 * (1.0 - contribution),
                contribution,
            ],
            model_name=(
                f"{contribution:.0%} organisation-target XGBoost + "
                "accepted ensemble"
            ),
        )
    return contributions


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.loc[
        :,
        [
            "mean_accuracy",
            "accuracy_change",
            "fold_wins",
            "worst_fold_change",
            "repair_recall_change",
            "xgboost_accuracy",
            "random_forest_accuracy",
            "passes_gate",
        ],
    ].copy()
    for column in (
        "mean_accuracy",
        "accuracy_change",
        "worst_fold_change",
        "repair_recall_change",
        "xgboost_accuracy",
        "random_forest_accuracy",
    ):
        display[column] = display[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:.4%}"
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
        choices=tuple(TARGET_ENCODING_POLICIES),
        help="Run one or more fixed policies; default is the complete screen.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

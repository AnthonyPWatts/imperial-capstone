"""Run the bounded triangular fuzzy-membership target screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "fuzzy-target-screen"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from fuzzy_target_evaluation import compare_candidate_evaluations_to_baseline
from fuzzy_target_evaluation import compare_fuzzy_blends_to_baseline
from fuzzy_target_evaluation import evaluate_fuzzy_policy
from fuzzy_target_evaluation import evaluate_fuzzy_component_crosses
from fuzzy_target_evaluation import FUZZY_POLICIES
from fuzzy_target_evaluation import summarise_candidate_evaluations
from fuzzy_target_evaluation import summarise_fuzzy_trials
from modelling_data import prepare_modelling_data


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned_data = partition_modelling_data(_load_modelling_data())
    cross_validation = make_cross_validation(partitioned_data)
    baseline_trial = joblib.load(BASELINE_PATH)
    baseline = baseline_trial.blend
    if (
        baseline.cross_validation_fingerprint
        != partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError("Accepted baseline uses a different fold fingerprint.")

    trials = {}
    for policy in FUZZY_POLICIES:
        cache_path = OUTPUT_DIR / f"{policy.key}.joblib"
        if cache_path.exists() and not args.force:
            trial = joblib.load(cache_path)
            _validate_cache(trial, policy, partitioned_data, cache_path)
            print(f"Loaded {policy.key}.", flush=True)
        else:
            trial = evaluate_fuzzy_policy(
                policy,
                partitioned_data,
                cross_validation,
            )
            joblib.dump(trial, cache_path)
        trials[policy.key] = trial

    summary = summarise_fuzzy_trials(partitioned_data, trials)
    gate = compare_fuzzy_blends_to_baseline(baseline, trials)
    crosses = evaluate_fuzzy_component_crosses(
        partitioned_data,
        cross_validation,
        hard_xgboost=baseline_trial.xgboost,
        hard_random_forest=baseline_trial.random_forest,
        trials=trials,
    )
    cross_summary = summarise_candidate_evaluations(
        partitioned_data,
        crosses,
    ).join(compare_candidate_evaluations_to_baseline(baseline, crosses))
    membership_summary = pd.concat(
        {
            key: trial.membership_summary
            for key, trial in trials.items()
        },
        names=["policy"],
    )
    blend_summary = summary.xs("blend", level="component").join(gate)
    best_policy = str(blend_summary.index[0])
    all_candidates = {
        **{
            f"{key}__both_fuzzy": trial.blend
            for key, trial in trials.items()
        },
        **crosses,
    }
    candidate_summary = summarise_candidate_evaluations(
        partitioned_data,
        all_candidates,
    ).join(
        compare_candidate_evaluations_to_baseline(
            baseline,
            all_candidates,
        )
    )
    best_candidate = str(candidate_summary.index[0])
    passing = candidate_summary.index[
        candidate_summary["passes_gate"]
    ].tolist()
    selected_candidate = passing[0] if passing else None
    local_test = None

    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    blend_summary.to_csv(OUTPUT_DIR / "blend-summary.csv")
    cross_summary.to_csv(OUTPUT_DIR / "cross-summary.csv")
    candidate_summary.to_csv(OUTPUT_DIR / "all-fuzzy-candidate-summary.csv")
    membership_summary.to_csv(OUTPUT_DIR / "membership-summary.csv")
    if local_test is not None:
        local_test.metrics.to_frame("value").to_csv(
            OUTPUT_DIR / "local-test-metrics.csv"
        )
        local_test.confusion_counts.to_csv(
            OUTPUT_DIR / "local-test-confusion.csv"
        )
    joblib.dump(
        {
            "baseline": baseline,
            "trials": trials,
            "crosses": crosses,
            "candidate_summary": summary,
            "blend_summary": blend_summary,
            "cross_summary": cross_summary,
            "all_fuzzy_candidate_summary": candidate_summary,
            "membership_summary": membership_summary,
            "best_policy": best_policy,
            "best_candidate": best_candidate,
            "selected_candidate": selected_candidate,
            "local_test": local_test,
        },
        OUTPUT_DIR / "fuzzy-target-screen.joblib",
    )
    payload = {
        "development_rows": len(partitioned_data.y_development),
        "policies": len(trials),
        "components_per_policy": 3,
        "baseline_accuracy": float(
            baseline.metric_summary.loc["accuracy", "mean"]
        ),
        "baseline_repair_recall": float(
            baseline.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        ),
        "best_policy": best_policy,
        "best_accuracy": float(blend_summary.loc[best_policy, "mean_accuracy"]),
        "best_accuracy_change": float(
            blend_summary.loc[best_policy, "accuracy_change"]
        ),
        "best_repair_recall": float(
            blend_summary.loc[best_policy, "repair_recall"]
        ),
        "best_candidate": best_candidate,
        "best_candidate_accuracy": float(
            candidate_summary.loc[best_candidate, "mean_accuracy"]
        ),
        "best_candidate_accuracy_change": float(
            candidate_summary.loc[best_candidate, "accuracy_change"]
        ),
        "selected_candidate": selected_candidate,
        "local_test_accuracy": None,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nAccepted hard-label baseline", flush=True)
    print(
        f"accuracy={baseline.metric_summary.loc['accuracy', 'mean']:.4%}; "
        "repair recall="
        f"{baseline.metric_summary.loc['recall: functional needs repair', 'mean']:.4%}",
        flush=True,
    )
    print("\nFuzzy 55:45 blends", flush=True)
    print(_format_blend_summary(blend_summary), flush=True)
    print("\nOne-fuzzy-component crosses", flush=True)
    print(_format_cross_summary(cross_summary), flush=True)
    if selected_candidate is None:
        print("\nNo fuzzy-derived candidate passed the promotion gate.", flush=True)
        print("The labelled local test was not refitted or rescored.", flush=True)
    else:
        print(f"\nPromoted development candidate: {selected_candidate}.", flush=True)
        print("Local-test confirmation requires a separately frozen refit.", flush=True)


def _validate_cache(trial, policy, partitioned_data, path: Path) -> None:
    if trial.policy != policy:
        raise ValueError(f"Cached fuzzy policy does not match: {path}.")
    for evaluation in (trial.xgboost, trial.random_forest, trial.blend):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError(f"Cached fuzzy evaluation uses different folds: {path}.")


def _load_modelling_data():
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _format_blend_summary(summary: pd.DataFrame) -> str:
    display = summary.loc[
        :,
        [
            "adjacent_overlap",
            "effective_repair_share",
            "mean_accuracy",
            "accuracy_change",
            "fold_wins",
            "worst_fold_change",
            "repair_recall",
            "repair_recall_change",
            "log_loss",
            "passes_gate",
        ],
    ].copy()
    for column in (
        "effective_repair_share",
        "mean_accuracy",
        "repair_recall",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    for column in (
        "accuracy_change",
        "worst_fold_change",
        "repair_recall_change",
    ):
        display[column] = display[column].map(lambda value: f"{value * 100:+.4f} pp")
    return display.to_string()


def _format_cross_summary(summary: pd.DataFrame) -> str:
    display = summary.loc[
        :,
        [
            "mean_accuracy",
            "accuracy_change",
            "fold_wins",
            "worst_fold_change",
            "repair_recall",
            "repair_recall_change",
            "log_loss",
            "passes_gate",
        ],
    ].copy()
    for column in ("mean_accuracy", "repair_recall"):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    for column in (
        "accuracy_change",
        "worst_fold_change",
        "repair_recall_change",
    ):
        display[column] = display[column].map(lambda value: f"{value * 100:+.4f} pp")
    return display.to_string()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit all fuzzy policies even when fold-compatible caches exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""Run and persist the fixed-model geography feature-family screen."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "geography-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from geography_evaluation import evaluate_geography_policy
from geography_evaluation import evaluate_cross_policy_hybrids
from geography_evaluation import make_lga_grouped_partition
from geography_evaluation import summarise_cross_policy_hybrids
from geography_evaluation import summarise_geography_trials
from geography_features import GEOGRAPHY_POLICIES
from geography_features import summarise_geography_policies
from modelling_data import prepare_modelling_data


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = _load_modelling_data()
    partitioned_data = partition_modelling_data(modelling_data)

    if args.grouped_policy:
        _run_grouped_sensitivity(args, partitioned_data)
        return

    selected_keys = args.policy or list(GEOGRAPHY_POLICIES)
    trials = []
    for position, key in enumerate(selected_keys, start=1):
        path = OUTPUT_DIR / f"frozen-{key}.joblib"
        if path.exists() and not args.force:
            print(
                f"[{position}/{len(selected_keys)}] Loading completed policy {key}.",
                flush=True,
            )
            trials.append(joblib.load(path))
            continue
        print(
            f"[{position}/{len(selected_keys)}] Evaluating geography policy {key}.",
            flush=True,
        )
        trial = evaluate_geography_policy(
            GEOGRAPHY_POLICIES[key],
            partitioned_data,
            make_cross_validation(partitioned_data),
        )
        joblib.dump(trial, path)
        trials.append(trial)
        print(
            f"Persisted {key}: accuracy="
            f"{trial.blend.metric_summary.loc['accuracy', 'mean']:.4%}.",
            flush=True,
        )

    if "baseline" in {trial.policy.key for trial in trials}:
        summary = summarise_geography_trials(trials)
        _write_summary(summary, "frozen-summary")
        print(_format_summary(summary), flush=True)
        by_key = {trial.policy.key: trial for trial in trials}
        if "coordinate_centroids" in by_key:
            hybrids = evaluate_cross_policy_hybrids(
                partitioned_data,
                make_cross_validation(partitioned_data),
                by_key["baseline"],
                by_key["coordinate_centroids"],
            )
            for evaluation in hybrids:
                key = (
                    evaluation.model_name.casefold()
                    .replace(" + ", "-")
                    .replace(" ", "-")
                )
                joblib.dump(evaluation, OUTPUT_DIR / f"hybrid-{key}.joblib")
            hybrid_summary = summarise_cross_policy_hybrids(
                by_key["baseline"],
                hybrids,
            )
            _write_summary(hybrid_summary, "hybrid-summary")
    policy_table = summarise_geography_policies()
    policy_table.to_csv(OUTPUT_DIR / "policy-register.csv")


def _run_grouped_sensitivity(args, partitioned_data) -> None:
    grouped_partition, grouped_cv, fold_summary = make_lga_grouped_partition(
        partitioned_data
    )
    keys = list(dict.fromkeys(["baseline", *args.grouped_policy]))
    trials = []
    for position, key in enumerate(keys, start=1):
        path = OUTPUT_DIR / f"lga-grouped-{key}.joblib"
        if path.exists() and not args.force:
            print(
                f"[{position}/{len(keys)}] Loading grouped policy {key}.",
                flush=True,
            )
            trials.append(joblib.load(path))
            continue
        print(
            f"[{position}/{len(keys)}] Evaluating LGA-disjoint policy {key}.",
            flush=True,
        )
        trial = evaluate_geography_policy(
            GEOGRAPHY_POLICIES[key],
            grouped_partition,
            grouped_cv,
            evaluation_label="LGA-disjoint folds",
        )
        joblib.dump(trial, path)
        trials.append(trial)
    summary = summarise_geography_trials(trials)
    _write_summary(summary, "lga-grouped-summary")
    fold_summary.to_csv(OUTPUT_DIR / "lga-grouped-folds.csv")
    print(_format_summary(summary), flush=True)


def _load_modelling_data():
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _write_summary(summary: pd.DataFrame, stem: str) -> None:
    summary.to_csv(OUTPUT_DIR / f"{stem}.csv")
    records = json.loads(summary.reset_index().to_json(orient="records"))
    (OUTPUT_DIR / f"{stem}.json").write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.loc[
        :,
        [
            "mean_accuracy",
            "accuracy_change",
            "fold_wins",
            "worst_fold_change",
            "repair_recall",
            "passes_gate",
        ],
    ].copy()
    for column in (
        "mean_accuracy",
        "accuracy_change",
        "worst_fold_change",
        "repair_recall",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        action="append",
        choices=tuple(GEOGRAPHY_POLICIES),
        help="Run one or more frozen-fold policies; default is the full screen.",
    )
    parser.add_argument(
        "--grouped-policy",
        action="append",
        choices=tuple(key for key in GEOGRAPHY_POLICIES if key != "baseline"),
        help=(
            "Run LGA-disjoint sensitivity for baseline and the named policy. "
            "May be repeated."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

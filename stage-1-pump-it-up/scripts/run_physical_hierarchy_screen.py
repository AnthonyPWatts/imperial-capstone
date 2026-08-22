"""Run and persist the four bounded physical-category hierarchy screens."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "physical-hierarchy-screen"
MODEL_SCREEN_PATH = (
    PROJECT_DIR / ".runtime" / "gpu-model-screen" / "combination-screen.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from modelling_data import prepare_modelling_data
from physical_hierarchy_evaluation import evaluate_physical_hierarchy_policy
from physical_hierarchy_evaluation import evaluate_component_hybrids
from physical_hierarchy_evaluation import make_cached_baseline_trial
from physical_hierarchy_evaluation import summarise_component_hybrids
from physical_hierarchy_evaluation import summarise_physical_hierarchy_trials
from physical_hierarchy_features import HIERARCHY_FAMILIES
from physical_hierarchy_features import PHYSICAL_HIERARCHY_POLICIES
from physical_hierarchy_features import summarise_physical_hierarchy_policies


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned_data = partition_modelling_data(_load_modelling_data())
    cross_validation = make_cross_validation(partitioned_data)
    accepted_evaluations = _load_accepted_evaluations()
    selected_families = args.family or list(HIERARCHY_FAMILIES)

    combined_summaries = []
    for family_position, family_key in enumerate(selected_families, start=1):
        policies = PHYSICAL_HIERARCHY_POLICIES[family_key]
        print(
            f"\nFamily {family_position}/{len(selected_families)}: {family_key}",
            flush=True,
        )
        baseline = make_cached_baseline_trial(
            policies["baseline"],
            partitioned_data,
            cross_validation,
            accepted_evaluations,
        )
        trials = [baseline]
        challengers = [
            policy for key, policy in policies.items() if key != "baseline"
        ]
        for position, policy in enumerate(challengers, start=1):
            path = OUTPUT_DIR / f"{family_key}-{policy.key}.joblib"
            if path.exists() and not args.force:
                print(
                    f"[{position}/{len(challengers)}] Loading {policy.key}.",
                    flush=True,
                )
                trial = joblib.load(path)
                if (
                    trial.blend.cross_validation_fingerprint
                    != partitioned_data.cross_validation_fingerprint
                ):
                    raise ValueError(f"Cached policy uses different folds: {path}")
            else:
                print(
                    f"[{position}/{len(challengers)}] Evaluating {policy.key}.",
                    flush=True,
                )
                trial = evaluate_physical_hierarchy_policy(
                    policy,
                    partitioned_data,
                    cross_validation,
                )
                joblib.dump(trial, path)
            trials.append(trial)
            print(
                f"{family_key}:{policy.key} accuracy="
                f"{trial.blend.metric_summary.loc['accuracy', 'mean']:.4%}.",
                flush=True,
            )

        summary = summarise_physical_hierarchy_trials(trials)
        _write_summary(summary, family_key)
        summarise_physical_hierarchy_policies(family_key).to_csv(
            OUTPUT_DIR / f"{family_key}-policy-register.csv"
        )
        print(_format_summary(summary), flush=True)
        tagged = summary.copy()
        tagged.insert(0, "family", family_key)
        combined_summaries.append(tagged.reset_index())

    combined = pd.concat(combined_summaries, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "combined-summary.csv", index=False)
    if set(selected_families) == set(HIERARCHY_FAMILIES):
        _run_component_hybrids(
            partitioned_data,
            cross_validation,
            accepted_evaluations,
        )


def _run_component_hybrids(
    partitioned_data,
    cross_validation,
    accepted_evaluations,
) -> None:
    baseline = make_cached_baseline_trial(
        PHYSICAL_HIERARCHY_POLICIES["extraction"]["baseline"],
        partitioned_data,
        cross_validation,
        accepted_evaluations,
    )
    trials = {
        "baseline": baseline,
        "extraction_group": joblib.load(
            OUTPUT_DIR / "extraction-group_only.joblib"
        ),
        "source_class": joblib.load(
            OUTPUT_DIR / "source-source_plus_class.joblib"
        ),
        "quality_group": joblib.load(OUTPUT_DIR / "quality-group_only.joblib"),
        "waterpoint_both": joblib.load(
            OUTPUT_DIR / "waterpoint-both_levels.joblib"
        ),
    }
    hybrids = evaluate_component_hybrids(
        partitioned_data,
        cross_validation,
        trials,
    )
    summary = summarise_component_hybrids(baseline.blend, hybrids)
    summary.to_csv(OUTPUT_DIR / "component-hybrids.csv")
    joblib.dump(
        {"evaluations": hybrids, "summary": summary},
        OUTPUT_DIR / "component-hybrids.joblib",
    )
    print("\nComponent hybrids", flush=True)
    print(_format_hybrid_summary(summary), flush=True)


def _load_accepted_evaluations():
    if not MODEL_SCREEN_PATH.exists():
        raise FileNotFoundError(
            "The accepted model-family evidence cache is required: "
            f"{MODEL_SCREEN_PATH}"
        )
    return joblib.load(MODEL_SCREEN_PATH)["evaluations"]


def _load_modelling_data():
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _write_summary(summary: pd.DataFrame, family_key: str) -> None:
    summary.to_csv(OUTPUT_DIR / f"{family_key}-summary.csv")
    records = json.loads(summary.reset_index().to_json(orient="records"))
    (OUTPUT_DIR / f"{family_key}-summary.json").write_text(
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
            "parsimony_candidate",
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


def _format_hybrid_summary(summary: pd.DataFrame) -> str:
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
        "--family",
        action="append",
        choices=tuple(HIERARCHY_FAMILIES),
        help="Run one family; repeat to select several. Default: all four.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

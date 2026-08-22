"""Run the full binary-training, three-class-evaluation modelling sequence."""

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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "binary-reduction-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from binary_reduction_evaluation import ACCELERATED_SPECS
from binary_reduction_evaluation import compare_binary_diversity
from binary_reduction_evaluation import compare_to_reference
from binary_reduction_evaluation import evaluate_binary_accelerated_candidate
from binary_reduction_evaluation import evaluate_binary_ensembles
from binary_reduction_evaluation import evaluate_binary_pipeline_candidate
from binary_reduction_evaluation import fit_selected_binary_recipe_on_local_test
from binary_reduction_evaluation import summarise_binary_evaluations
from data_partitioning import make_cross_validation, partition_modelling_data
from modelling_data import prepare_modelling_data


PIPELINE_ORDER = (
    "decision_tree",
    "logistic_regression",
    "gaussian_naive_bayes",
    "knn",
    "extra_trees",
    "histogram_boosting",
    "random_forest",
    "mlp_relu_128_64",
)
ACCELERATED_ORDER = (
    "catboost_d8",
    "xgboost_child1",
    "lightgbm_bagged63",
)


def main() -> None:
    args = _parse_arguments()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned_data = partition_modelling_data(_load_modelling_data())
    cross_validation = make_cross_validation(partitioned_data)
    standalone = {}

    for key in PIPELINE_ORDER:
        standalone[key] = _load_or_run_pipeline(
            key,
            partitioned_data,
            cross_validation,
            force=args.force,
        )
    for key in ACCELERATED_ORDER:
        standalone[key] = _load_or_run_accelerated(
            key,
            partitioned_data,
            cross_validation,
            force=args.force,
        )

    standalone_summary = summarise_binary_evaluations(
        partitioned_data,
        cross_validation,
        standalone,
    )
    best_standalone_key = str(standalone_summary.index[0])
    print("\nStandalone candidates", flush=True)
    print(_format_summary(standalone_summary), flush=True)

    ensembles = evaluate_binary_ensembles(
        partitioned_data,
        cross_validation,
        standalone,
    )
    ensemble_summary = summarise_binary_evaluations(
        partitioned_data,
        cross_validation,
        ensembles,
    )
    combined = {**standalone, **ensembles}
    combined_summary = summarise_binary_evaluations(
        partitioned_data,
        cross_validation,
        combined,
    )
    gate = compare_to_reference(combined, reference_key=best_standalone_key)
    combined_summary = combined_summary.join(gate)
    print("\nEnsemble candidates", flush=True)
    print(_format_summary(ensemble_summary), flush=True)

    passing_ensembles = [
        key
        for key in ensembles
        if bool(gate.loc[key, "passes_gate"])
    ]
    if passing_ensembles:
        selected_key = max(
            passing_ensembles,
            key=lambda key: combined_summary.loc[
                key,
                "mean_three_class_accuracy",
            ],
        )
        selection_reason = "best ensemble passing the predeclared gate"
    else:
        selected_key = best_standalone_key
        selection_reason = "no ensemble passed the gate; retained best standalone"

    local_test = fit_selected_binary_recipe_on_local_test(
        partitioned_data,
        selected_key=selected_key,
        component_evaluations=standalone,
    )
    diversity = compare_binary_diversity(partitioned_data, standalone)

    standalone_summary.to_csv(OUTPUT_DIR / "standalone-summary.csv")
    ensemble_summary.to_csv(OUTPUT_DIR / "ensemble-summary.csv")
    combined_summary.to_csv(OUTPUT_DIR / "combined-summary.csv")
    diversity.to_csv(OUTPUT_DIR / "standalone-diversity.csv")
    local_test.metrics.to_frame("value").to_csv(
        OUTPUT_DIR / "local-test-metrics.csv"
    )
    local_test.confusion_counts.to_csv(
        OUTPUT_DIR / "local-test-confusion.csv"
    )
    joblib.dump(
        {
            "standalone": standalone,
            "ensembles": ensembles,
            "standalone_summary": standalone_summary,
            "ensemble_summary": ensemble_summary,
            "combined_summary": combined_summary,
            "diversity": diversity,
            "selected_key": selected_key,
            "selection_reason": selection_reason,
            "local_test": local_test,
        },
        OUTPUT_DIR / "binary-reduction-screen.joblib",
    )
    payload = {
        "development_rows": len(partitioned_data.y_development),
        "development_repair_rows": int(
            partitioned_data.y_development.eq("functional needs repair").sum()
        ),
        "standalone_candidates": len(standalone),
        "ensemble_candidates": len(ensembles),
        "best_standalone": best_standalone_key,
        "best_standalone_accuracy": float(
            standalone_summary.iloc[0]["mean_three_class_accuracy"]
        ),
        "selected_key": selected_key,
        "selection_reason": selection_reason,
        "selected_development_accuracy": float(
            combined_summary.loc[selected_key, "mean_three_class_accuracy"]
        ),
        "selected_local_test_accuracy": float(
            local_test.metrics["three_class_accuracy"]
        ),
        "selected_local_test_conditional_binary_accuracy": float(
            local_test.metrics["conditional_binary_accuracy"]
        ),
        "selected_local_test_repair_recall": float(
            local_test.metrics["repair_recall"]
        ),
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nSelection", flush=True)
    print(f"{selected_key}: {selection_reason}.", flush=True)
    print("\nLocal three-class test", flush=True)
    print(local_test.metrics.to_string(), flush=True)


def _load_or_run_pipeline(
    key,
    partitioned_data,
    cross_validation,
    *,
    force,
):
    path = OUTPUT_DIR / f"{key}.joblib"
    if path.exists() and not force:
        evaluation = joblib.load(path)
        _validate_cache(evaluation, partitioned_data, path)
        print(f"Loaded {key}.", flush=True)
        return evaluation
    evaluation = evaluate_binary_pipeline_candidate(
        key,
        partitioned_data,
        cross_validation,
    )
    joblib.dump(evaluation, path)
    return evaluation


def _load_or_run_accelerated(
    key,
    partitioned_data,
    cross_validation,
    *,
    force,
):
    path = OUTPUT_DIR / f"{key}.joblib"
    if path.exists() and not force:
        evaluation = joblib.load(path)
        _validate_cache(evaluation, partitioned_data, path)
        print(f"Loaded {key}.", flush=True)
        return evaluation
    evaluation = evaluate_binary_accelerated_candidate(
        ACCELERATED_SPECS[key],
        partitioned_data,
        cross_validation,
    )
    joblib.dump(evaluation, path)
    return evaluation


def _validate_cache(evaluation, partitioned_data, path) -> None:
    if (
        evaluation.cross_validation_fingerprint
        != partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError(f"Cached binary evaluation uses different folds: {path}.")


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
            "mean_three_class_accuracy",
            "mean_conditional_binary_accuracy",
            "functional_recall",
            "non_functional_recall",
            "conditional_log_loss",
        ],
    ].copy()
    for column in (
        "mean_three_class_accuracy",
        "mean_conditional_binary_accuracy",
        "functional_recall",
        "non_functional_recall",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit all candidates even when fold-compatible caches exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

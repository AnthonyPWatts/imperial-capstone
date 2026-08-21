"""Run and persist one frozen-fold model trial for the GPU screen."""

from __future__ import annotations

import argparse
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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "gpu-model-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from gpu_model_evaluation import CATBOOST_VARIANTS, LIGHTGBM_VARIANTS
from gpu_model_evaluation import SKLEARN_TREE_VARIANTS
from gpu_model_evaluation import XGBOOST_VARIANTS
from gpu_model_evaluation import evaluate_gpu_candidate
from gpu_model_evaluation import make_catboost_spec, make_lightgbm_spec
from gpu_model_evaluation import make_sklearn_tree_spec
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import with_seed
from model_evaluation import evaluate_equal_weight_soft_vote
from model_evaluation import evaluate_initial_histogram_boosting
from model_evaluation import evaluate_random_forest
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

    if args.family == "catboost":
        spec = make_catboost_spec(variant=args.variant)
        if args.seed != 20260821:
            spec = with_seed(spec, seed=args.seed)
        evaluation = evaluate_gpu_candidate(
            spec,
            partitioned_data,
            cross_validation,
        )
        components = None
    elif args.family == "xgboost":
        spec = make_xgboost_spec(variant=args.variant)
        if args.seed != 20260821:
            spec = with_seed(spec, seed=args.seed)
        evaluation = evaluate_gpu_candidate(
            spec,
            partitioned_data,
            cross_validation,
        )
        components = None
    elif args.family == "lightgbm":
        spec = make_lightgbm_spec(variant=args.variant)
        if args.seed != 20260821:
            spec = with_seed(spec, seed=args.seed)
        evaluation = evaluate_gpu_candidate(
            spec,
            partitioned_data,
            cross_validation,
        )
        components = None
    elif args.family == "sklearn-tree":
        spec = make_sklearn_tree_spec(variant=args.variant)
        if args.seed != 20260821:
            spec = with_seed(spec, seed=args.seed)
        evaluation = evaluate_gpu_candidate(
            spec,
            partitioned_data,
            cross_validation,
        )
        components = None
    else:
        spec = None
        random_forest = evaluate_random_forest(
            partitioned_data,
            cross_validation,
        )
        histogram_boosting = evaluate_initial_histogram_boosting(
            partitioned_data,
            cross_validation,
        )
        evaluation = evaluate_equal_weight_soft_vote(
            partitioned_data,
            cross_validation,
            random_forest,
            histogram_boosting,
            model_name="incumbent 50% Random Forest + 50% boosting",
        )
        components = {
            "Random Forest": random_forest,
            "histogram boosting": histogram_boosting,
        }

    slug = _slug(evaluation.model_name)
    destination = OUTPUT_DIR / f"{slug}.joblib"
    summary_path = OUTPUT_DIR / f"{slug}.json"
    if (destination.exists() or summary_path.exists()) and not args.force:
        raise FileExistsError(
            f"Trial output already exists; pass --force to replace it: {destination}"
        )
    payload = {
        "spec": spec,
        "evaluation": evaluation,
        "components": components,
    }
    joblib.dump(payload, destination)
    summary = {
        "model_name": evaluation.model_name,
        "mean_accuracy": float(
            evaluation.metric_summary.loc["accuracy", "mean"]
        ),
        "fold_accuracy": evaluation.fold_metrics["accuracy"].tolist(),
        "mean_repair_recall": float(
            evaluation.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        ),
        "mean_non_functional_recall": float(
            evaluation.metric_summary.loc[
                "recall: non functional",
                "mean",
            ]
        ),
        "selected_iterations": (
            evaluation.diagnostics["selected_iterations"].astype(int).tolist()
            if "selected_iterations" in evaluation.diagnostics
            else None
        ),
        "joblib": str(destination),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "family",
        choices=(
            "catboost",
            "xgboost",
            "lightgbm",
            "sklearn-tree",
            "incumbent",
        ),
    )
    parser.add_argument("variant", nargs="?", default="")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.family == "catboost" and args.variant not in CATBOOST_VARIANTS:
        parser.error(
            "CatBoost variant must be one of: "
            + ", ".join(CATBOOST_VARIANTS)
        )
    if args.family == "xgboost" and args.variant not in XGBOOST_VARIANTS:
        parser.error(
            "XGBoost variant must be one of: "
            + ", ".join(XGBOOST_VARIANTS)
        )
    if args.family == "lightgbm" and args.variant not in LIGHTGBM_VARIANTS:
        parser.error(
            "LightGBM variant must be one of: "
            + ", ".join(LIGHTGBM_VARIANTS)
        )
    if (
        args.family == "sklearn-tree"
        and args.variant not in SKLEARN_TREE_VARIANTS
    ):
        parser.error(
            "sklearn tree variant must be one of: "
            + ", ".join(SKLEARN_TREE_VARIANTS)
        )
    if args.family == "incumbent" and args.variant:
        parser.error("The incumbent trial does not take a variant.")
    if args.family == "incumbent" and args.seed != 20260821:
        parser.error("The incumbent trial does not take a seed override.")
    return args


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


if __name__ == "__main__":
    main()

"""Evaluate one extra seed inside the promoted identity CatBoost voter."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "catboost-identity-seed-bag"
BASELINE_PATH = (
    PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
)
IDENTITY_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "catboost-identity-screen"
    / "complete-deferred-identities.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_evaluation import IDENTITY_BAG_SECOND_SEED
from catboost_identity_evaluation import evaluate_seeded_complete_identity_catboost
from data_partitioning import make_cross_validation, partition_modelling_data
from gpu_model_evaluation import INNER_STOP_SEED
from model_evaluation import evaluate_equal_weight_soft_vote
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitioned = partition_modelling_data(
        prepare_modelling_data(
            pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
            pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
            pd.read_csv(DATA_DIR / "TestSetValues.csv"),
        )
    )
    baseline = joblib.load(BASELINE_PATH)
    identity = joblib.load(IDENTITY_PATH)
    _validate_evidence(partitioned, baseline, identity.evaluation)
    cache_path = OUTPUT_DIR / f"complete-identities-seed-{IDENTITY_BAG_SECOND_SEED}.joblib"
    if cache_path.exists():
        second_seed = joblib.load(cache_path)
        if (
            second_seed.evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Cached identity seed uses different folds.")
        print(f"Loaded {cache_path}.", flush=True)
    else:
        second_seed = evaluate_seeded_complete_identity_catboost(
            partitioned,
            make_cross_validation(partitioned),
        )
        joblib.dump(second_seed, cache_path)

    seed_bag = evaluate_equal_weight_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        identity.evaluation,
        second_seed.evaluation,
        model_name="Equal complete-identity CatBoost seed bag",
    )
    leader = _vote(
        partitioned,
        baseline,
        identity.evaluation,
        "Promoted complete-identity vote",
    )
    second_seed_vote = _vote(
        partitioned,
        baseline,
        second_seed.evaluation,
        "Second-seed identity vote",
    )
    seed_bag_vote = _vote(
        partitioned,
        baseline,
        seed_bag,
        "Equal-seed-bag identity vote",
    )
    summary = _summarise(
        leader,
        identity.evaluation,
        second_seed.evaluation,
        seed_bag,
        second_seed_vote,
        seed_bag_vote,
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    original_predictions = _predictions(identity.evaluation)
    second_predictions = _predictions(second_seed.evaluation)
    result = {
        "first_seed": INNER_STOP_SEED,
        "second_seed": IDENTITY_BAG_SECOND_SEED,
        "standalone_seed_disagreement": float(
            np.mean(original_predictions != second_predictions)
        ),
        "seed_bag_vote_accuracy": float(
            summary.loc["seed_bag_identity_vote", "mean_accuracy"]
        ),
        "change_vs_promoted_identity_vote": float(
            summary.loc["seed_bag_identity_vote", "leader_change"]
        ),
        "passes_gate": bool(
            summary.loc["seed_bag_identity_vote", "passes_gate"]
        ),
        "local_test_opened": False,
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(summary), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def _vote(partitioned, baseline, identity_component, name):
    return evaluate_weighted_soft_vote(
        partitioned,
        make_cross_validation(partitioned),
        baseline.xgboost,
        baseline.random_forest,
        identity_component,
        weights=[0.44, 0.36, 0.20],
        model_name=name,
    )


def _validate_evidence(partitioned, baseline, identity) -> None:
    for evaluation in (baseline.xgboost, baseline.random_forest, identity):
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError("Identity seed comparison uses different folds.")


def _summarise(
    leader,
    identity,
    second_seed,
    seed_bag,
    second_seed_vote,
    seed_bag_vote,
):
    leader_accuracy = float(leader.metric_summary.loc["accuracy", "mean"])
    leader_repair = float(
        leader.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    rows = []
    for key, candidate in {
        "promoted_identity_vote": leader,
        "first_seed_identity_catboost": identity,
        "second_seed_identity_catboost": second_seed,
        "identity_catboost_seed_bag": seed_bag,
        "second_seed_identity_vote": second_seed_vote,
        "seed_bag_identity_vote": seed_bag_vote,
    }.items():
        fold_change = candidate.fold_metrics.accuracy - leader.fold_metrics.accuracy
        accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        change = accuracy - leader_accuracy
        repair_change = repair - leader_repair
        rows.append(
            {
                "candidate": key,
                "mean_accuracy": accuracy,
                "leader_change": change,
                "fold_wins_vs_leader": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall_change": repair_change,
                "passes_gate": key == "seed_bag_identity_vote" and (
                    change >= 0.001
                    and int(fold_change.gt(0).sum()) >= 3
                    and float(fold_change.min()) >= -0.0025
                    and repair_change >= -0.02
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("candidate")
        .sort_values(["mean_accuracy", "candidate"], ascending=[False, True])
    )


def _predictions(evaluation):
    return evaluation.out_of_fold_probabilities.idxmax(axis=1).to_numpy()


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in (
        "mean_accuracy",
        "leader_change",
        "worst_fold_change",
        "repair_recall_change",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


if __name__ == "__main__":
    main()

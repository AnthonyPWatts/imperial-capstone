"""Replay the day's three submission recipes with minority oversampling."""

from __future__ import annotations

import argparse
import hashlib
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
BASE_RUNTIME_DIR = PROJECT_DIR / ".runtime" / "oversampled-submission-replay"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import partition_modelling_data
from final_model import CLASS_LABELS
from final_model import write_validated_submission
from modelling_data import prepare_modelling_data
from oversampling import OVERSAMPLING_SEED
from oversampling import random_oversample_minority
from oversampled_submission_replay import SUBMISSION_RECIPES
from oversampled_submission_replay import build_replay_competition_predictions
from oversampled_submission_replay import evaluate_replay_recipes
from oversampled_submission_replay import fit_replay_components
from oversampled_submission_replay import summarise_submission_csv
from oversampled_submission_replay import target_count_for_replica_fraction
from oversampled_submission_replay import target_count_for_multiplier


ORIGINAL_SUBMISSIONS = {
    "random-forest-histogram-boosting": (
        STAGE_DIR
        / "submissions"
        / "2026-08-21-forest-boosting"
        / "01-random-forest-histogram-boosting.csv"
    ),
    "55-xgboost-depth-8-child-1-current-one-hot-45-random-forest": (
        STAGE_DIR
        / "submissions"
        / "2026-08-21-model-family-screen"
        / "01-55-xgboost-depth-8-child-1-current-one-hot-45-random-forest.csv"
    ),
    "60-xgboost-depth-6-7-8-local-bag-40-random-forest": (
        STAGE_DIR
        / "submissions"
        / "2026-08-21-model-family-screen"
        / "02-60-xgboost-depth-6-7-8-local-bag-40-random-forest.csv"
    ),
}


def main() -> None:
    arguments = _parse_arguments()
    replica_fraction = arguments.replica_fraction
    target_multiplier = arguments.target_multiplier
    if target_multiplier is not None:
        suffix = f"-{round(target_multiplier * 100):03d}x"
        target_policy = f"final repair-class count is {target_multiplier:g}x natural"
    else:
        replica_fraction = 1.0 if replica_fraction is None else replica_fraction
        suffix = (
            "" if replica_fraction == 1.0
            else f"-{round(replica_fraction * 100):02d}"
        )
        target_policy = (
            "raise functional needs repair to the second-largest class"
            if replica_fraction == 1.0
            else f"retain {replica_fraction:.0%} of replicas added by the full policy"
        )
    runtime_dir = PROJECT_DIR / ".runtime" / f"oversampled-submission-replay{suffix}"
    output_dir = (
        STAGE_DIR
        / "submissions"
        / f"2026-08-21-oversampled-replay{suffix}"
    )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = partition_modelling_data(modelling_data)
    submission_template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")

    original_checkpoint = BASE_RUNTIME_DIR / "local-test-original-components.joblib"
    if original_checkpoint.is_file():
        original_local_fits = joblib.load(original_checkpoint)
        print(f"Reused unchanged original-training checkpoint: {original_checkpoint}.")
    else:
        original_local_fits = _fit_and_checkpoint(
            runtime_dir,
            "local-test-original",
            partitioned.X_development,
            partitioned.y_development,
            partitioned.X_local_test,
        )
    if target_multiplier is not None:
        development_target_count = target_count_for_multiplier(
            partitioned.y_development.value_counts(),
            target_multiplier=target_multiplier,
        )
    else:
        development_target_count = target_count_for_replica_fraction(
            partitioned.y_development.value_counts(),
            replica_fraction=replica_fraction,
        )
    oversampled_development = random_oversample_minority(
        partitioned.X_development,
        partitioned.y_development,
        partitioned.development_ids,
        target_count=development_target_count,
        seed=OVERSAMPLING_SEED,
    )
    oversampled_local_fits = _fit_and_checkpoint(
        runtime_dir,
        "local-test-oversampled",
        oversampled_development.X,
        oversampled_development.y,
        partitioned.X_local_test,
    )

    original_local = evaluate_replay_recipes(
        partitioned.y_local_test,
        original_local_fits,
    )
    oversampled_local = evaluate_replay_recipes(
        partitioned.y_local_test,
        oversampled_local_fits,
    )

    if target_multiplier is not None:
        full_target_count = target_count_for_multiplier(
            modelling_data.y_original.value_counts(),
            target_multiplier=target_multiplier,
        )
    else:
        full_target_count = target_count_for_replica_fraction(
            modelling_data.y_original.value_counts(),
            replica_fraction=replica_fraction,
        )
    oversampled_full = random_oversample_minority(
        modelling_data.X_original,
        modelling_data.y_original,
        modelling_data.original_ids,
        target_count=full_target_count,
        seed=OVERSAMPLING_SEED,
    )
    oversampled_competition_fits = _fit_and_checkpoint(
        runtime_dir,
        "competition-oversampled",
        oversampled_full.X,
        oversampled_full.y,
        modelling_data.X_competition,
    )
    competition_predictions = build_replay_competition_predictions(
        modelling_data,
        submission_template,
        oversampled_competition_fits,
    )

    submission_records = {}
    for position, (name, prediction) in enumerate(
        competition_predictions.items(),
        start=1,
    ):
        destination = output_dir / f"{position:02d}-{name}-oversampled.csv"
        write_validated_submission(prediction, destination)
        submission_records[name] = {
            "original": summarise_submission_csv(ORIGINAL_SUBMISSIONS[name]),
            "oversampled": summarise_submission_csv(destination),
            "prediction_share_change": {
                label: float(
                    prediction.class_shares[label]
                    - summarise_submission_csv(ORIGINAL_SUBMISSIONS[name])[
                        "prediction_share"
                    ][label]
                )
                for label in CLASS_LABELS
            },
        }

    results = {
        "experiment": "09-oversampled-submission-replay",
        "purpose": (
            "Change only the training class distribution for the three frozen "
            "21 August submission recipes."
        ),
        "competition_labels_available": False,
        "competition_accuracy_available": False,
        "local_test_role": "confirmatory labelled holdout; not a tuning set",
        "oversampling": {
            "method": "random replication with replacement",
            "target_policy": target_policy,
            "seed": OVERSAMPLING_SEED,
            "replica_fraction": replica_fraction,
            "target_multiplier": target_multiplier,
            "development_counts_before": _counts(
                oversampled_development.counts_before
            ),
            "development_counts_after": _counts(
                oversampled_development.counts_after
            ),
            "development_rows_before": len(partitioned.X_development),
            "development_rows_after": len(oversampled_development.X),
            "full_labelled_counts_before": _counts(oversampled_full.counts_before),
            "full_labelled_counts_after": _counts(oversampled_full.counts_after),
            "full_labelled_rows_before": len(modelling_data.X_original),
            "full_labelled_rows_after": len(oversampled_full.X),
        },
        "fingerprints": {
            "development": partitioned.development_fingerprint,
            "local_test": partitioned.local_test_fingerprint,
            "cross_validation": partitioned.cross_validation_fingerprint,
        },
        "recipes": {
            name: [
                {"component": component, "weight": weight}
                for component, weight in recipe
            ]
            for name, recipe in SUBMISSION_RECIPES.items()
        },
        "local_test": {
            name: {
                "original_training": original_local[name],
                "oversampled_training": oversampled_local[name],
                "metric_change": {
                    metric: float(
                        oversampled_local[name][metric]
                        - original_local[name][metric]
                    )
                    for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                },
                "class_recall_change": {
                    label: float(
                        oversampled_local[name]["class_recall"][label]
                        - original_local[name]["class_recall"][label]
                    )
                    for label in CLASS_LABELS
                },
            }
            for name in SUBMISSION_RECIPES
        },
        "competition_predictions": submission_records,
        "fit_seconds": {
            "local_test_original": _fit_seconds(original_local_fits),
            "local_test_oversampled": _fit_seconds(oversampled_local_fits),
            "competition_oversampled": _fit_seconds(
                oversampled_competition_fits
            ),
        },
    }
    results_path = runtime_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


def _fit_and_checkpoint(
    runtime_dir,
    name,
    X_training,
    y_training,
    X_prediction,
):
    fits = fit_replay_components(X_training, y_training, X_prediction)
    destination = runtime_dir / f"{name}-components.joblib"
    joblib.dump(fits, destination, compress=3)
    print(
        f"Checkpointed {name}: {destination.name}, "
        f"sha256={hashlib.sha256(destination.read_bytes()).hexdigest()}."
    )
    return fits


def _parse_arguments():
    parser = argparse.ArgumentParser()
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument(
        "--replica-fraction",
        type=float,
        default=None,
        help="Fraction of the default added repair-class replicas to retain.",
    )
    scaling.add_argument(
        "--target-multiplier",
        type=float,
        default=None,
        help="Final repair-class row count as a multiple of its natural count.",
    )
    arguments = parser.parse_args()
    if (
        arguments.replica_fraction is not None
        and (
            arguments.replica_fraction < 0
            or arguments.replica_fraction > 1
        )
    ):
        parser.error("--replica-fraction must fall between zero and one")
    if (
        arguments.target_multiplier is not None
        and arguments.target_multiplier < 1
    ):
        parser.error("--target-multiplier must be at least one")
    return arguments


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(label): int(value) for label, value in values.items()}


def _fit_seconds(fits) -> dict[str, float]:
    return {
        component: float(fit.elapsed_seconds)
        for component, fit in fits.items()
    }


if __name__ == "__main__":
    main()

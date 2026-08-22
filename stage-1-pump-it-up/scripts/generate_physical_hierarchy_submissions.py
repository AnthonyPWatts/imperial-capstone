"""Refit and write three exploratory physical-hierarchy submissions."""

from __future__ import annotations

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
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "physical-hierarchy-screen"
OUTPUT_DIR = STAGE_DIR / "submissions" / "2026-08-22-physical-hierarchies"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from final_model import CLASS_LABELS
from final_model import write_validated_submission
from modelling_data import prepare_modelling_data
from physical_hierarchy_submission import (
    fit_physical_hierarchy_submission_candidates,
)


def main() -> None:
    source_trial = joblib.load(RUNTIME_DIR / "source-source_plus_class.joblib")
    waterpoint_trial = joblib.load(RUNTIME_DIR / "waterpoint-both_levels.joblib")
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    submission_template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")

    print(
        "Refitting four unique full-data components for three exploratory entries.",
        flush=True,
    )
    candidates = fit_physical_hierarchy_submission_candidates(
        modelling_data,
        submission_template,
        source_trial,
        waterpoint_trial,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    labels = {}
    for position, candidate in enumerate(candidates, start=1):
        destination = OUTPUT_DIR / f"{position:02d}-{candidate.spec.key}.csv"
        write_validated_submission(candidate.prediction, destination)
        reloaded = pd.read_csv(destination)
        _validate_written_submission(
            reloaded,
            submission_template,
            expected_rows=len(modelling_data.X_competition),
        )
        labels[candidate.spec.key] = reloaded["status_group"]
        record = {
            "rank": position,
            "key": candidate.spec.key,
            "label": candidate.spec.label,
            "status": "prepared_not_uploaded",
            "exploratory": True,
            "xgboost_policy": (
                f"{candidate.spec.xgboost_policy.family}:"
                f"{candidate.spec.xgboost_policy.key}"
            ),
            "random_forest_policy": (
                f"{candidate.spec.random_forest_policy.family}:"
                f"{candidate.spec.random_forest_policy.key}"
            ),
            "weights": {"xgboost": 0.55, "random_forest": 0.45},
            "xgboost_iterations": candidate.selected_iterations,
            "competition_class_shares": {
                label: float(candidate.prediction.class_shares.loc[label])
                for label in CLASS_LABELS
            },
            "fit_and_predict_seconds": {
                key: float(value)
                for key, value in candidate.prediction.component_seconds.items()
            },
            "csv": destination.name,
            "rows": len(reloaded),
            "unique_ids": int(reloaded["id"].nunique()),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
        records.append(record)
        print(
            f"Prepared {position}/3: {destination.name} "
            f"({record['sha256'][:12]}...).",
            flush=True,
        )

    disagreement = _pairwise_disagreement(labels)
    manifest = {
        "date": "2026-08-22",
        "status": "prepared_not_uploaded",
        "local_test_opened": False,
        "selection_note": (
            "Exploratory hierarchy recipes; none passed the predeclared "
            "promotion gate."
        ),
        "entries": records,
        "competition_prediction_disagreement": disagreement,
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def _validate_written_submission(
    submission: pd.DataFrame,
    submission_template: pd.DataFrame,
    *,
    expected_rows: int,
) -> None:
    if list(submission.columns) != ["id", "status_group"]:
        raise ValueError("A written submission has invalid columns.")
    if len(submission) != expected_rows:
        raise ValueError("A written submission has the wrong row count.")
    if submission["id"].duplicated().any():
        raise ValueError("A written submission contains duplicate IDs.")
    if not submission["id"].equals(submission_template["id"]):
        raise ValueError("A written submission does not preserve template ID order.")
    if submission.isna().any().any():
        raise ValueError("A written submission contains missing values.")
    if not set(submission["status_group"]).issubset(CLASS_LABELS):
        raise ValueError("A written submission contains an invalid label.")


def _pairwise_disagreement(labels: dict[str, pd.Series]) -> dict[str, float]:
    values = {}
    keys = list(labels)
    for left_position, left in enumerate(keys):
        for right in keys[left_position + 1 :]:
            values[f"{left}__vs__{right}"] = float(
                labels[left].ne(labels[right]).mean()
            )
    return values


if __name__ == "__main__":
    main()

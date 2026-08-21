"""Refit and write the strongest distinct candidates from the model screen."""

from __future__ import annotations

import hashlib
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
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "gpu-model-screen"
OUTPUT_DIR = STAGE_DIR / "submissions" / "2026-08-21-model-family-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from final_model import write_validated_submission
from model_screen_submission import fit_selected_screen_candidates
from model_screen_submission import select_submission_candidate_names
from modelling_data import prepare_modelling_data


def main() -> None:
    screen_payload = joblib.load(RUNTIME_DIR / "combination-screen.joblib")
    trial_payloads = _load_trial_payloads()
    selected_names = select_submission_candidate_names(screen_payload)
    if not selected_names:
        print("No model-screen candidate passed the selection gate.")
        return

    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    submission_template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")
    candidates = fit_selected_screen_candidates(
        modelling_data,
        submission_template,
        screen_payload,
        trial_payloads,
        selected_names,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for position, candidate in enumerate(candidates, start=1):
        destination = OUTPUT_DIR / f"{position:02d}-{_slug(candidate.name)}.csv"
        write_validated_submission(candidate.prediction, destination)
        records.append(
            {
                "rank": position,
                "candidate": candidate.name,
                "recipe": [
                    {"component": component, "weight": weight}
                    for component, weight in candidate.recipe
                ],
                "oof": screen_payload["summary"].loc[
                    candidate.name
                ].to_dict(),
                "competition_class_shares": (
                    candidate.prediction.class_shares.to_dict()
                ),
                "fit_and_predict_seconds": (
                    candidate.prediction.component_seconds.to_dict()
                ),
                "selected_iterations": (
                    candidate.selected_iterations.to_dict()
                ),
                "csv": str(destination),
                "rows": len(candidate.prediction.submission),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )

    report_path = RUNTIME_DIR / "selected-submissions.json"
    report_path.write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


def _load_trial_payloads() -> dict[str, dict]:
    payloads = {}
    for path in sorted(RUNTIME_DIR.glob("*.joblib")):
        if path.name == "combination-screen.joblib":
            continue
        payload = joblib.load(path)
        payloads[payload["evaluation"].model_name] = payload
    return payloads


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


if __name__ == "__main__":
    main()

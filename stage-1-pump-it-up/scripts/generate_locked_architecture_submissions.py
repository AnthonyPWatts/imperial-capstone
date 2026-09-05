"""Generate gated architecture hypotheses from the locked deep archive."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from typing import Callable

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "locked-architecture-2026-09-05"
OUTPUT_DIR = (
    STAGE_DIR / "submissions" / "2026-09-05-locked-architecture-substitutions"
)
SPATIAL_DIR = PROJECT_DIR / ".runtime" / "fresh-spatial-grid-catboost"
RF_DIR = PROJECT_DIR / ".runtime" / "fresh-rf-family-screen-seed-20260905"
INCUMBENT_CSV = (
    STAGE_DIR
    / "submissions"
    / "2026-08-23-deep-archive-seed-20260824"
    / "01-deep-archive-seed-20260824.csv"
)
DATA_HASHES = {
    "TrainingSetValues.csv": "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d",
    "TrainingSetLabels.csv": "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24",
    "TestSetValues.csv": "a222110d5606910953607efa5112eafb1d6c30a483c4cbcd0b92c8306125c9b5",
    "SubmissionFormat.csv": "387b15f692da6196b1f2610be1aa327a8cbe8dffa15454b25882f41a4077b9f6",
}
INCUMBENT_INPUTS = {
    "deep": (
        PROJECT_DIR
        / ".runtime/deep-archive-seed-20260824-competition/deep-xgboost.joblib",
        "71a08cfaa0992bc3940abadae7d844d79f682ff1fd7fed8b894822f926cea835",
    ),
    "accepted": (
        PROJECT_DIR
        / ".runtime/class-membership-analysis/accepted-competition-probabilities.joblib",
        "c6bd9a710b2c3e055f07c6d541a5cb026674974e26dd65afcd45884247e48a2e",
    ),
    "identity": (
        PROJECT_DIR
        / ".runtime/catboost-identity-competition/complete-identity-catboost.joblib",
        "1a0ae110ac7e2f758474c6a2ff0dcb11f4fef0ca9a236df7ba4af4466d02b4e8",
    ),
    "archive": (
        PROJECT_DIR
        / ".runtime/archive-synthesis-competition/archive-components.joblib",
        "3a4226f03ee2fa2b8ad2da7132fa25f9ecf81d4e1a1760f0f8e644d0b49c4046",
    ),
}
INCUMBENT_CSV_HASH = (
    "fe5de9ea46bad2b35226bc97ebdfb743807fb758df1d259e8f8a51609808fee2"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from deep_archive_confirmation import blend_deep_archive
from final_model import CLASS_LABELS
from final_model import build_competition_prediction
from final_model import write_validated_submission
from fresh_rf_family_evaluation import LOCKED_RF_VARIANTS
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from locked_architecture_candidates import COMBINED_SUBSTITUTION
from locked_architecture_candidates import RF_SUBSTITUTION
from locked_architecture_candidates import SPATIAL_SUBSTITUTION
from locked_architecture_candidates import build_locked_substitution_probabilities
from locked_architecture_candidates import decide_locked_replacements
from locked_architecture_candidates import sha256_file
from locked_architecture_candidates import validate_competition_ids
from locked_architecture_candidates import validate_replacement_cache
from locked_architecture_candidates import write_json_without_overwrite
from locked_screen_evidence import load_rf_screen_evidence
from locked_screen_evidence import load_spatial_screen_evidence
from modelling_data import ModellingData
from modelling_data import prepare_modelling_data
from spatial_grid_catboost_evaluation import (
    engineer_spatial_grid_catboost_features,
)


def main() -> None:
    modelling, template = _load_data()
    spatial = load_spatial_screen_evidence(SPATIAL_DIR, modelling)
    rf = load_rf_screen_evidence(RF_DIR, modelling)
    decision = decide_locked_replacements(
        spatial.result,
        rf.result,
        allowed_rf_variants=LOCKED_RF_VARIANTS,
    )
    incumbent, seconds, incumbent_hashes = _load_incumbent(modelling, template)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spatial_cache = None
    if decision.use_spatial_grid_catboost:
        spatial_cache = _replacement_component(
            RUNTIME_DIR / "spatial-grid-catboost.joblib",
            modelling,
            candidate="spatial_grid_catboost",
            spec=spatial.spec,
            iterations=spatial.iterations,
            evidence_hashes=spatial.hashes,
            fitter=lambda: fit_gpu_candidate_probabilities(
                spatial.spec,
                modelling.X_original,
                modelling.y_original,
                modelling.X_competition,
                iterations=spatial.iterations,
                catboost_feature_engineer=(
                    engineer_spatial_grid_catboost_features
                ),
            ),
        )
    rf_cache = None
    if decision.rf_variant is not None:
        rf_cache = _replacement_component(
            RUNTIME_DIR / f"rf-{_slug(decision.rf_variant)}.joblib",
            modelling,
            candidate=f"rf_family:{decision.rf_variant}",
            spec=rf.spec,
            iterations=rf.iterations,
            evidence_hashes=rf.hashes,
            fitter=lambda: fit_gpu_candidate_probabilities(
                rf.spec,
                modelling.X_original,
                modelling.y_original,
                modelling.X_competition,
                iterations=rf.iterations,
            ),
        )

    candidates = build_locked_substitution_probabilities(
        incumbent,
        decision,
        spatial_grid_catboost=_probabilities(spatial_cache),
        rf_family=_probabilities(rf_cache),
    )
    records = _write_candidates(
        candidates,
        modelling,
        template,
        pd.read_csv(INCUMBENT_CSV),
        seconds,
        decision.rf_variant,
    )
    manifest = {
        "date": "2026-09-05",
        "status": "unconfirmed_architecture_hypotheses",
        "auto_promoted": False,
        "review_required_before_upload": True,
        "evidence_limitation": (
            "Fresh gates cover standalone replacement components, not their "
            "fixed-weight deep-archive substitutions; aligned fresh OOF "
            "probabilities for every incumbent component do not exist."
        ),
        "protocol": {
            "training_rows": len(modelling.y_original),
            "historical_local_subset_used": False,
            "post_hoc_row_changes": False,
            "weights": DEEP_ARCHIVE_WEIGHTS,
        },
        "screen_decision": asdict(decision),
        "source_sha256": DATA_HASHES,
        "incumbent_sha256": incumbent_hashes,
        "screen_sha256": {
            "spatial": spatial.hashes,
            "rf": rf.hashes,
        },
        "replacement_caches": {
            key: _cache_manifest(value, path)
            for key, value, path in (
                (
                    "spatial",
                    spatial_cache,
                    RUNTIME_DIR / "spatial-grid-catboost.joblib",
                ),
                (
                    "rf",
                    rf_cache,
                    RUNTIME_DIR / f"rf-{_slug(decision.rf_variant or '')}.joblib",
                ),
            )
            if value is not None
        },
        "candidates": records,
    }
    write_json_without_overwrite(OUTPUT_DIR / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


def _load_data() -> tuple[ModellingData, pd.DataFrame]:
    for name, expected in DATA_HASHES.items():
        _require_hash(DATA_DIR / name, expected)
    modelling = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")
    if list(template.columns) != ["id", "status_group"]:
        raise ValueError("Submission template schema changed.")
    if not template["id"].reset_index(drop=True).equals(
        modelling.competition_ids.reset_index(drop=True)
    ):
        raise ValueError("Submission template IDs changed or are misordered.")
    if set(modelling.y_original) != set(CLASS_LABELS):
        raise ValueError("Training labels changed.")
    return modelling, template


def _load_incumbent(
    modelling: ModellingData,
    template: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.Series, dict[str, str]]:
    loaded = {}
    hashes = {}
    for name, (path, expected_hash) in INCUMBENT_INPUTS.items():
        _require_hash(path, expected_hash)
        loaded[name] = joblib.load(path)
        validate_competition_ids(
            loaded[name],
            modelling.competition_ids,
            name=name,
        )
        hashes[name] = expected_hash
    _require_hash(INCUMBENT_CSV, INCUMBENT_CSV_HASH)
    hashes["submission"] = INCUMBENT_CSV_HASH
    if loaded["deep"].get("seed") != 20260824:
        raise ValueError("Incumbent deep-XGBoost seed changed.")
    if loaded["deep"].get("iterations") != 600:
        raise ValueError("Incumbent deep-XGBoost iterations changed.")
    if loaded["identity"].get("iterations") != 2173:
        raise ValueError("Incumbent identity-CatBoost iterations changed.")
    components = {
        "deep_xgboost": loaded["deep"]["probabilities"],
        "spatial_xgboost": loaded["archive"]["spatial_xgboost"],
        "random_forest": loaded["accepted"]["random_forest_probabilities"],
        "frequency_random_forest": loaded["archive"][
            "frequency_random_forest"
        ],
        "spatial_random_forest": loaded["archive"]["spatial_random_forest"],
        "identity_catboost": loaded["identity"]["probabilities"],
    }
    probabilities = blend_deep_archive(components)
    prediction = build_competition_prediction(
        modelling,
        template,
        probabilities,
        pd.Series(dtype="float64", name="fit and predict seconds"),
    )
    if not pd.read_csv(INCUMBENT_CSV).equals(
        prediction.submission.reset_index(drop=True)
    ):
        raise ValueError("Locked components do not reconstruct the incumbent.")
    seconds = pd.Series(
        {
            "deep_xgboost": loaded["deep"]["fit_and_predict_seconds"],
            "random_forest": loaded["accepted"]["component_seconds"].loc[
                "Random Forest"
            ],
            "identity_catboost": loaded["identity"][
                "fit_and_predict_seconds"
            ],
        },
        name="fit and predict seconds",
        dtype="float64",
    )
    return components, seconds, hashes


def _replacement_component(
    path: Path,
    modelling: ModellingData,
    *,
    candidate: str,
    spec: object,
    iterations: int,
    evidence_hashes: dict[str, str],
    fitter: Callable[[], tuple[np.ndarray, float]],
) -> dict[str, object]:
    metadata = {
        "version": 1,
        "candidate": candidate,
        "spec": asdict(spec),
        "iterations": iterations,
        "source_sha256": DATA_HASHES,
        "screen_sha256": evidence_hashes,
    }
    if path.exists():
        payload = joblib.load(path)
    else:
        probabilities, elapsed = fitter()
        payload = {
            "metadata": metadata,
            "competition_ids": modelling.competition_ids.copy(),
            "probabilities": probabilities,
            "fit_and_predict_seconds": elapsed,
        }
        joblib.dump(payload, path, compress=3)
    return validate_replacement_cache(
        payload,
        metadata,
        modelling.competition_ids,
        expected_rows=len(modelling.X_competition),
        name=candidate,
    )


def _write_candidates(
    candidates: dict[str, np.ndarray],
    modelling: ModellingData,
    template: pd.DataFrame,
    incumbent: pd.DataFrame,
    seconds: pd.Series,
    rf_variant: str | None,
) -> list[dict[str, object]]:
    records = []
    for number, (name, probabilities) in enumerate(candidates.items(), start=1):
        prediction = build_competition_prediction(
            modelling,
            template,
            probabilities,
            seconds,
        )
        path = OUTPUT_DIR / f"{number:02d}-{name.replace('_', '-')}.csv"
        write_validated_submission(prediction, path)
        written = pd.read_csv(path)
        if not written.equals(prediction.submission.reset_index(drop=True)):
            raise ValueError(f"Written submission changed: {path}")
        substitutions = []
        if name in {SPATIAL_SUBSTITUTION, COMBINED_SUBSTITUTION}:
            substitutions.append(
                {"component": "identity_catboost", "weight": 0.20}
            )
        if name in {RF_SUBSTITUTION, COMBINED_SUBSTITUTION}:
            substitutions.append(
                {
                    "component": "random_forest",
                    "replacement": rf_variant,
                    "weight": 0.18,
                }
            )
        records.append(
            {
                "candidate": name,
                "substitutions": substitutions,
                "csv": path.name,
                "sha256": sha256_file(path),
                "rows": len(written),
                "unique_ids": int(written["id"].nunique()),
                "disagreement_vs_incumbent": float(
                    written["status_group"].ne(incumbent["status_group"]).mean()
                ),
                "class_shares": {
                    label: float(prediction.class_shares.loc[label])
                    for label in CLASS_LABELS
                },
                "post_hoc_rows_changed": 0,
            }
        )
    return records


def _cache_manifest(payload: dict[str, object], path: Path) -> dict[str, object]:
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "metadata": payload["metadata"],
        "fit_and_predict_seconds": float(payload["fit_and_predict_seconds"]),
    }


def _probabilities(payload: dict[str, object] | None) -> np.ndarray | None:
    return None if payload is None else payload["probabilities"]


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 changed for {path}: {actual}.")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


if __name__ == "__main__":
    main()

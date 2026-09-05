"""Generate the optional deep-XGBoost two-seed variance-hedge submission."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "deep-xgboost-variance-hedge-competition"
OUTPUT_DIR = (
    STAGE_DIR
    / "submissions"
    / "2026-09-05-optional-deep-xgboost-variance-hedge"
)
CANDIDATE_NAME = (
    "01-optional-stable-positive-near-miss-deep-xgboost-"
    "seed-20260824-20260905-average.csv"
)
MANIFEST_NAME = "manifest.json"
SEED_20260905_CACHE = RUNTIME_DIR / "deep-xgboost-seed-20260905.joblib"
EVIDENCE_RESULT = (
    PROJECT_DIR
    / ".runtime/deep-xgboost-two-seed-variance-check/result.json"
)
EVIDENCE_CACHE = (
    PROJECT_DIR
    / ".runtime/deep-xgboost-two-seed-variance-check/"
    "seed-average-evaluation.joblib"
)
INCUMBENT_CSV = (
    STAGE_DIR
    / "submissions/2026-08-23-deep-archive-seed-20260824/"
    "01-deep-archive-seed-20260824.csv"
)

DATA_SHA256 = {
    "TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    "TrainingSetLabels.csv": (
        "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
    ),
    "TestSetValues.csv": (
        "a222110d5606910953607efa5112eafb1d6c30a483c4cbcd0b92c8306125c9b5"
    ),
    "SubmissionFormat.csv": (
        "387b15f692da6196b1f2610be1aa327a8cbe8dffa15454b25882f41a4077b9f6"
    ),
}
SOURCE_SHA256 = {
    "src/gpu_model_evaluation.py": (
        "e6867cac83f1acd5437a34288748578662cb106d219ac00c3cb2f76c7fd884b5"
    ),
    "src/top_common_identity_evaluation.py": (
        "cc2719cbebd9e776c3e410a10b5eaa017e19f1fb702d4303c2498fdc48904c85"
    ),
    "src/feature_engineering.py": (
        "b900fd38c74ce1dee840428752aa3c342d1eea8a124804d5fd1f001a405b4026"
    ),
    "src/model_preprocessing.py": (
        "8333221a2e9b889db99699707aacf1eab973c2b3e3ed3cea57ab777170af8762"
    ),
    "src/data_preparation.py": (
        "2aeda2f9229275ddf9bb207422d4ae4aacf3c05c46e16321147d1fa5c6e496f6"
    ),
    "src/modelling_data.py": (
        "8651ee63c740fedf16ae55a1882eb4dd577fc93388f8452f6c2a3381e689e878"
    ),
    "src/deep_archive_confirmation.py": (
        "aa156cd3b69c8be2d86608244355ed6c99b9d1cf4d0819417160a35bbd636b3d"
    ),
    "src/final_model.py": (
        "798e2505706f150d2c9d535e6bdb906732a44c1442ade35356b9a869d52186b8"
    ),
    "src/deep_xgboost_seed_average.py": (
        "4ad3a1c31f17305e2d2e6d189a5ca7abfa92926c4d96c86c19ac56e05772ea3b"
    ),
    "scripts/run_deep_xgboost_seed_average.py": (
        "354f8f721f3e816e81357ac1a8c6ebb6ed41f1dd0cf7d072397fcbeacbee3884"
    ),
}
PINNED_CACHE_INPUTS = {
    "two_seed_evidence_result": (
        EVIDENCE_RESULT,
        "e71917a2a46c3630ff7b4398c38dcd4a1dc80783dd931334fa3e658e7e5bbe9d",
    ),
    "two_seed_average_evaluation": (
        EVIDENCE_CACHE,
        "e8b3403e9e9d4ec7b587fb27c7ba966b9fdd463f6bab262febd6e46aea0bd1ae",
    ),
    "incumbent_deep_xgboost": (
        PROJECT_DIR
        / ".runtime/deep-archive-seed-20260824-competition/deep-xgboost.joblib",
        "71a08cfaa0992bc3940abadae7d844d79f682ff1fd7fed8b894822f926cea835",
    ),
    "incumbent_accepted_components": (
        PROJECT_DIR
        / ".runtime/class-membership-analysis/"
        "accepted-competition-probabilities.joblib",
        "c6bd9a710b2c3e055f07c6d541a5cb026674974e26dd65afcd45884247e48a2e",
    ),
    "incumbent_identity_catboost": (
        PROJECT_DIR
        / ".runtime/catboost-identity-competition/"
        "complete-identity-catboost.joblib",
        "1a0ae110ac7e2f758474c6a2ff0dcb11f4fef0ca9a236df7ba4af4466d02b4e8",
    ),
    "incumbent_archive_components": (
        PROJECT_DIR
        / ".runtime/archive-synthesis-competition/archive-components.joblib",
        "3a4226f03ee2fa2b8ad2da7132fa25f9ecf81d4e1a1760f0f8e644d0b49c4046",
    ),
    "incumbent_submission": (
        INCUMBENT_CSV,
        "fe5de9ea46bad2b35226bc97ebdfb743807fb758df1d259e8f8a51609808fee2",
    ),
}

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deep_xgboost_variance_hedge import build_variance_hedge_probabilities
from deep_xgboost_variance_hedge import canonical_deep_xgboost_recipe
from deep_xgboost_variance_hedge import canonical_ensemble_recipe
from deep_xgboost_variance_hedge import canonical_probability_average_recipe
from deep_xgboost_variance_hedge import make_deep_component_cache_metadata
from deep_xgboost_variance_hedge import OPTIONAL_STATUS
from deep_xgboost_variance_hedge import OPT_IN_TOKEN
from deep_xgboost_variance_hedge import require_explicit_opt_in
from deep_xgboost_variance_hedge import require_sha256
from deep_xgboost_variance_hedge import SEED_20260824
from deep_xgboost_variance_hedge import SEED_20260905
from deep_xgboost_variance_hedge import sha256_file
from deep_xgboost_variance_hedge import validate_component_ids
from deep_xgboost_variance_hedge import validate_legacy_deep_component_cache
from deep_xgboost_variance_hedge import validate_near_miss_evidence
from deep_xgboost_variance_hedge import validate_submission_frame
from deep_xgboost_variance_hedge import validate_versioned_deep_component_cache
from deep_xgboost_variance_hedge import write_json_without_overwrite
from deep_xgboost_variance_hedge import write_submission_without_overwrite
from final_model import build_competition_prediction
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from modelling_data import ModellingData
from modelling_data import prepare_modelling_data
from top_common_identity_evaluation import ARCHIVED_DEPTH_17_ITERATIONS
from top_common_identity_evaluation import make_top_common_identity_preprocessor


def main(*, opt_in: str | None) -> None:
    """Fit the missing seed component and write the optional candidate once."""

    require_explicit_opt_in(opt_in)
    destination = OUTPUT_DIR / CANDIDATE_NAME
    manifest_path = OUTPUT_DIR / MANIFEST_NAME
    _require_new_output_paths(destination, manifest_path)
    pinned_hashes = _validate_pinned_files()
    evidence = _read_json(EVIDENCE_RESULT)
    validate_near_miss_evidence(evidence)
    modelling, template = _load_data()
    incumbent, component_seconds = _load_incumbent(modelling, template)
    cache_metadata = make_deep_component_cache_metadata(
        seed=SEED_20260905,
        training_rows=len(modelling.y_original),
        competition_rows=len(modelling.X_competition),
        data_sha256=DATA_SHA256,
        source_sha256=SOURCE_SHA256,
        evidence_sha256={
            "result.json": PINNED_CACHE_INPUTS[
                "two_seed_evidence_result"
            ][1],
            "seed-average-evaluation.joblib": PINNED_CACHE_INPUTS[
                "two_seed_average_evaluation"
            ][1],
        },
    )
    seed_20260905 = _load_or_fit_seed_20260905(
        modelling,
        cache_metadata,
    )
    hedge = build_variance_hedge_probabilities(
        incumbent,
        incumbent["deep_xgboost"],
        seed_20260905["probabilities"],
    )
    component_seconds.loc["deep_xgboost_seed_20260905"] = float(
        seed_20260905["fit_and_predict_seconds"]
    )
    prediction = build_competition_prediction(
        modelling,
        template,
        hedge.candidate_ensemble,
        component_seconds,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_submission_without_overwrite(
        prediction.submission,
        modelling.competition_ids,
        destination,
    )
    written = pd.read_csv(destination)
    validate_submission_frame(written, modelling.competition_ids)
    incumbent_submission = pd.read_csv(INCUMBENT_CSV)
    manifest = {
        "date": "2026-09-05",
        "status": OPTIONAL_STATUS,
        "auto_promoted": False,
        "review_required_before_upload": True,
        "promotion_decision": "do_not_promote",
        "evidence_verdict": "stable_positive_near_miss",
        "explicit_cli_opt_in": opt_in,
        "caveat": (
            "This optional variance hedge did not pass the locked +0.10pp "
            "two-baseline promotion gate; generation does not promote it."
        ),
        "protocol": {
            "training_rows": len(modelling.y_original),
            "competition_rows": len(modelling.X_competition),
            "historical_local_subset_used": False,
            "post_hoc_row_changes": 0,
            "only_replaced_component": "deep_xgboost",
            "deep_component_weight": 0.33,
        },
        "recipes": {
            "seed_20260824": canonical_deep_xgboost_recipe(SEED_20260824),
            "seed_20260905": canonical_deep_xgboost_recipe(SEED_20260905),
            "probability_average": canonical_probability_average_recipe(),
            "ensemble": canonical_ensemble_recipe(),
        },
        "pinned_data_sha256": DATA_SHA256,
        "current_replay_and_training_source_sha256": SOURCE_SHA256,
        "historical_evidence_producer_sha256": evidence["source_sha256"],
        "pinned_input_sha256": pinned_hashes,
        "generated_component_cache": {
            "path": str(SEED_20260905_CACHE.relative_to(PROJECT_DIR)),
            "sha256": sha256_file(SEED_20260905_CACHE),
            "metadata": cache_metadata,
            "fit_and_predict_seconds": float(
                seed_20260905["fit_and_predict_seconds"]
            ),
        },
        "candidate": {
            "csv": destination.name,
            "sha256": sha256_file(destination),
            "rows": len(written),
            "unique_ids": int(written["id"].nunique()),
            "competition_prediction_disagreement_vs_incumbent": float(
                written["status_group"]
                .ne(incumbent_submission["status_group"])
                .mean()
            ),
            "competition_class_shares": {
                label: float(prediction.class_shares.loc[label])
                for label in CLASS_LABELS
            },
        },
    }
    write_json_without_overwrite(manifest, manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


def _validate_pinned_files() -> dict[str, str]:
    hashes = {}
    for name, expected in DATA_SHA256.items():
        hashes[f"data/{name}"] = require_sha256(DATA_DIR / name, expected)
    for relative, expected in SOURCE_SHA256.items():
        hashes[relative] = require_sha256(STAGE_DIR / relative, expected)
    for name, (path, expected) in PINNED_CACHE_INPUTS.items():
        hashes[name] = require_sha256(path, expected)
    return hashes


def _load_data() -> tuple[ModellingData, pd.DataFrame]:
    modelling = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")
    if list(template.columns) != ["id", "status_group"]:
        raise ValueError("SubmissionFormat.csv columns changed.")
    if not template["id"].reset_index(drop=True).equals(
        modelling.competition_ids.reset_index(drop=True)
    ):
        raise ValueError("SubmissionFormat.csv IDs changed or are misordered.")
    if set(modelling.y_original) != set(CLASS_LABELS):
        raise ValueError("Full-training target classes changed.")
    return modelling, template


def _load_incumbent(
    modelling: ModellingData,
    template: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], pd.Series]:
    deep = joblib.load(PINNED_CACHE_INPUTS["incumbent_deep_xgboost"][0])
    accepted = joblib.load(
        PINNED_CACHE_INPUTS["incumbent_accepted_components"][0]
    )
    identity = joblib.load(
        PINNED_CACHE_INPUTS["incumbent_identity_catboost"][0]
    )
    archive = joblib.load(
        PINNED_CACHE_INPUTS["incumbent_archive_components"][0]
    )
    validate_legacy_deep_component_cache(
        deep,
        modelling.competition_ids,
        expected_seed=SEED_20260824,
    )
    _validate_incumbent_cache_schemas(accepted, identity, archive)
    for name, payload in (
        ("accepted incumbent", accepted),
        ("identity incumbent", identity),
        ("archive incumbent", archive),
    ):
        validate_component_ids(payload, modelling.competition_ids, name=name)

    components = {
        "deep_xgboost": deep["probabilities"],
        "spatial_xgboost": archive["spatial_xgboost"],
        "random_forest": accepted["random_forest_probabilities"],
        "frequency_random_forest": archive["frequency_random_forest"],
        "spatial_random_forest": archive["spatial_random_forest"],
        "identity_catboost": identity["probabilities"],
    }
    for probabilities in components.values():
        validate_probabilities(probabilities, len(modelling.X_competition))
    incumbent = build_variance_hedge_probabilities(
        components,
        components["deep_xgboost"],
        components["deep_xgboost"],
    ).incumbent_ensemble
    prediction = build_competition_prediction(
        modelling,
        template,
        incumbent,
        pd.Series(dtype="float64", name="fit and predict seconds"),
    )
    incumbent_submission = pd.read_csv(INCUMBENT_CSV)
    validate_submission_frame(incumbent_submission, modelling.competition_ids)
    if not incumbent_submission.equals(prediction.submission.reset_index(drop=True)):
        raise ValueError("Pinned components do not reconstruct the incumbent.")
    seconds = pd.Series(
        {
            "deep_xgboost_seed_20260824": deep["fit_and_predict_seconds"],
            "spatial_xgboost": archive["component_seconds"].loc[
                "spatial-height XGBoost"
            ],
            "random_forest": accepted["component_seconds"].loc["Random Forest"],
            "frequency_random_forest": archive["component_seconds"].loc[
                "frequency Random Forest"
            ],
            "spatial_random_forest": archive["component_seconds"].loc[
                "spatial-height Random Forest"
            ],
            "identity_catboost": identity["fit_and_predict_seconds"],
        },
        name="fit and predict seconds",
        dtype="float64",
    )
    if not np.isfinite(seconds).all() or seconds.le(0).any():
        raise ValueError("Pinned incumbent component timings are invalid.")
    return components, seconds


def _validate_incumbent_cache_schemas(accepted, identity, archive) -> None:
    expected_keys = {
        "accepted": {
            "competition_ids",
            "xgboost_probabilities",
            "random_forest_probabilities",
            "blend_probabilities",
            "xgboost_iterations",
            "component_seconds",
        },
        "identity": {
            "competition_ids",
            "iterations",
            "probabilities",
            "fit_and_predict_seconds",
        },
        "archive": {
            "competition_ids",
            "spatial_xgboost",
            "spatial_random_forest",
            "frequency_random_forest",
            "spatial_xgboost_iterations",
            "component_seconds",
        },
    }
    for name, payload in (
        ("accepted", accepted),
        ("identity", identity),
        ("archive", archive),
    ):
        if not isinstance(payload, dict) or set(payload) != expected_keys[name]:
            raise ValueError(f"Pinned {name} cache schema changed.")
    if identity["iterations"] != 2173:
        raise ValueError("Pinned identity-CatBoost iterations changed.")
    if accepted["xgboost_iterations"] != 1014:
        raise ValueError("Pinned accepted XGBoost iterations changed.")
    if archive["spatial_xgboost_iterations"] != 1082:
        raise ValueError("Pinned spatial XGBoost iterations changed.")


def _load_or_fit_seed_20260905(
    modelling: ModellingData,
    metadata: dict[str, object],
) -> dict[str, object]:
    if SEED_20260905_CACHE.exists():
        return validate_versioned_deep_component_cache(
            joblib.load(SEED_20260905_CACHE),
            metadata,
            modelling.competition_ids,
        )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    spec = replace(
        make_xgboost_spec(
            variant="archived depth 17",
            seed=SEED_20260905,
        ),
        name=(
            "XGBoost archived depth 17 "
            "[raw top-50 deferred identities; seed 20260905]"
        ),
        feature_policy="accepted_plus_raw_top_50_deferred_identities",
    )
    probabilities, seconds = fit_gpu_candidate_probabilities(
        spec,
        modelling.X_original,
        modelling.y_original,
        modelling.X_competition,
        iterations=ARCHIVED_DEPTH_17_ITERATIONS,
        preprocessor_factory=make_top_common_identity_preprocessor,
    )
    validate_probabilities(probabilities, len(modelling.X_competition))
    payload = {
        "cache_version": 1,
        "metadata": metadata,
        "competition_ids": modelling.competition_ids.copy(),
        "probabilities": probabilities,
        "fit_and_predict_seconds": seconds,
    }
    if SEED_20260905_CACHE.exists():
        raise FileExistsError(
            f"Refusing to overwrite component cache: {SEED_20260905_CACHE}."
        )
    joblib.dump(payload, SEED_20260905_CACHE, compress=3)
    return validate_versioned_deep_component_cache(
        joblib.load(SEED_20260905_CACHE),
        metadata,
        modelling.competition_ids,
    )


def _require_new_output_paths(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing variance-hedge output(s): "
            + ", ".join(existing)
        )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}.")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an optional variance hedge that remains explicitly "
            "classified do_not_promote."
        )
    )
    parser.add_argument(
        "--opt-in",
        required=True,
        choices=(OPT_IN_TOKEN,),
        help="Deliberate acknowledgement of the locked near-miss verdict.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(opt_in=_parse_args().opt_in)

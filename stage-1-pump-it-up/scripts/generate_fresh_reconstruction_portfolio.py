"""Generate the exact three-entry fresh-reconstruction submission slate.

This script is intentionally inert without its exact ``--opt-in`` token.  It
validates the final all-label OOF evidence and source hashes before opening the
supplied competition covariates or fitting the two missing full-data models.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
SRC_DIR = STAGE_DIR / "src"
DATA_DIR = STAGE_DIR / "data"
EVIDENCE_DIR = (
    PROJECT_DIR / ".runtime/fresh-deep-archive-reconstruction-seed-20260905"
)
SPATIAL_DIR = PROJECT_DIR / ".runtime/fresh-spatial-grid-catboost"
RF_DIR = PROJECT_DIR / ".runtime/fresh-rf-family-screen-seed-20260905"
RUNTIME_DIR = PROJECT_DIR / ".runtime/fresh-reconstruction-portfolio-competition"
OUTPUT_DIR = STAGE_DIR / "submissions/2026-09-05-fresh-reconstruction-portfolio"
EVIDENCE_RESULT = EVIDENCE_DIR / "result.json"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

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
# Filled only after the generator and evidence producer are final.  The runner
# fails closed if any of these implementation paths subsequently changes.
PINNED_SOURCE_SHA256 = {
    "src/catboost_identity_evaluation.py": (
        "b078bb5132cf994c4078aa50d16108841903a0715cd1061d19a57272432e2e9f"
    ),
    "src/data_partitioning.py": (
        "ccf43b4bca8cd542ec100783f6aa7bbb23dc15946ec29880604bed97d1cd3de3"
    ),
    "src/data_preparation.py": (
        "2aeda2f9229275ddf9bb207422d4ae4aacf3c05c46e16321147d1fa5c6e496f6"
    ),
    "src/deep_archive_confirmation.py": (
        "aa156cd3b69c8be2d86608244355ed6c99b9d1cf4d0819417160a35bbd636b3d"
    ),
    "src/feature_engineering.py": (
        "b900fd38c74ce1dee840428752aa3c342d1eea8a124804d5fd1f001a405b4026"
    ),
    "src/final_model.py": (
        "798e2505706f150d2c9d535e6bdb906732a44c1442ade35356b9a869d52186b8"
    ),
    "src/fresh_deep_archive_reconstruction.py": (
        "9f74a3c71cf6a78785e20631e0603e8ab0f8188cbbbc9cf5e8fc696b4390287b"
    ),
    "src/fresh_rf_family_evaluation.py": (
        "856d6b7fb969b06a2e2c044cd46a0ed6b2da89daae76fff9efe7e8f1368d616b"
    ),
    "src/fresh_submission_portfolio.py": (
        "148e12b2ec78d588d6c9b16faf75f446c810081e90ab2d61074e755ac9d4c1b6"
    ),
    "src/gpu_model_evaluation.py": (
        "e6867cac83f1acd5437a34288748578662cb106d219ac00c3cb2f76c7fd884b5"
    ),
    "src/locked_architecture_candidates.py": (
        "376ce158634709b87a0f787b0526ebbe5f99cdcf3b6a6a4f2d530522c028d291"
    ),
    "src/locked_screen_evidence.py": (
        "50b57a22c9ccf533c4b580fcc01847fe3a750978e530430b0d2d3cd7c9ace55e"
    ),
    "src/model_evaluation.py": (
        "2c8d668654bfb9a6bbe04b17e1fec1f9d9ce16af647d891b7807d96c77988c20"
    ),
    "src/model_preprocessing.py": (
        "8333221a2e9b889db99699707aacf1eab973c2b3e3ed3cea57ab777170af8762"
    ),
    "src/modelling_data.py": (
        "8651ee63c740fedf16ae55a1882eb4dd577fc93388f8452f6c2a3381e689e878"
    ),
    "src/source_data_validation.py": (
        "fdb373a25ff888a98862f3c6bd460b0299bf21e1691b1e7e1f8c5f0f7c2ba3c7"
    ),
    "src/spatial_grid_catboost_evaluation.py": (
        "84b53ac5d3505e4c1d1ca9e4ea2e1ab94321c1c01331ff00c87c153dfb1739c3"
    ),
    "src/target_encoding_features.py": (
        "bce953f40d3dd8093eacb47deef14dc25460428f3d643f98fa92cf07b0f79fd6"
    ),
    "scripts/generate_locked_architecture_submissions.py": (
        "6bc7aa433c03b46fd24c96c45dce75a6575b247ab8d6665cd2c6d9deaa627fa3"
    ),
    "scripts/run_fresh_deep_archive_reconstruction.py": (
        "ffe49d5ea5fccfe84b113c2dd4244e2262c7ab64caf32d5ad431e46dd3652316"
    ),
}
PINNED_EVIDENCE_RESULT_SHA256 = (
    "0e76b5b9180ad43d4bd95a75aeb8e87e8051700c7045f72573c2b31bbb086261"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from final_model import build_competition_prediction
from fresh_deep_archive_reconstruction import CATBOOST_IDENTITY_GRID_BAG
from fresh_deep_archive_reconstruction import COMBINED_FALLBACK_CANDIDATE
from fresh_deep_archive_reconstruction import RF_CURRENT_LEAF_1_BAG
from fresh_rf_family_evaluation import make_locked_rf_specs
from fresh_submission_portfolio import analyse_portfolio_label_distinctness
from fresh_submission_portfolio import build_fresh_portfolio_probabilities
from fresh_submission_portfolio import make_component_cache_metadata
from fresh_submission_portfolio import make_component_cache_payload
from fresh_submission_portfolio import OPT_IN_TOKEN
from fresh_submission_portfolio import PORTFOLIO_CANDIDATES
from fresh_submission_portfolio import PORTFOLIO_FILENAMES
from fresh_submission_portfolio import PORTFOLIO_STATUS
from fresh_submission_portfolio import require_explicit_opt_in
from fresh_submission_portfolio import require_materially_distinct_portfolio
from fresh_submission_portfolio import sha256_file
from fresh_submission_portfolio import validate_component_cache
from fresh_submission_portfolio import validate_fresh_portfolio_evidence
from fresh_submission_portfolio import write_json_without_overwrite
from fresh_submission_portfolio import write_submission_without_overwrite
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import resolve_sklearn_tree_parameters
from locked_screen_evidence import load_rf_screen_evidence
from locked_screen_evidence import load_spatial_screen_evidence
from spatial_grid_catboost_evaluation import engineer_spatial_grid_catboost_features

# These existing helpers strictly validate the four supplied data files,
# immutable incumbent probability caches and reconstruction of the .8298 CSV.
from generate_locked_architecture_submissions import _load_data
from generate_locked_architecture_submissions import _load_incumbent


def main(*, opt_in: str | None) -> None:
    require_explicit_opt_in(opt_in)
    _require_new_output_directory()
    source_hashes = _validate_source_hashes()
    evidence_hashes, evidence = _load_and_validate_evidence()

    # Competition data is deliberately opened only after every labelled-only
    # decision, evidence file and implementation source has passed validation.
    modelling, template = _load_data()
    incumbent, component_seconds, incumbent_hashes = _load_incumbent(
        modelling,
        template,
    )
    spatial_evidence = load_spatial_screen_evidence(SPATIAL_DIR, modelling)
    rf_evidence = load_rf_screen_evidence(RF_DIR, modelling)
    rf_spec = validate_rf_leaf_1_refit_recipe(rf_evidence.result)
    if spatial_evidence.iterations != 2436:
        raise ValueError("Fresh portfolio spatial-grid CatBoost iterations changed.")

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    spatial = _load_or_fit_component(
        RUNTIME_DIR / "spatial-grid-catboost.joblib",
        modelling,
        candidate="spatial_grid_catboost",
        recipe={
            "spec": asdict(spatial_evidence.spec),
            "iterations": spatial_evidence.iterations,
            "feature_engineer": (
                "spatial_grid_catboost_evaluation."
                "engineer_spatial_grid_catboost_features"
            ),
            "within_slot_target": "identity_catboost",
            "slot_weight": 0.20,
        },
        source_hashes=source_hashes,
        evidence_hashes={
            "fresh_reconstruction_result": evidence_hashes["result.json"],
            **{
                f"spatial_screen:{name}": digest
                for name, digest in spatial_evidence.hashes.items()
            },
        },
        fitter=lambda: fit_gpu_candidate_probabilities(
            spatial_evidence.spec,
            modelling.X_original,
            modelling.y_original,
            modelling.X_competition,
            iterations=spatial_evidence.iterations,
            catboost_feature_engineer=engineer_spatial_grid_catboost_features,
        ),
    )
    forest = _load_or_fit_component(
        RUNTIME_DIR / "rf-features-0-3.joblib",
        modelling,
        candidate="rf_features_0_3",
        recipe={
            "spec": asdict(rf_spec),
            "iterations": 500,
            "resolved_model_parameters": resolve_sklearn_tree_parameters(rf_spec),
            "preprocessor": "model_preprocessing.make_initial_preprocessor",
            "within_slot_target": "random_forest",
            "slot_weight": 0.18,
        },
        source_hashes=source_hashes,
        evidence_hashes={
            "fresh_reconstruction_result": evidence_hashes["result.json"],
            **{
                f"rf_screen:{name}": digest
                for name, digest in rf_evidence.hashes.items()
            },
        },
        fitter=lambda: fit_gpu_candidate_probabilities(
            rf_spec,
            modelling.X_original,
            modelling.y_original,
            modelling.X_competition,
            iterations=500,
        ),
    )
    probability_set = build_fresh_portfolio_probabilities(
        incumbent,
        spatial_grid_catboost=spatial["probabilities"],
        rf_features_0_3=forest["probabilities"],
    )
    incumbent_prediction = build_competition_prediction(
        modelling,
        template,
        probability_set.incumbent,
        component_seconds,
    )
    predictions = {
        candidate: build_competition_prediction(
            modelling,
            template,
            probabilities,
            component_seconds,
        )
        for candidate, probabilities in probability_set.candidates.items()
    }
    distinctness = analyse_portfolio_label_distinctness(
        incumbent_prediction.submission["status_group"],
        {
            candidate: prediction.submission["status_group"]
            for candidate, prediction in predictions.items()
        },
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    preflight_manifest = _manifest_base(
        opt_in,
        evidence,
        evidence_hashes,
        source_hashes,
        incumbent_hashes,
        spatial,
        forest,
        distinctness,
    )
    try:
        require_materially_distinct_portfolio(distinctness)
    except ValueError:
        write_json_without_overwrite(
            {
                **preflight_manifest,
                "status": "rejected_redundant_hard_label_outputs",
                "csv_files_written": 0,
                "candidates": [],
            },
            MANIFEST_PATH,
        )
        raise

    records = []
    incumbent_labels = incumbent_prediction.submission["status_group"]
    for candidate in PORTFOLIO_CANDIDATES:
        prediction = predictions[candidate]
        path = OUTPUT_DIR / PORTFOLIO_FILENAMES[candidate]
        write_submission_without_overwrite(
            prediction.submission,
            modelling.competition_ids,
            path,
        )
        reloaded = pd.read_csv(path)
        records.append(
            {
                "candidate": candidate,
                "evidence_tier": "exploratory",
                "csv": path.name,
                "sha256": sha256_file(path),
                "rows": len(reloaded),
                "unique_ids": int(reloaded["id"].nunique()),
                "disagreements_vs_incumbent": int(
                    reloaded["status_group"].ne(incumbent_labels).sum()
                ),
                "class_shares": {
                    label: float(share)
                    for label, share in prediction.class_shares.items()
                },
                "post_hoc_rows_changed": 0,
            }
        )
    write_json_without_overwrite(
        {
            **preflight_manifest,
            "status": PORTFOLIO_STATUS,
            "csv_files_written": 3,
            "candidates": records,
        },
        MANIFEST_PATH,
    )
    print(json.dumps(json.loads(MANIFEST_PATH.read_text()), indent=2), flush=True)


def _load_or_fit_component(
    path,
    modelling,
    *,
    candidate,
    recipe,
    source_hashes,
    evidence_hashes,
    fitter,
):
    metadata = make_component_cache_metadata(
        candidate=candidate,
        recipe=recipe,
        data_sha256=DATA_SHA256,
        source_sha256=source_hashes,
        evidence_sha256=evidence_hashes,
    )
    if path.exists():
        payload = joblib.load(path)
    else:
        probabilities, seconds = fitter()
        payload = make_component_cache_payload(
            metadata,
            modelling.competition_ids,
            probabilities,
            seconds,
        )
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite component cache: {path}.")
        joblib.dump(payload, path, compress=3)
        payload = joblib.load(path)
    return validate_component_cache(payload, metadata, modelling.competition_ids)


def validate_rf_leaf_1_refit_recipe(screen_result):
    """Return leaf-1 only when the real RF screen pins its exact recipe."""

    if not isinstance(screen_result, dict):
        raise ValueError("Fresh RF screen result is malformed.")
    rf_spec = make_locked_rf_specs()["Random Forest features 0.3"]
    expected_parameters = resolve_sklearn_tree_parameters(rf_spec)
    if (
        screen_result.get("locked_specs", {}).get(
            "Random Forest features 0.3"
        )
        != asdict(rf_spec)
        or screen_result.get("locked_model_parameters", {}).get(
            "Random Forest features 0.3"
        )
        != expected_parameters
        or expected_parameters.get("n_estimators") != 500
    ):
        raise ValueError("Fresh portfolio RF leaf-1 refit recipe changed.")
    return rf_spec


def _load_and_validate_evidence():
    if sha256_file(EVIDENCE_RESULT) != PINNED_EVIDENCE_RESULT_SHA256:
        raise ValueError("Fresh reconstruction result SHA-256 changed.")
    result = json.loads(EVIDENCE_RESULT.read_text(encoding="utf-8"))
    validate_fresh_portfolio_evidence(result)
    hashes = {"result.json": PINNED_EVIDENCE_RESULT_SHA256}
    for filename, expected in result["output_sha256"].items():
        path = EVIDENCE_DIR / filename
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Fresh reconstruction output changed: {filename}.")
        hashes[filename] = actual
    return hashes, result


def _validate_source_hashes():
    hashes = {}
    for relative, expected in PINNED_SOURCE_SHA256.items():
        actual = sha256_file(STAGE_DIR / relative)
        if actual != expected:
            raise ValueError(f"Fresh portfolio source changed: {relative}.")
        hashes[relative] = actual
    return hashes


def _require_new_output_directory() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Refusing to reuse fresh portfolio output directory: {OUTPUT_DIR}."
        )


def _manifest_base(
    opt_in,
    evidence,
    evidence_hashes,
    source_hashes,
    incumbent_hashes,
    spatial,
    forest,
    distinctness,
):
    return {
        "date": "2026-09-05",
        "explicit_cli_opt_in": opt_in,
        "auto_uploaded": False,
        "review_required_before_upload": True,
        "external_data_used": False,
        "protocol": {
            "training_rows": 59_400,
            "competition_rows": 14_850,
            "historical_local_subset_used": False,
            "competition_covariates": "supplied TestSetValues.csv only",
            "archive_slot_weights": evidence["deep_archive_weights"],
            "within_slot_weights": {"incumbent": 0.5, "alternative": 0.5},
            "weight_search": False,
            "row_level_rules": False,
            "post_hoc_rows_changed": 0,
            "candidate_order": list(PORTFOLIO_CANDIDATES),
        },
        "fresh_evidence_sha256": evidence_hashes,
        "pinned_source_sha256": source_hashes,
        "pinned_data_sha256": DATA_SHA256,
        "incumbent_input_sha256": incumbent_hashes,
        "generator_sha256": sha256_file(Path(__file__)),
        "replacement_component_caches": {
            "spatial_grid_catboost": {
                "path": str(
                    (RUNTIME_DIR / "spatial-grid-catboost.joblib").relative_to(
                        PROJECT_DIR
                    )
                ),
                "sha256": sha256_file(RUNTIME_DIR / "spatial-grid-catboost.joblib"),
                "metadata": spatial["metadata"],
                "fit_and_predict_seconds": float(spatial["fit_and_predict_seconds"]),
            },
            "rf_features_0_3": {
                "path": str(
                    (RUNTIME_DIR / "rf-features-0-3.joblib").relative_to(PROJECT_DIR)
                ),
                "sha256": sha256_file(RUNTIME_DIR / "rf-features-0-3.joblib"),
                "metadata": forest["metadata"],
                "fit_and_predict_seconds": float(forest["fit_and_predict_seconds"]),
            },
        },
        "hard_label_distinctness": {
            "disagreements_vs_incumbent": distinctness.vs_incumbent,
            "pairwise_disagreements": distinctness.pairwise,
            "redundant_candidates": list(distinctness.redundant_candidates),
            "redundant_pairs": list(distinctness.redundant_pairs),
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the fixed fresh-reconstruction submission portfolio."
    )
    parser.add_argument(
        "--opt-in",
        required=True,
        choices=(OPT_IN_TOKEN,),
        help="Deliberate acknowledgement of the exploratory evidence tier.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(opt_in=_parse_args().opt_in)

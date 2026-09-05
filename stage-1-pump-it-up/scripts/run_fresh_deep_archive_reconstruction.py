"""Reconstruct the exact deep archive on fresh all-label folds.

Only three incumbent components are missing on the seed-20260905 folds.  Each
is fitted and cached one fold at a time so an interrupted run is resumable.
The loader uses the two labelled source files only; it never opens historical
local membership or competition data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = (
    PROJECT_DIR
    / ".runtime"
    / "fresh-deep-archive-reconstruction-seed-20260905"
)
NORMALISED_DIR = PROJECT_DIR / ".runtime" / "normalised-top-common-screen"
GENERALISATION_DIR = PROJECT_DIR / ".runtime" / "generalisation-diagnosis"
STRICT_SEED_05_DIR = GENERALISATION_DIR / "strict-fold-cache-v1"
TWO_SEED_DIR = PROJECT_DIR / ".runtime" / "deep-xgboost-two-seed-variance-check"
RF_DIR = PROJECT_DIR / ".runtime" / "fresh-rf-family-screen-seed-20260905"
CATBOOST_DIR = PROJECT_DIR / ".runtime" / "fresh-spatial-grid-catboost"
TEN_SEED_DIR = (
    PROJECT_DIR / ".runtime" / "deep-xgboost-ten-seed-bag-fresh-20260905"
)
TEN_SEED_EVALUATION_PATH = TEN_SEED_DIR / "ten-seed-bag-evaluation.joblib"
TEN_SEED_EVALUATION_SHA256 = (
    "9f0ce3e120f4ac4ff48925f81450615d8fd9bc398b82601d57a888716f37daf4"
)
TEN_SEED_FIT_TIME_TWO_SEED_SHA256 = {
    "result.json": (
        "e71917a2a46c3630ff7b4398c38dcd4a1dc80783dd931334fa3e658e7e5bbe9d"
    ),
    "seed-average-evaluation.joblib": (
        "e8b3403e9e9d4ec7b587fb27c7ba966b9fdd463f6bab262febd6e46aea0bd1ae"
    ),
}
SOURCE_SHA256 = {
    "TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    "TrainingSetLabels.csv": (
        "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
    ),
}
FRESH_FOLD_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)
MISSING_COMPONENTS = (
    "frequency_random_forest",
    "spatial_random_forest",
    "spatial_xgboost",
)
CPU_COMPONENTS = MISSING_COMPONENTS[:2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from categorical_frequency_forest_evaluation import FREQUENCY_MINIMUM_SUPPORT
from categorical_frequency_forest_evaluation import FREQUENCY_MODEL_FEATURES
from categorical_frequency_forest_evaluation import FREQUENCY_SOURCE_FEATURES
from categorical_frequency_forest_evaluation import (
    make_categorical_frequency_preprocessor,
)
from data_partitioning import make_cross_validation
from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from deep_xgboost_seed_average import confirm_seed_average
from deep_xgboost_seed_average import rebuild_and_validate_evaluation
from deep_xgboost_ten_seed_bag import FIT_SEEDS
from deep_xgboost_ten_seed_bag import LOCKED_SEEDS
from deep_xgboost_ten_seed_bag import TEN_BAG_NAME
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from fresh_deep_archive_reconstruction import FRESH_RECONSTRUCTION_CACHE_VERSION
from fresh_deep_archive_reconstruction import CATBOOST_IDENTITY_GRID_BAG
from fresh_deep_archive_reconstruction import DEEP_SEED_BAG
from fresh_deep_archive_reconstruction import RF_CURRENT_LEAF_1_BAG
from fresh_deep_archive_reconstruction import RF_CURRENT_LEAF_2_BAG
from fresh_deep_archive_reconstruction import assemble_fold_evaluation
from fresh_deep_archive_reconstruction import (
    combine_fresh_deep_archive_fallback,
)
from fresh_deep_archive_reconstruction import (
    compare_fresh_deep_archive_candidates,
)
from fresh_deep_archive_reconstruction import (
    compare_fresh_deep_ten_seed_direct_replacement,
)
from fresh_deep_archive_reconstruction import DEEP_TEN_SEED_DIRECT_REPLACEMENT
from fresh_deep_archive_reconstruction import make_fold_cache_payload
from fresh_deep_archive_reconstruction import validate_fold_cache_payload
from fresh_rf_family_evaluation import FRESH_CROSS_VALIDATION_SEED
from fresh_rf_family_evaluation import CURRENT_RF_VARIANT
from fresh_rf_family_evaluation import FOLD_WIN_GATE as RF_FOLD_WIN_GATE
from fresh_rf_family_evaluation import (
    REPAIR_RECALL_DELTA_GATE as RF_REPAIR_RECALL_DELTA_GATE,
)
from fresh_rf_family_evaluation import (
    WORST_FOLD_DELTA_GATE as RF_WORST_FOLD_DELTA_GATE,
)
from fresh_rf_family_evaluation import make_full_labelled_partition
from fresh_rf_family_evaluation import make_locked_rf_specs
from fresh_rf_family_evaluation import validate_rf_cache_payload
from generalisation_diagnosis import (
    validate_fold_cache_payload as validate_generalisation_fold_cache,
)
from gpu_model_evaluation import INNER_STOP_FRACTION
from gpu_model_evaluation import INNER_STOP_SEED
from gpu_model_evaluation import XGBOOST_ITERATION_CAP
from gpu_model_evaluation import XGBOOST_STOPPING_ROUNDS
from gpu_model_evaluation import XGBOOST_VARIANTS
from gpu_model_evaluation import _fit_outer_fold
from gpu_model_evaluation import make_sklearn_tree_spec
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import resolve_sklearn_tree_parameters
from locked_architecture_candidates import sha256_file
from locked_architecture_candidates import write_json_without_overwrite
from locked_rf_ensemble_confirmation import ACCURACY_FIRST_RF_CHALLENGER
from locked_rf_ensemble_confirmation import compare_fresh_rf_probability_bags
from locked_rf_ensemble_confirmation import FORMAL_RF_CHALLENGER
from locked_catboost_probability_bag import (
    compare_fresh_catboost_probability_bag,
)
from locked_screen_evidence import load_rf_screen_evidence
from locked_screen_evidence import load_spatial_screen_evidence
from locked_screen_evidence import recompute_and_validate_oof_evaluation
from model_evaluation import RANDOM_FOREST_SEED
from modelling_data import ModellingData
from model_preprocessing import RARE_CATEGORY_MINIMUM
from run_deep_xgboost_seed_average import _load_seed_20260824
from run_generalisation_diagnosis import FOLD_CACHE_DIR
from run_generalisation_diagnosis import SOURCE_SHA256 as GENERALISATION_HASHES
from run_generalisation_diagnosis import _fold_cache_metadata
from run_generalisation_diagnosis import _model_recipe as generalisation_recipe
from source_data_validation import validate_aligned_ids
from spatial_height_imputation_evaluation import SPATIAL_HEIGHT_NEIGHBOURS
from spatial_height_imputation_evaluation import make_spatial_height_preprocessor
from top_common_identity_evaluation import TOP_COMMON_VALUES


def main(
    *,
    components: tuple[str, ...],
    cache_only: bool,
) -> None:
    modelling = _load_labelled_modelling_data()
    partitioned = make_full_labelled_partition(
        modelling,
        cross_validation_seed=FRESH_CROSS_VALIDATION_SEED,
    )
    if partitioned.cross_validation_fingerprint != FRESH_FOLD_FINGERPRINT:
        raise ValueError("Fresh all-label fold membership changed.")
    cross_validation = make_cross_validation(partitioned)
    splits = list(cross_validation.split())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for component in components:
        recipe = component_recipe(component)
        payloads = {}
        metadata_by_fold = {}
        for fold, (training_positions, validation_positions) in enumerate(
            splits,
            start=1,
        ):
            metadata = _fold_metadata(
                component,
                recipe,
                fold,
                len(training_positions),
                len(validation_positions),
            )
            expected_ids = partitioned.development_ids.iloc[
                validation_positions
            ].to_numpy()
            path = _fold_path(component, fold)
            if path.exists():
                payload = joblib.load(path)
                print(f"Loaded {path}.", flush=True)
            else:
                if cache_only:
                    raise FileNotFoundError(
                        f"Cache-only replay requires {path}."
                    )
                probabilities, diagnostics = _fit_component_fold(
                    component,
                    partitioned.X_development.iloc[training_positions],
                    partitioned.y_development.iloc[training_positions],
                    partitioned.X_development.iloc[validation_positions],
                    fold,
                )
                payload = make_fold_cache_payload(
                    metadata,
                    expected_ids,
                    probabilities,
                    diagnostics,
                )
                joblib.dump(payload, path, compress=3)
                payload = joblib.load(path)
                print(f"Cached {path}.", flush=True)
            validate_fold_cache_payload(payload, metadata, expected_ids)
            payloads[fold] = payload
            metadata_by_fold[fold] = metadata

        evaluation = assemble_fold_evaluation(
            partitioned,
            payloads,
            metadata_by_fold,
            component_model_name(component),
        )
        _write_aggregate_cache(
            component,
            recipe,
            payloads,
            evaluation,
            partitioned,
        )

    missing = [
        component
        for component in MISSING_COMPONENTS
        if not _aggregate_path(component).is_file()
    ]
    if missing:
        print(
            "Exact reconstruction remains intentionally pending: "
            + ", ".join(missing),
            flush=True,
        )
        return
    _reconstruct_and_score(modelling, partitioned)


def component_recipe(component: str) -> dict[str, object]:
    """Return the complete frozen estimator and feature recipe."""

    if component == "frequency_random_forest":
        spec = _forest_spec(component)
        return {
            "spec": asdict(spec),
            "resolved_model_parameters": resolve_sklearn_tree_parameters(spec),
            "preprocessor": (
                "categorical_frequency_forest_evaluation."
                "make_categorical_frequency_preprocessor"
            ),
            "frequency_source_features": list(FREQUENCY_SOURCE_FEATURES),
            "frequency_model_features": list(FREQUENCY_MODEL_FEATURES),
            "minimum_support": FREQUENCY_MINIMUM_SUPPORT,
            "normaliser": "target_encoding_features.normalise_identity",
            "numeric_imputation": "median with missing indicators",
            "class_order": list(CLASS_LABELS),
        }
    if component == "spatial_random_forest":
        spec = _forest_spec(component)
        return {
            "spec": asdict(spec),
            "resolved_model_parameters": resolve_sklearn_tree_parameters(spec),
            **_spatial_preprocessing_recipe(),
            "class_order": list(CLASS_LABELS),
        }
    if component == "spatial_xgboost":
        spec = _spatial_xgboost_spec()
        return {
            "spec": asdict(spec),
            "variant_parameters": dict(XGBOOST_VARIANTS[spec.variant]),
            "iteration_selection": {
                "cap": XGBOOST_ITERATION_CAP,
                "stopping_rounds": XGBOOST_STOPPING_ROUNDS,
                "inner_stop_fraction": INNER_STOP_FRACTION,
                "inner_stop_seed_base": INNER_STOP_SEED,
                "outer_fold_seed_offset": True,
                "refit_on_complete_outer_training_fold": True,
            },
            **_spatial_preprocessing_recipe(),
            "class_order": list(CLASS_LABELS),
        }
    raise ValueError(f"Unknown missing archive component: {component!r}.")


def component_model_name(component: str) -> str:
    return {
        "frequency_random_forest": (
            "Random Forest [all categorical occurrence counts]"
        ),
        "spatial_random_forest": "Random Forest [spatial height imputation]",
        "spatial_xgboost": (
            "XGBoost depth 8 child 1 [spatial height imputation]"
        ),
    }[component]


def _fit_component_fold(component, X_training, y_training, X_validation, fold):
    started = time.perf_counter()
    if component == "spatial_xgboost":
        probabilities, diagnostics = _fit_outer_fold(
            _spatial_xgboost_spec(),
            X_training,
            y_training,
            X_validation,
            fold_number=fold,
            preprocessor_factory=make_spatial_height_preprocessor,
            catboost_feature_engineer=None,
        )
    else:
        preprocessor = (
            make_categorical_frequency_preprocessor
            if component == "frequency_random_forest"
            else make_spatial_height_preprocessor
        )
        probabilities, diagnostics = _fit_outer_fold(
            _forest_spec(component),
            X_training,
            y_training,
            X_validation,
            fold_number=fold,
            preprocessor_factory=preprocessor,
            catboost_feature_engineer=None,
        )
    diagnostics = {
        **diagnostics,
        "validation_fold": fold,
        "total_seconds": time.perf_counter() - started,
    }
    validate_probabilities(probabilities, len(X_validation))
    print(
        f"Completed {component} fold {fold}/5 in "
        f"{diagnostics['total_seconds']:.1f}s.",
        flush=True,
    )
    return probabilities, diagnostics


def _fold_metadata(component, recipe, fold, training_rows, validation_rows):
    return {
        "candidate": component,
        "recipe": recipe,
        "source_sha256": SOURCE_SHA256,
        "labelled_rows": 59_400,
        "cross_validation_folds": 5,
        "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
        "validation_fold": fold,
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "historical_local_subset": "not reconstructed or consulted",
        "competition_data_opened": False,
    }


def _write_aggregate_cache(
    component,
    recipe,
    payloads,
    evaluation,
    partitioned,
) -> None:
    path = _aggregate_path(component)
    metadata = {
        "cache_version": FRESH_RECONSTRUCTION_CACHE_VERSION,
        "component": component,
        "recipe": recipe,
        "source_sha256": SOURCE_SHA256,
        "labelled_rows": 59_400,
        "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
        "fold_cache_sha256": {
            str(fold): sha256_file(_fold_path(component, fold))
            for fold in range(1, 6)
        },
        "historical_local_subset": "not reconstructed or consulted",
        "competition_data_opened": False,
    }
    aggregate = {"metadata": metadata, "evaluation": evaluation}
    if path.exists():
        existing = joblib.load(path)
        if not isinstance(existing, dict) or existing.get("metadata") != metadata:
            raise ValueError(f"Stale aggregate cache: {path}.")
        rebuild_and_validate_evaluation(
            existing.get("evaluation"),
            partitioned,
            make_cross_validation(partitioned),
            evidence_name=component,
        )
        if not existing["evaluation"].out_of_fold_probabilities.equals(
            evaluation.out_of_fold_probabilities
        ):
            raise ValueError(f"Aggregate cache probabilities changed: {path}.")
        return
    joblib.dump(aggregate, path, compress=3)
    print(f"Cached {path}.", flush=True)


def _reconstruct_and_score(modelling, partitioned) -> None:
    (
        incumbent,
        alternatives,
        evidence_hashes,
        component_evidence_eligibility,
        component_evidence,
    ) = _load_existing_evidence(modelling, partitioned)
    for component in MISSING_COMPONENTS:
        payload = joblib.load(_aggregate_path(component))
        evaluation = rebuild_and_validate_evaluation(
            payload["evaluation"],
            partitioned,
            make_cross_validation(partitioned),
            evidence_name=component,
        )
        incumbent[component] = evaluation.out_of_fold_probabilities.to_numpy(
            dtype="float64"
        )
        evidence_hashes[component] = sha256_file(_aggregate_path(component))

    ten_seed_evaluation, ten_seed_hashes = _load_ten_seed_evidence(partitioned)
    evidence_hashes["deep_ten_seed_bag"] = ten_seed_hashes

    comparison = compare_fresh_deep_archive_candidates(
        partitioned,
        incumbent,
        alternatives,
        component_evidence_eligibility=component_evidence_eligibility,
    )
    combined = combine_fresh_deep_archive_fallback(
        partitioned,
        comparison,
        (CATBOOST_IDENTITY_GRID_BAG, RF_CURRENT_LEAF_1_BAG),
    )
    ten_seed = compare_fresh_deep_ten_seed_direct_replacement(
        partitioned,
        incumbent,
        ten_seed_evaluation,
    )
    _write_csv_without_overwrite(
        OUTPUT_DIR / "candidate-summary.csv",
        comparison.summary,
    )
    _write_csv_without_overwrite(
        OUTPUT_DIR / "fold-paired-deltas.csv",
        comparison.fold_deltas,
        index=False,
    )
    _write_csv_without_overwrite(
        OUTPUT_DIR / "combined-catboost-rf-leaf-1-summary.csv",
        combined.summary,
    )
    _write_csv_without_overwrite(
        OUTPUT_DIR / "combined-catboost-rf-leaf-1-fold-paired-deltas.csv",
        combined.fold_deltas,
        index=False,
    )
    _write_csv_without_overwrite(
        OUTPUT_DIR / "ten-seed-direct-replacement-summary.csv",
        ten_seed.summary,
    )
    _write_csv_without_overwrite(
        OUTPUT_DIR / "ten-seed-direct-replacement-fold-paired-deltas.csv",
        ten_seed.fold_deltas,
        index=False,
    )
    all_summaries = pd.concat(
        [comparison.summary, ten_seed.summary, combined.summary],
        axis="rows",
        sort=False,
    )
    all_folds = pd.concat(
        [comparison.fold_deltas, ten_seed.fold_deltas, combined.fold_deltas],
        axis="rows",
        ignore_index=True,
        sort=False,
    )
    _write_csv_without_overwrite(
        OUTPUT_DIR / "all-candidate-summary.csv",
        all_summaries,
    )
    _write_csv_without_overwrite(
        OUTPUT_DIR / "all-fold-paired-deltas.csv",
        all_folds,
        index=False,
    )
    formal_passes = comparison.summary.index[
        comparison.summary["passes_formal_gate"]
    ].tolist()
    exploratory_passes = comparison.summary.index[
        comparison.summary["passes_exploratory_guard"]
    ].tolist()
    if bool(ten_seed.summary.iloc[0]["passes_formal_gate"]):
        formal_passes.append(DEEP_TEN_SEED_DIRECT_REPLACEMENT)
    combined_formal = bool(combined.summary.iloc[0]["passes_formal_gate"])
    combined_exploratory = bool(
        combined.summary.iloc[0]["passes_exploratory_guard"]
    )
    admissible_candidates = list(dict.fromkeys(formal_passes + exploratory_passes))
    if combined_formal or combined_exploratory:
        admissible_candidates.append(combined.candidate_evaluation.model_name)
    result = {
        "screen": "fresh exact deep-archive one-factor reconstruction",
        "status": "comparison_complete",
        "formal_passing_candidates": formal_passes,
        "exploratory_admissible_candidates": exploratory_passes,
        "admissible_candidates": admissible_candidates,
        "ten_seed_formal_only_append": {
            "candidate": DEEP_TEN_SEED_DIRECT_REPLACEMENT,
            "component_evidence_eligible": False,
            "exploratory_consideration_permitted": False,
            "result": json.loads(
                ten_seed.summary.reset_index().to_json(orient="records")
            )[0],
            "fold_deltas": json.loads(
                ten_seed.fold_deltas.to_json(orient="records")
            ),
        },
        "factorial_combination": {
            "candidate": combined.candidate_evaluation.model_name,
            "selected_candidates": list(combined.selected_candidates),
            "selected_families": list(combined.selected_families),
            "result": json.loads(
                combined.summary.reset_index().to_json(orient="records")
            )[0],
            "fold_deltas": json.loads(
                combined.fold_deltas.to_json(orient="records")
            ),
        },
        "deep_archive_weights": DEEP_ARCHIVE_WEIGHTS,
        "source_sha256": SOURCE_SHA256,
        "evidence_sha256": evidence_hashes,
        "component_evidence": component_evidence,
        "protocol": {
            "labelled_rows": 59_400,
            "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
            "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
            "historical_local_subset": "not reconstructed or consulted",
            "competition_data_opened": False,
            "competition_predictions_generated": False,
        },
        "results": json.loads(
            all_summaries.reset_index().to_json(orient="records")
        ),
        "output_sha256": {
            filename: sha256_file(OUTPUT_DIR / filename)
            for filename in (
                "candidate-summary.csv",
                "fold-paired-deltas.csv",
                "combined-catboost-rf-leaf-1-summary.csv",
                "combined-catboost-rf-leaf-1-fold-paired-deltas.csv",
                "ten-seed-direct-replacement-summary.csv",
                "ten-seed-direct-replacement-fold-paired-deltas.csv",
                "all-candidate-summary.csv",
                "all-fold-paired-deltas.csv",
            )
        },
    }
    write_json_without_overwrite(OUTPUT_DIR / "result.json", result)
    print(comparison.summary.to_string(), flush=True)


def _load_existing_evidence(modelling, partitioned):
    cross_validation = make_cross_validation(partitioned)
    source_hashes = {
        f"data/{name}": digest for name, digest in SOURCE_SHA256.items()
    }
    deep_24, deep_hashes = _load_seed_20260824(
        partitioned,
        cross_validation,
        source_hashes,
    )
    deep_05 = _load_seed_20260905(partitioned)
    deep_confirmation = confirm_seed_average(
        partitioned,
        cross_validation,
        {
            "seed_20260824": deep_24,
            "seed_20260905": deep_05,
        },
    )

    load_rf_screen_evidence(RF_DIR, modelling)
    rf_specs = make_locked_rf_specs()
    rf = {}
    for candidate in (
        CURRENT_RF_VARIANT,
        FORMAL_RF_CHALLENGER,
        ACCURACY_FIRST_RF_CHALLENGER,
    ):
        path = RF_DIR / f"{_slug(candidate)}.joblib"
        evaluation = validate_rf_cache_payload(
            joblib.load(path),
            rf_specs[candidate],
            partitioned,
        )
        rf[candidate] = recompute_and_validate_oof_evaluation(
            evaluation,
            partitioned,
            cross_validation,
            candidate=candidate,
        )
    rf_confirmation = compare_fresh_rf_probability_bags(partitioned, rf)

    load_spatial_screen_evidence(CATBOOST_DIR, modelling)
    cat = {}
    for candidate, filename in (
        ("complete_identity_catboost", "complete-identity-catboost.joblib"),
        ("spatial_grid_catboost", "spatial-grid-catboost.joblib"),
    ):
        evaluation = joblib.load(CATBOOST_DIR / filename)["trial"].evaluation
        cat[candidate] = recompute_and_validate_oof_evaluation(
            evaluation,
            partitioned,
            cross_validation,
            candidate=candidate,
        )
    cat_confirmation = compare_fresh_catboost_probability_bag(
        partitioned,
        cat["complete_identity_catboost"],
        cat["spatial_grid_catboost"],
    )

    deep_bag = deep_confirmation.average.out_of_fold_probabilities.to_numpy(
        dtype="float64"
    )
    incumbent = {
        "deep_xgboost": deep_24.out_of_fold_probabilities.to_numpy(
            dtype="float64"
        ),
        "random_forest": rf[CURRENT_RF_VARIANT]
        .out_of_fold_probabilities.to_numpy(dtype="float64"),
        "identity_catboost": cat[
            "complete_identity_catboost"
        ].out_of_fold_probabilities.to_numpy(dtype="float64"),
    }
    alternatives = {
        "spatial_grid_catboost": cat[
            "spatial_grid_catboost"
        ].out_of_fold_probabilities.to_numpy(dtype="float64"),
        "rf_features_0_3": rf[
            FORMAL_RF_CHALLENGER
        ].out_of_fold_probabilities.to_numpy(dtype="float64"),
        "rf_features_0_3_leaf_2": rf[
            ACCURACY_FIRST_RF_CHALLENGER
        ].out_of_fold_probabilities.to_numpy(dtype="float64"),
        "deep_seed_20260824_20260905_bag": deep_bag,
    }
    hashes = {
        "deep_seed_20260824": deep_hashes,
        "deep_seed_20260905_strict_folds": {
            str(fold): sha256_file(FOLD_CACHE_DIR / f"fold-{fold}.joblib")
            for fold in range(1, 6)
        },
        "rf_result": sha256_file(RF_DIR / "result.json"),
        "catboost_result": sha256_file(CATBOOST_DIR / "result.json"),
    }
    cat_row = cat_confirmation.summary.iloc[0]
    rf_leaf_1 = rf_confirmation.summary.loc[FORMAL_RF_CHALLENGER]
    rf_leaf_2 = rf_confirmation.summary.loc[ACCURACY_FIRST_RF_CHALLENGER]
    rf_leaf_1_stable_near_miss = bool(
        rf_leaf_1["mean_accuracy_delta"] > 0
        and int(rf_leaf_1["fold_wins_vs_incumbent"]) >= RF_FOLD_WIN_GATE
        and rf_leaf_1["worst_fold_delta"] >= RF_WORST_FOLD_DELTA_GATE
        and rf_leaf_1["repair_recall_delta"]
        >= RF_REPAIR_RECALL_DELTA_GATE
    )
    eligibility = {
        CATBOOST_IDENTITY_GRID_BAG: bool(cat_row["passes_gate"]),
        RF_CURRENT_LEAF_1_BAG: bool(
            rf_leaf_1["passes_gate"] or rf_leaf_1_stable_near_miss
        ),
        # This refers to the fixed within-slot 50:50 bag, which passes the
        # fresh RF gate.  The unbagged leaf-2 challenger remains a documented
        # repair-recall guard exception and is never substituted directly.
        RF_CURRENT_LEAF_2_BAG: bool(rf_leaf_2["passes_gate"]),
        DEEP_SEED_BAG: bool(
            deep_confirmation.passes_gate_against_both
            or deep_confirmation.stable_positive_near_miss
        ),
    }
    component_evidence = {
        "catboost_identity_grid_bag": json.loads(
            cat_confirmation.summary.reset_index().to_json(orient="records")
        ),
        "rf_probability_bags": json.loads(
            rf_confirmation.summary.reset_index().to_json(orient="records")
        ),
        "deep_two_seed_bag": {
            "passes_gate_against_both": (
                deep_confirmation.passes_gate_against_both
            ),
            "stable_positive_near_miss": (
                deep_confirmation.stable_positive_near_miss
            ),
            "verdict": deep_confirmation.verdict,
            "comparisons": json.loads(
                deep_confirmation.summary.reset_index().to_json(
                    orient="records"
                )
            ),
        },
        "eligibility": eligibility,
    }
    return incumbent, alternatives, hashes, eligibility, component_evidence


def _load_ten_seed_evidence(partitioned):
    """Load the predeclared ten-seed bag under its complete evidence recipe."""

    if not TEN_SEED_EVALUATION_PATH.is_file():
        raise FileNotFoundError(
            f"Ten-seed evidence is not complete: {TEN_SEED_EVALUATION_PATH}."
        )
    aggregate_sha256 = sha256_file(TEN_SEED_EVALUATION_PATH)
    if aggregate_sha256 != TEN_SEED_EVALUATION_SHA256:
        raise ValueError("Ten-seed aggregate evidence hash changed.")
    payload = joblib.load(TEN_SEED_EVALUATION_PATH)
    if not isinstance(payload, dict) or set(payload) != {
        "cache_version",
        "metadata",
        "evaluation",
    }:
        raise ValueError("Ten-seed aggregate evidence schema changed.")
    expected_metadata = _ten_seed_expected_metadata()
    if payload["cache_version"] != 1:
        raise ValueError("Ten-seed aggregate evidence version changed.")
    if payload["metadata"] != expected_metadata:
        raise ValueError("Ten-seed aggregate evidence metadata changed.")
    evaluation = rebuild_and_validate_evaluation(
        payload["evaluation"],
        partitioned,
        make_cross_validation(partitioned),
        evidence_name="ten_seed_bag",
    )
    if evaluation.model_name != TEN_BAG_NAME:
        raise ValueError("Ten-seed aggregate model identity changed.")
    return evaluation, {
        "aggregate": aggregate_sha256,
        "new_seed_fold_caches": expected_metadata[
            "new_seed_fold_cache_sha256"
        ],
        "baseline_evidence": expected_metadata["baseline_evidence_sha256"],
        "fit_time_baseline_evidence": expected_metadata[
            "fit_time_baseline_evidence_sha256"
        ],
        "validation_sources": expected_metadata[
            "current_validation_source_sha256"
        ],
    }


def _ten_seed_expected_metadata() -> dict[str, object]:
    current_sources = {
        f"data/{name}": digest for name, digest in SOURCE_SHA256.items()
    }
    for relative in (
        "src/deep_xgboost_ten_seed_bag.py",
        "src/deep_xgboost_seed_average.py",
        "src/gpu_model_evaluation.py",
        "src/top_common_identity_evaluation.py",
        "src/fresh_rf_family_evaluation.py",
        "src/data_preparation.py",
        "src/model_evaluation.py",
        "src/generalisation_diagnosis.py",
        "scripts/run_generalisation_diagnosis.py",
    ):
        current_sources[relative] = sha256_file(STAGE_DIR / relative)

    new_fold_hashes = {}
    for seed in FIT_SEEDS:
        new_fold_hashes[str(seed)] = {
            f"seed-{seed}-fold-{fold}.joblib": sha256_file(
                TEN_SEED_DIR / f"seed-{seed}-fold-{fold}.joblib"
            )
            for fold in range(1, 6)
        }
    baseline_hashes = {
        "seed_20260824": {
            "result.json": sha256_file(NORMALISED_DIR / "result.json"),
            "raw-top-50-identities-evaluation.joblib": sha256_file(
                NORMALISED_DIR / "raw-top-50-identities-evaluation.joblib"
            ),
            **{
                f"raw-top-50-identities-fold-{fold}.joblib": sha256_file(
                    NORMALISED_DIR
                    / f"raw-top-50-identities-fold-{fold}.joblib"
                )
                for fold in range(1, 6)
            },
        },
        "seed_20260905": {
            "result.json": sha256_file(GENERALISATION_DIR / "result.json"),
            "migration.json": sha256_file(
                STRICT_SEED_05_DIR / "migration.json"
            ),
            **{
                f"fold-{fold}.joblib": sha256_file(
                    STRICT_SEED_05_DIR / f"fold-{fold}.joblib"
                )
                for fold in range(1, 6)
            },
        },
        "fixed_two_seed_bag": {
            "result.json": sha256_file(TWO_SEED_DIR / "result.json"),
            "seed-average-evaluation.joblib": sha256_file(
                TWO_SEED_DIR / "seed-average-evaluation.joblib"
            ),
        },
    }
    fit_time_baseline_hashes = {
        key: dict(value) for key, value in baseline_hashes.items()
    }
    fit_time_baseline_hashes["fixed_two_seed_bag"] = dict(
        TEN_SEED_FIT_TIME_TWO_SEED_SHA256
    )
    return {
        "cache_version": 1,
        "candidate": "equal_ten_seed_raw_top_50_deep_xgboost_bag",
        "locked_seeds": list(LOCKED_SEEDS),
        "seed_weight": 0.1,
        "recipe_by_seed": {
            str(seed): _ten_seed_model_recipe(seed) for seed in LOCKED_SEEDS
        },
        "labelled_rows": 59_400,
        "folds": 5,
        "fold_seed": FRESH_CROSS_VALIDATION_SEED,
        "fold_fingerprint": FRESH_FOLD_FINGERPRINT,
        "data_sha256": SOURCE_SHA256,
        "current_validation_source_sha256": current_sources,
        "baseline_evidence_sha256": baseline_hashes,
        "fit_time_baseline_evidence_sha256": fit_time_baseline_hashes,
        "new_seed_fold_cache_sha256": new_fold_hashes,
        "competition_data_opened": False,
        "competition_predictions_generated": False,
    }


def _ten_seed_model_recipe(seed: int) -> dict[str, object]:
    spec = replace(
        make_xgboost_spec(variant="archived depth 17", seed=seed),
        name=(
            "XGBoost archived depth 17 "
            f"[raw top-50 identities; seed {seed}]"
        ),
        feature_policy="accepted_plus_raw_top_50_deferred_identities",
    )
    return {
        "spec": asdict(spec),
        "variant_parameters": dict(XGBOOST_VARIANTS[spec.variant]),
        "iterations": 600,
        "preprocessor": (
            "top_common_identity_evaluation."
            "make_top_common_identity_preprocessor"
        ),
        "identity_fields": list(DEFERRED_HIGH_CARDINALITY_FEATURES),
        "top_k": TOP_COMMON_VALUES,
        "normalise": False,
        "class_order": list(CLASS_LABELS),
    }


def _load_seed_20260905(partitioned):
    spec = make_xgboost_spec(variant="archived depth 17", seed=20260905)
    recipe = generalisation_recipe(spec)
    probabilities = np.full((59_400, len(CLASS_LABELS)), np.nan)
    diagnostics = []
    for fold, (_, positions) in enumerate(
        make_cross_validation(partitioned).split(),
        start=1,
    ):
        metadata = _fold_cache_metadata(
            fold=fold,
            training_rows=47_520,
            validation_rows=11_880,
            labelled_rows=59_400,
            fold_fingerprint=(
                "0341c8deaf52c4af835b414d28e565c1a6b11899021a5aa7fea283a6850c26d8"
            ),
            source_hashes=GENERALISATION_HASHES,
            model_recipe=recipe,
        )
        payload = joblib.load(FOLD_CACHE_DIR / f"fold-{fold}.joblib")
        fold_probabilities, seconds = validate_generalisation_fold_cache(
            payload,
            metadata,
            partitioned.development_ids.iloc[positions].to_numpy(),
        )
        probabilities[positions] = fold_probabilities
        diagnostics.append(
            {
                "validation_fold": fold,
                "selected_iterations": 600,
                "fit_seconds": seconds,
            }
        )
    from model_evaluation import build_candidate_evaluation

    return build_candidate_evaluation(
        model_name="XGBoost archived depth 17 seed 20260905",
        partitioned_data=partitioned,
        cross_validation=make_cross_validation(partitioned),
        probability_values=probabilities,
        diagnostic_rows=diagnostics,
    )


def _load_labelled_modelling_data() -> ModellingData:
    values_path = DATA_DIR / "TrainingSetValues.csv"
    labels_path = DATA_DIR / "TrainingSetLabels.csv"
    for path in (values_path, labels_path):
        if sha256_file(path) != SOURCE_SHA256[path.name]:
            raise ValueError(f"Locked source hash changed: {path.name}.")
    values = pd.read_csv(values_path)
    labels = pd.read_csv(labels_path)
    validate_aligned_ids(values, labels)
    if list(labels.columns) != ["id", "status_group"]:
        raise ValueError("Training label schema changed.")
    from data_preparation import remove_known_redundant_columns

    prepared = remove_known_redundant_columns(values)
    identifiers = prepared.pop("id").reset_index(drop=True)
    predictors = prepared.reset_index(drop=True)
    target = labels["status_group"].reset_index(drop=True)
    if predictors.shape != (59_400, 36) or target.shape != (59_400,):
        raise ValueError("Full-labelled modelling shape changed.")
    return ModellingData(
        original_ids=identifiers,
        X_original=predictors,
        y_original=target,
        competition_ids=identifiers.iloc[0:0].copy(),
        X_competition=predictors.iloc[0:0].copy(),
    )


def _spatial_xgboost_spec():
    from dataclasses import replace

    return replace(
        make_xgboost_spec(variant="depth 8 child 1"),
        name="XGBoost depth 8 child 1 [spatial height imputation]",
        feature_policy="ten-neighbour spatial GPS-height imputation",
    )


def _forest_spec(component: str):
    from dataclasses import replace

    return replace(
        make_sklearn_tree_spec(
            variant="Current Random Forest",
            seed=RANDOM_FOREST_SEED,
        ),
        name=component_model_name(component),
        feature_policy={
            "frequency_random_forest": (
                "fold-fitted all-categorical occurrence counts"
            ),
            "spatial_random_forest": (
                "ten-neighbour spatial GPS-height imputation"
            ),
        }[component],
    )


def _spatial_preprocessing_recipe() -> dict[str, object]:
    return {
        "preprocessor": (
            "spatial_height_imputation_evaluation."
            "make_spatial_height_preprocessor"
        ),
        "model_features": list(MODEL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "spatial_height_neighbours": SPATIAL_HEIGHT_NEIGHBOURS,
        "distance": "haversine inverse-distance weighting",
        "invalid_coordinate_fallback": "outer-training measured-height median",
        "numeric_imputation": "median with missing indicators",
        "categorical_encoding": "one-hot infrequent_if_exist",
        "rare_category_minimum": RARE_CATEGORY_MINIMUM,
    }


def _fold_path(component: str, fold: int) -> Path:
    return OUTPUT_DIR / f"{component.replace('_', '-')}-fold-{fold}.joblib"


def _aggregate_path(component: str) -> Path:
    return OUTPUT_DIR / f"{component.replace('_', '-')}-evaluation.joblib"


def _slug(value: str) -> str:
    return "-".join(value.casefold().split()).replace(".", "-")


def _write_csv_without_overwrite(
    path: Path,
    frame: pd.DataFrame,
    *,
    index: bool = True,
) -> None:
    expected = frame.to_csv(index=index, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(f"Refusing to overwrite different CSV: {path}.")
        return
    path.write_text(expected, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=MISSING_COMPONENTS,
        default=list(MISSING_COMPONENTS),
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Refuse to fit any absent component fold.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    main(
        components=tuple(arguments.components),
        cache_only=arguments.cache_only,
    )

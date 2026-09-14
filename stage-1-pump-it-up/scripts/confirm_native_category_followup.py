"""Rebuild the current seven-model incumbent on a second fixed fold plan.

This is a partition-stability check on the same labelled population, not a new
untouched dataset. Only challengers passing the locked first screen are fitted.
"""

from pathlib import Path
import json
import sys
import time

import numpy as np

STAGE = Path(__file__).resolve().parents[1]
PROJECT = STAGE.parent
sys.path.insert(0, str(STAGE / "src"))

from catboost_identity_evaluation import engineer_complete_identity_catboost_features
from data_partitioning import make_cross_validation
from deep_archive_confirmation import blend_deep_archive
from fresh_rf_family_evaluation import make_full_labelled_partition
from gpu_model_evaluation import (
    _fit_outer_fold, fit_gpu_candidate_probabilities,
    make_catboost_spec, make_sklearn_tree_spec, make_xgboost_spec,
)
from locked_architecture_candidates import sha256_file, write_json_without_overwrite
from model_preprocessing import make_initial_preprocessor
from native_category_followup import (
    BLENDS, CHALLENGERS, RECIPES, fit_fold, blend_candidates, compare_probabilities,
)
from spatial_grid_catboost_evaluation import engineer_spatial_grid_catboost_features
from top_common_identity_evaluation import make_top_common_identity_preprocessor
from run_fresh_deep_archive_reconstruction import (
    _fit_component_fold, _load_labelled_modelling_data, MISSING_COMPONENTS,
)
from run_native_category_followup_screen import OUTPUT as SCREEN, cached_fit, source_hashes

OUTPUT = PROJECT / ".runtime/native-category-followup-20260906-confirmation"
BASE_COMPONENTS = ("deep_xgboost", "spatial_xgboost", "random_forest",
                   "frequency_random_forest", "spatial_random_forest",
                   "identity_catboost", "spatial_grid_catboost")


def baseline_fit(component, X_train, y_train, X_validation, fold):
    started = time.perf_counter()
    if component in MISSING_COMPONENTS:
        return _fit_component_fold(component, X_train, y_train, X_validation, fold)
    if component == "deep_xgboost":
        probabilities, seconds = fit_gpu_candidate_probabilities(
            make_xgboost_spec(variant="archived depth 17", seed=20260824),
            X_train, y_train, X_validation, iterations=600,
            preprocessor_factory=make_top_common_identity_preprocessor,
        )
        return probabilities, {"validation_fold": fold, "selected_iterations": 600,
                               "total_seconds": seconds}
    if component == "random_forest":
        spec = make_sklearn_tree_spec(variant="Current Random Forest")
        engineer = None
    else:
        spec = make_catboost_spec(variant="d8")
        engineer = (engineer_complete_identity_catboost_features
                    if component == "identity_catboost"
                    else engineer_spatial_grid_catboost_features)
    probabilities, diagnostics = _fit_outer_fold(
        spec, X_train, y_train, X_validation, fold_number=fold,
        preprocessor_factory=make_initial_preprocessor,
        catboost_feature_engineer=engineer,
    )
    return probabilities, {**diagnostics, "validation_fold": fold,
                           "total_seconds": time.perf_counter() - started}


def main():
    screen_protocol = json.loads((SCREEN / "protocol.json").read_text())
    screen_result = json.loads((SCREEN / "result.json").read_text())
    if screen_result["protocol_sha256"] != sha256_file(SCREEN / "protocol.json"):
        raise ValueError("Screen protocol hash changed.")
    if screen_protocol["source_hashes"] != source_hashes():
        raise ValueError("Source implementations changed after screening.")
    for filename, digest in screen_result["fold_cache_sha256"].items():
        if sha256_file(SCREEN / filename) != digest:
            raise ValueError("Screen cache digest changed.")
    admitted = [candidate for candidate, blend in zip(CHALLENGERS, BLENDS)
                if screen_result["comparisons"][blend]["passes_guard"]]
    if admitted != screen_result["admitted_to_confirmation"] or not admitted:
        raise ValueError("No eligible locked challengers for confirmation.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    modelling = _load_labelled_modelling_data()
    partition = make_full_labelled_partition(modelling, cross_validation_seed=20260906)
    cv = make_cross_validation(partition)
    protocol = {"screen_result_sha256": sha256_file(SCREEN / "result.json"),
                "screen_protocol_sha256": sha256_file(SCREEN / "protocol.json"),
                "source_hashes": source_hashes(),
                "runner_sha256": sha256_file(Path(__file__)),
                "screen_runner_sha256": sha256_file(STAGE / "scripts/run_native_category_followup_screen.py"),
                "baseline_runner_sha256": sha256_file(STAGE / "scripts/run_fresh_deep_archive_reconstruction.py"),
                "fold_seed": 20260906, "fold_fingerprint": partition.cross_validation_fingerprint,
                "labelled_rows": len(modelling.y_original), "folds": 5,
                "confirmation_limitation": "new partition, same previously explored labelled population",
                "admitted": admitted, "recipes": {name: RECIPES[name] for name in admitted},
                "incumbent_components": list(BASE_COMPONENTS),
                "guard": screen_protocol["guard_on_both_screens"],
                "competition_data_opened": False,
                "full_data_iterations": "median of five confirmation inner-stop selections"}
    write_json_without_overwrite(OUTPUT / "protocol.json", protocol)
    probabilities = {name: np.full((len(modelling.y_original), 3), np.nan)
                     for name in (*BASE_COMPONENTS, *admitted)}
    diagnostics = {name: [] for name in probabilities}
    # Complete each fold before moving on, avoiding cross-fold cache mixing.
    for fold, (training, validation) in enumerate(cv.split(), 1):
        X_train = partition.X_development.iloc[training]
        y_train = partition.y_development.iloc[training]
        X_validation = partition.X_development.iloc[validation]
        for component in probabilities:
            metadata = {"protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
                        "candidate": component, "fold": fold,
                        "fold_fingerprint": partition.cross_validation_fingerprint}
            fitter = (lambda: fit_fold(component, X_train, y_train, X_validation, fold)) if component in admitted else (
                lambda: baseline_fit(component, X_train, y_train, X_validation, fold))
            probabilities[component][validation], diagnostic = cached_fit(
                OUTPUT, component, partition, validation, fold, metadata, fitter)
            diagnostics[component].append(diagnostic)

    grid = probabilities["spatial_grid_catboost"]
    archive = {name: probabilities[name] for name in BASE_COMPONENTS if name != "spatial_grid_catboost"}
    incumbent = blend_deep_archive(archive) + 0.1 * (grid - archive["identity_catboost"])
    candidates = blend_candidates(incumbent, grid, {name: probabilities[name] for name in admitted})
    comparisons = {name: compare_probabilities(partition.y_development, partition.validation_folds,
                                                incumbent, values)
                   for name, values in candidates.items()}
    eligible = [blend for candidate, blend in zip(CHALLENGERS, BLENDS)
                if candidate in admitted and comparisons[blend]["passes_guard"]]
    if len(eligible) == 2 and screen_result["comparisons"]["combined_locked_changes"]["eligible"] and comparisons["combined_locked_changes"]["passes_guard"]:
        eligible.append("combined_locked_changes")
    result = {"protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "comparisons": comparisons, "eligible_submissions": eligible,
              "full_data_iterations": {name: int(np.median([row["selected_iterations"]
                                                            for row in diagnostics[name]]))
                                       for name in admitted},
              "diagnostics": diagnostics,
              "fold_cache_sha256": {p.name: sha256_file(p) for p in sorted(OUTPUT.glob("*-fold-*.joblib"))}}
    write_json_without_overwrite(OUTPUT / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

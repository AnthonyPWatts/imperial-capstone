"""Screen two locked challengers on reused September 5 OOF evidence."""

from pathlib import Path
import sys

import joblib
import numpy as np

STAGE = Path(__file__).resolve().parents[1]
PROJECT = STAGE.parent
sys.path.insert(0, str(STAGE / "src"))

from data_partitioning import make_cross_validation
from deep_archive_confirmation import blend_deep_archive
from fresh_rf_family_evaluation import make_full_labelled_partition
from locked_architecture_candidates import sha256_file, write_json_without_overwrite
from native_category_followup import (
    CHALLENGERS, BLENDS, RECIPES, fit_fold, blend_candidates,
    compare_probabilities, cache_digest, validate_cache,
)
from run_fresh_deep_archive_reconstruction import (
    _load_labelled_modelling_data, _load_existing_evidence,
    MISSING_COMPONENTS, _aggregate_path, rebuild_and_validate_evaluation,
)

OUTPUT = PROJECT / ".runtime/native-category-followup-20260906-screen"


def source_hashes():
    return {str(path.relative_to(STAGE)): sha256_file(path)
            for path in sorted((STAGE / "src").glob("*.py"))}


def cached_fit(directory, component, partition, positions, fold, metadata, fitter):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{component}-fold-{fold}.joblib"
    ids = partition.development_ids.iloc[positions].to_numpy()
    if path.exists():
        payload = joblib.load(path)
    else:
        probabilities, diagnostics = fitter()
        payload = {"metadata": metadata, "ids": ids, "probabilities": probabilities,
                   "diagnostics": diagnostics,
                   "digest": cache_digest(metadata, ids, probabilities, diagnostics)}
        joblib.dump(payload, path, compress=3)
        payload = joblib.load(path)
    probabilities, diagnostics = validate_cache(payload, metadata, ids)
    print(f"{component} fold {fold}: {diagnostics}", flush=True)
    return probabilities, diagnostics


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    modelling = _load_labelled_modelling_data()
    partition = make_full_labelled_partition(modelling)
    cv = make_cross_validation(partition)
    protocol = {
        "recipes": RECIPES, "fold_seed": 20260905,
        "fold_fingerprint": partition.cross_validation_fingerprint,
        "fold_role": "reused screening; not untouched confirmation",
        "confirmation_seed": 20260906, "confirmation_folds": 5,
        "incumbent": "0.8304: six archive components with identity/grid CatBoost 50:50",
        "blends": {BLENDS[0]: "incumbent + 0.05 * (depth10 - grid_depth8)",
                   BLENDS[1]: "0.90 * incumbent + 0.10 * native_lightgbm",
                   "combined_locked_changes": "0.90 * depth10_blend + 0.10 * native_lightgbm; only if both individual guards pass"},
        "guard_on_both_screens": {"accuracy_delta": "> 0", "fold_wins": ">= 3/5",
                                  "worst_fold_delta": ">= -0.001", "repair_recall_delta": ">= -0.01"},
        "maximum_submissions": 3, "weight_search": False, "competition_data_opened": False,
        "inner_stopping": "10% stratified outer training; refit complete outer training",
        "source_hashes": source_hashes(), "runner_sha256": sha256_file(Path(__file__)),
        "data_hashes": {name: sha256_file(STAGE / "data" / name)
                        for name in ("TrainingSetValues.csv", "TrainingSetLabels.csv")},
    }
    write_json_without_overwrite(OUTPUT / "protocol.json", protocol)
    components, alternatives, _, _, _ = _load_existing_evidence(modelling, partition)
    for component in MISSING_COMPONENTS:
        evaluation = rebuild_and_validate_evaluation(
            joblib.load(_aggregate_path(component))["evaluation"], partition, cv,
            evidence_name=component)
        components[component] = evaluation.out_of_fold_probabilities.to_numpy(dtype="float64")
    grid = alternatives["spatial_grid_catboost"]
    incumbent = blend_deep_archive(components) + 0.1 * (grid - components["identity_catboost"])
    predictions = {}
    diagnostics_by_candidate = {}
    for candidate in CHALLENGERS:
        probabilities = np.full((len(modelling.y_original), 3), np.nan)
        diagnostics_by_candidate[candidate] = []
        for fold, (training, validation) in enumerate(cv.split(), 1):
            metadata = {"protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
                        "candidate": candidate, "fold": fold,
                        "fold_fingerprint": partition.cross_validation_fingerprint}
            probabilities[validation], diagnostic = cached_fit(
                OUTPUT, candidate, partition, validation, fold, metadata,
                lambda: fit_fold(candidate, partition.X_development.iloc[training],
                                 partition.y_development.iloc[training],
                                 partition.X_development.iloc[validation], fold))
            diagnostics_by_candidate[candidate].append(diagnostic)
        predictions[candidate] = probabilities
    comparisons = {name: compare_probabilities(partition.y_development, partition.validation_folds,
                                                incumbent, probabilities)
                   for name, probabilities in blend_candidates(incumbent, grid, predictions).items()}
    if not all(comparisons[name]["passes_guard"] for name in BLENDS):
        comparisons["combined_locked_changes"]["eligible"] = False
    else:
        comparisons["combined_locked_changes"]["eligible"] = comparisons["combined_locked_changes"]["passes_guard"]
    result = {"protocol_sha256": sha256_file(OUTPUT / "protocol.json"),
              "comparisons": comparisons, "diagnostics": diagnostics_by_candidate,
              "admitted_to_confirmation": [candidate for candidate, blend in zip(CHALLENGERS, BLENDS)
                                           if comparisons[blend]["passes_guard"]],
              "fold_cache_sha256": {p.name: sha256_file(p) for p in sorted(OUTPUT.glob("*-fold-*.joblib"))}}
    write_json_without_overwrite(OUTPUT / "result.json", result)
    import json
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

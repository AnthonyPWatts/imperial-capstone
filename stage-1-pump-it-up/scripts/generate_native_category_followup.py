"""Generate only twice-gated locked challengers, without uploading anything."""

from pathlib import Path
import argparse
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

STAGE = Path(__file__).resolve().parents[1]
PROJECT = STAGE.parent
sys.path.insert(0, str(STAGE / "src"))

from deep_archive_confirmation import blend_deep_archive
from final_model import build_competition_prediction
from locked_architecture_candidates import sha256_file, write_json_without_overwrite
from native_category_followup import (
    BLENDS, CHALLENGERS, RECIPES, blend_candidates, fit_probabilities,
    cache_digest, validate_cache,
)
from generate_locked_architecture_submissions import _load_data, _load_incumbent
from run_native_category_followup_screen import OUTPUT as SCREEN, source_hashes
from confirm_native_category_followup import OUTPUT as CONFIRMATION

OUTPUT = STAGE / "submissions/2026-09-06-native-category-followup"
RUNTIME = PROJECT / ".runtime/native-category-followup-20260906-competition"
BEST_CSV = STAGE / "submissions/2026-09-05-fresh-reconstruction-portfolio/02-fresh-catboost-identity-grid-50-50-within-slot-bag.csv"
BEST_SHA256 = "e492c7ca37ec12750809016a44ecf27fedafacda1eecb3e2db7c8e6368fa446b"
GRID_CACHE = PROJECT / ".runtime/fresh-reconstruction-portfolio-competition/spatial-grid-catboost.joblib"
GRID_SHA256 = "587b7301e8da7c038f7853cce0b6f5c6cd10445a5f0ecfb5df055799ecab691e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", required=True)
    parser.parse_args()
    screen = json.loads((SCREEN / "result.json").read_text())
    confirmation = json.loads((CONFIRMATION / "result.json").read_text())
    protocol = json.loads((CONFIRMATION / "protocol.json").read_text())
    for directory, result in ((SCREEN, screen), (CONFIRMATION, confirmation)):
        if sha256_file(directory / "protocol.json") != result["protocol_sha256"]:
            raise ValueError("Evidence protocol hash changed.")
        for filename, digest in result["fold_cache_sha256"].items():
            if sha256_file(directory / filename) != digest:
                raise ValueError("Evidence cache hash changed.")
    if protocol["screen_result_sha256"] != sha256_file(SCREEN / "result.json"):
        raise ValueError("Linked screen evidence changed.")
    if protocol["source_hashes"] != source_hashes():
        raise ValueError("Fitted source implementations changed.")
    if protocol["runner_sha256"] != sha256_file(STAGE / "scripts/confirm_native_category_followup.py"):
        raise ValueError("Confirmation runner changed.")
    eligible = [blend for blend in BLENDS if screen["comparisons"][blend]["passes_guard"]
                and blend in confirmation["comparisons"]
                and confirmation["comparisons"][blend]["passes_guard"]]
    if (len(eligible) == 2 and screen["comparisons"]["combined_locked_changes"]["eligible"]
            and confirmation["comparisons"]["combined_locked_changes"]["passes_guard"]):
        eligible.append("combined_locked_changes")
    if eligible != confirmation["eligible_submissions"] or not eligible:
        raise ValueError("No valid twice-gated submission candidates.")
    for path, digest in ((BEST_CSV, BEST_SHA256), (GRID_CACHE, GRID_SHA256)):
        if sha256_file(path) != digest:
            raise ValueError(f"Incumbent evidence hash changed: {path.name}.")

    # Only now open competition covariates and load immutable incumbent outputs.
    modelling, template = _load_data()
    components, seconds, incumbent_hashes = _load_incumbent(modelling, template)
    grid_payload = joblib.load(GRID_CACHE)
    if not np.array_equal(grid_payload["competition_ids"], modelling.competition_ids):
        raise ValueError("Grid cache competition IDs changed.")
    grid = grid_payload["probabilities"]
    incumbent = blend_deep_archive(components) + 0.1 * (grid - components["identity_catboost"])
    incumbent_prediction = build_competition_prediction(modelling, template, incumbent, seconds)
    if not pd.read_csv(BEST_CSV).equals(incumbent_prediction.submission.reset_index(drop=True)):
        raise ValueError("Current-best CSV could not be reconstructed exactly.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    needed = [candidate for candidate, blend in zip(CHALLENGERS, BLENDS) if blend in eligible]
    predictions = {}
    for candidate in needed:
        iterations = confirmation["full_data_iterations"][candidate]
        metadata = {"recipe": RECIPES[candidate], "iterations": iterations,
                    "confirmation_result_sha256": sha256_file(CONFIRMATION / "result.json"),
                    "source_hashes": source_hashes(), "generator_sha256": sha256_file(Path(__file__)),
                    "data_hashes": {name: sha256_file(STAGE / "data" / name) for name in
                                    ("TrainingSetValues.csv", "TrainingSetLabels.csv", "TestSetValues.csv", "SubmissionFormat.csv")}}
        path = RUNTIME / f"{candidate}.joblib"
        ids = modelling.competition_ids.to_numpy()
        if path.exists():
            payload = joblib.load(path)
        else:
            started = time.perf_counter()
            probabilities = fit_probabilities(candidate, modelling.X_original,
                                              modelling.y_original, modelling.X_competition,
                                              iterations=iterations)
            diagnostics = {"iterations": iterations, "seconds": time.perf_counter() - started}
            payload = {"metadata": metadata, "ids": ids, "probabilities": probabilities,
                       "diagnostics": diagnostics,
                       "digest": cache_digest(metadata, ids, probabilities, diagnostics)}
            joblib.dump(payload, path, compress=3)
            payload = joblib.load(path)
        predictions[candidate], _ = validate_cache(payload, metadata, ids)

    blended = blend_candidates(incumbent, grid, predictions)
    records = []
    seen_labels = []
    for index, name in enumerate(eligible, 1):
        prediction = build_competition_prediction(modelling, template, blended[name], seconds)
        submission = prediction.submission.reset_index(drop=True)
        changed = int((submission.status_group != incumbent_prediction.submission.status_group).sum())
        if changed == 0 or any(submission.equals(previous) for previous in seen_labels):
            raise ValueError("A generated candidate duplicates existing labels.")
        seen_labels.append(submission)
        path = OUTPUT / f"{index:02d}-{name.replace('_', '-')}.csv"
        if path.exists():
            if not pd.read_csv(path).equals(submission):
                raise ValueError("Refusing to overwrite changed competition predictions.")
        else:
            submission.to_csv(path, index=False)
        if not pd.read_csv(path).equals(submission):
            raise ValueError("Competition CSV round-trip failed.")
        records.append({"candidate": name, "file": path.name, "sha256": sha256_file(path),
                        "rows": len(submission), "changed_labels_vs_0_8304": changed,
                        "screen": screen["comparisons"][name],
                        "confirmation": confirmation["comparisons"][name]})
    manifest = {"status": "generated and validated; not submitted by this script",
                "selection": "two locked partition guards; no leaderboard-adaptive edits",
                "screen_result_sha256": sha256_file(SCREEN / "result.json"),
                "confirmation_result_sha256": sha256_file(CONFIRMATION / "result.json"),
                "generator_sha256": sha256_file(Path(__file__)),
                "incumbent_sha256": BEST_SHA256, "grid_cache_sha256": GRID_SHA256,
                "incumbent_component_hashes": incumbent_hashes,
                "new_component_hashes": {path.name: sha256_file(path) for path in sorted(RUNTIME.glob("*.joblib"))},
                "candidates": records}
    write_json_without_overwrite(OUTPUT / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

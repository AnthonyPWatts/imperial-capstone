"""Independently replay the locked confirmation from aligned fold caches."""

from pathlib import Path
import json
import sys

import joblib
import numpy as np

STAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE / "src"))
from data_partitioning import make_cross_validation
from fresh_rf_family_evaluation import make_full_labelled_partition
from locked_architecture_candidates import sha256_file, write_json_without_overwrite
from native_category_followup import validate_cache, encode_target
from run_fresh_deep_archive_reconstruction import _load_labelled_modelling_data
from confirm_native_category_followup import OUTPUT


def main():
    result = json.loads((OUTPUT / "result.json").read_text())
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    protocol_hash = sha256_file(OUTPUT / "protocol.json")
    if result["protocol_sha256"] != protocol_hash:
        raise ValueError("Confirmation protocol changed.")
    modelling = _load_labelled_modelling_data()
    partition = make_full_labelled_partition(modelling, cross_validation_seed=20260906)
    if partition.cross_validation_fingerprint != protocol["fold_fingerprint"]:
        raise ValueError("Confirmation fold assignment changed.")
    # Spell out the current seven-model weights independently of the blend helper.
    weights = {"deep_xgboost": 0.33, "spatial_xgboost": 0.11, "random_forest": 0.18,
               "frequency_random_forest": 0.09, "spatial_random_forest": 0.09,
               "identity_catboost": 0.10, "spatial_grid_catboost": 0.10}
    probabilities = {name: np.full((59_400, 3), np.nan)
                     for name in (*weights, *protocol["admitted"])}
    for fold, (_, validation) in enumerate(make_cross_validation(partition).split(), 1):
        for name in probabilities:
            path = OUTPUT / f"{name}-fold-{fold}.joblib"
            if sha256_file(path) != result["fold_cache_sha256"][path.name]:
                raise ValueError("Fold evidence changed.")
            metadata = {"protocol_sha256": protocol_hash, "candidate": name,
                        "fold": fold, "fold_fingerprint": partition.cross_validation_fingerprint}
            probabilities[name][validation], _ = validate_cache(
                joblib.load(path), metadata, partition.development_ids.iloc[validation].to_numpy())
    incumbent = sum(weight * probabilities[name] for name, weight in weights.items())
    target = encode_target(partition.y_development)
    baseline = incumbent.argmax(axis=1)
    candidates = {}
    if "catboost_grid_depth10" in probabilities:
        candidates["depth10_grid_half_slot"] = incumbent - 0.05 * probabilities["spatial_grid_catboost"] + 0.05 * probabilities["catboost_grid_depth10"]
    if "lightgbm_native_identity_grid" in probabilities:
        candidates["native_lightgbm_at_0_10"] = 0.9 * incumbent + 0.1 * probabilities["lightgbm_native_identity_grid"]
    if len(candidates) == 2:
        candidates["combined_locked_changes"] = 0.9 * candidates["depth10_grid_half_slot"] + 0.1 * probabilities["lightgbm_native_identity_grid"]
    rows = {}
    for name, values in candidates.items():
        prediction = values.argmax(axis=1)
        corrected = int(((baseline != target) & (prediction == target)).sum())
        regressed = int(((baseline == target) & (prediction != target)).sum())
        paired = (prediction == target).astype(int) - (baseline == target).astype(int)
        delta = float(paired.mean())
        standard_error = float(paired.std(ddof=1) / np.sqrt(len(paired)))
        expected = result["comparisons"][name]
        actual_folds = [float(paired[partition.validation_folds.to_numpy() == fold].mean())
                        for fold in range(1, 6)]
        if corrected - regressed != expected["net_correct_rows"]:
            raise ValueError("Independent net-row replay differed.")
        np.testing.assert_allclose(actual_folds, expected["fold_deltas"], atol=1e-15, rtol=0)
        np.testing.assert_allclose((prediction == target).mean(), expected["accuracy"], atol=1e-15, rtol=0)
        rows[name] = {"corrected_rows": corrected, "regressed_rows": regressed,
                      "net_correct_rows": corrected - regressed,
                      "accuracy_delta": delta,
                      "descriptive_paired_95_percent_interval": [delta - 1.96 * standard_error,
                                                                delta + 1.96 * standard_error],
                      "interval_caveat": "normal approximation; ignores adaptive selection and correlated overlapping training folds"}
    audit = {"result_sha256": sha256_file(OUTPUT / "result.json"),
             "status": "independent seven-weight reconstruction and all fold/ID/digest checks passed",
             "candidates": rows}
    write_json_without_overwrite(OUTPUT / "independent-audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

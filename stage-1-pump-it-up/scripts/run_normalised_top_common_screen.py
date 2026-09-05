"""Compare raw and normalised top-50 identities on one locked fresh design."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
import hashlib
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
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "normalised-top-common-screen"
FOLD_SEED = 20260905
MODEL_SEED = 20260824
FOLDS = 5
ITERATIONS = 600
CACHE_VERSION = 1
EXPECTED_FOLD_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)
EXPECTED_SOURCE_SHA256 = {
    "TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    "TrainingSetLabels.csv": (
        "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
    ),
}
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from final_model import validate_probabilities
from fresh_rf_family_evaluation import make_full_labelled_partition
from gpu_model_evaluation import CLASS_LABELS, XGBOOST_VARIANTS
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import build_candidate_evaluation
from modelling_data import prepare_modelling_data
from normalised_top_common_evaluation import FOLD_WIN_GATE
from normalised_top_common_evaluation import MEAN_ACCURACY_GAIN_GATE
from normalised_top_common_evaluation import REPAIR_RECALL_DELTA_GATE
from normalised_top_common_evaluation import WORST_FOLD_DELTA_GATE
from normalised_top_common_evaluation import summarise_normalised_comparison
from top_common_identity_evaluation import TOP_COMMON_VALUES
from top_common_identity_evaluation import make_normalised_top_common_identity_preprocessor
from top_common_identity_evaluation import make_top_common_identity_preprocessor


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = _validate_source_hashes()
    modelling = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = make_full_labelled_partition(
        modelling,
        cross_validation_seed=FOLD_SEED,
    )
    if partitioned.cross_validation_fingerprint != EXPECTED_FOLD_FINGERPRINT:
        raise ValueError("Locked fresh-fold membership changed.")
    cross_validation = make_cross_validation(partitioned)
    representations = {
        "raw_top_50_identities": {
            "normalise": False,
            "preprocessor": make_top_common_identity_preprocessor,
            "label": "raw top-50 deferred identities",
            "feature_policy": "accepted_plus_raw_top_50_deferred_identities",
        },
        "normalised_top_50_identities": {
            "normalise": True,
            "preprocessor": make_normalised_top_common_identity_preprocessor,
            "label": "normalised top-50 deferred identities",
            "feature_policy": "accepted_plus_normalised_top_50_deferred_identities",
        },
    }
    evaluations = {}
    recipes = {}
    for candidate, representation in representations.items():
        spec = replace(
            make_xgboost_spec(variant="archived depth 17", seed=MODEL_SEED),
            name=f"XGBoost archived depth 17 [{representation['label']}; seed {MODEL_SEED}]",
            feature_policy=representation["feature_policy"],
        )
        recipe = _recipe(spec, normalise=representation["normalise"])
        recipes[candidate] = recipe
        evaluations[candidate] = _evaluate(
            candidate,
            spec,
            recipe,
            representation["preprocessor"],
            partitioned,
            cross_validation,
            source_hashes,
        )

    summary, folds = summarise_normalised_comparison(
        evaluations["raw_top_50_identities"],
        evaluations["normalised_top_50_identities"],
    )
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    folds.to_csv(OUTPUT_DIR / "fold-deltas.csv")
    for candidate, evaluation in evaluations.items():
        joblib.dump(
            {"recipe": recipes[candidate], "evaluation": evaluation},
            OUTPUT_DIR / f"{candidate.replace('_', '-')}-evaluation.joblib",
            compress=3,
        )
    passes = bool(
        summary.loc["normalised_top_50_identities", "passes_gate"]
    )
    result = {
        "protocol": {
            "labelled_rows": len(partitioned.y_development),
            "folds": FOLDS,
            "fold_seed": FOLD_SEED,
            "fold_fingerprint": partitioned.cross_validation_fingerprint,
            "model_seed": MODEL_SEED,
            "iterations": ITERATIONS,
            "top_k": TOP_COMMON_VALUES,
            "selection": "single predeclared raw-versus-normalised comparison",
            "historical_local_subset": "not reconstructed or consulted",
            "competition_predictions_generated": False,
        },
        "gate": {
            "minimum_mean_accuracy_gain": MEAN_ACCURACY_GAIN_GATE,
            "minimum_fold_wins": FOLD_WIN_GATE,
            "minimum_worst_fold_delta": WORST_FOLD_DELTA_GATE,
            "minimum_repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
        },
        "source_sha256": source_hashes,
        "recipes": recipes,
        "candidate_summary": _json_records(summary),
        "fold_deltas": _json_records(folds),
        "passes_gate": passes,
        "selected_candidate": (
            "normalised_top_50_identities" if passes else "raw_top_50_identities"
        ),
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


def _evaluate(
    candidate,
    spec,
    recipe,
    preprocessor_factory,
    partitioned,
    cross_validation,
    source_hashes,
):
    probabilities = np.full((len(partitioned.y_development), len(CLASS_LABELS)), np.nan)
    diagnostics = []
    for fold, (training_positions, validation_positions) in enumerate(
        cross_validation.split(), start=1
    ):
        metadata = {
            "cache_version": CACHE_VERSION,
            "candidate": candidate,
            "recipe": recipe,
            "source_sha256": source_hashes,
            "labelled_rows": len(partitioned.y_development),
            "folds": FOLDS,
            "fold_seed": FOLD_SEED,
            "fold_fingerprint": partitioned.cross_validation_fingerprint,
            "validation_fold": fold,
        }
        expected_ids = partitioned.development_ids.iloc[validation_positions].to_numpy()
        path = OUTPUT_DIR / f"{candidate.replace('_', '-')}-fold-{fold}.joblib"
        if path.exists():
            payload = joblib.load(path)
            _validate_fold_cache(payload, metadata, expected_ids, len(validation_positions), path)
            fold_probabilities = payload["probabilities"]
            fold_diagnostics = payload["diagnostics"]
            print(f"Loaded {path}.", flush=True)
        else:
            started = time.perf_counter()
            fold_probabilities, fit_seconds = fit_gpu_candidate_probabilities(
                spec,
                partitioned.X_development.iloc[training_positions],
                partitioned.y_development.iloc[training_positions],
                partitioned.X_development.iloc[validation_positions],
                iterations=ITERATIONS,
                preprocessor_factory=preprocessor_factory,
            )
            fold_diagnostics = {
                "validation_fold": fold,
                "selected_iterations": ITERATIONS,
                "inner_fit_rows": len(training_positions),
                "inner_stop_rows": 0,
                "stopping_seconds": 0.0,
                "refit_predict_seconds": fit_seconds,
                "total_seconds": time.perf_counter() - started,
            }
            payload = {
                "metadata": metadata,
                "validation_ids": expected_ids,
                "probabilities": fold_probabilities,
                "diagnostics": fold_diagnostics,
            }
            _validate_fold_cache(payload, metadata, expected_ids, len(validation_positions), path)
            joblib.dump(payload, path, compress=3)
            print(
                f"Completed {candidate} fold {fold}/{FOLDS} in "
                f"{fold_diagnostics['total_seconds']:.1f}s.",
                flush=True,
            )
        probabilities[validation_positions] = fold_probabilities
        diagnostics.append(fold_diagnostics)
    validate_probabilities(probabilities, len(partitioned.y_development))
    return build_candidate_evaluation(
        model_name=spec.name,
        partitioned_data=partitioned,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=diagnostics,
    )


def _validate_fold_cache(payload, metadata, expected_ids, expected_rows, path) -> None:
    expected_keys = {"metadata", "validation_ids", "probabilities", "diagnostics"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"Malformed fold cache: {path}")
    if payload["metadata"] != metadata:
        raise ValueError(f"Stale fold cache metadata: {path}")
    if not np.array_equal(payload["validation_ids"], expected_ids):
        raise ValueError(f"Misaligned validation IDs in fold cache: {path}")
    validate_probabilities(payload["probabilities"], expected_rows)
    diagnostics = payload["diagnostics"]
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("validation_fold") != metadata["validation_fold"]
        or diagnostics.get("selected_iterations") != ITERATIONS
    ):
        raise ValueError(f"Invalid fold diagnostics: {path}")


def _recipe(spec, *, normalise: bool) -> dict[str, object]:
    return {
        "spec": asdict(spec),
        "variant_parameters": dict(XGBOOST_VARIANTS[spec.variant]),
        "iterations": ITERATIONS,
        "identity_fields": list(DEFERRED_HIGH_CARDINALITY_FEATURES),
        "top_k": TOP_COMMON_VALUES,
        "normalise": normalise,
        "normaliser": (
            "target_encoding_features.normalise_identity: "
            "strip, collapse whitespace, casefold, distinct blank/missing sentinels"
            if normalise
            else "raw string values with missing sentinel"
        ),
    }


def _validate_source_hashes() -> dict[str, str]:
    hashes = {}
    for filename, expected in EXPECTED_SOURCE_SHA256.items():
        actual = _sha256(DATA_DIR / filename)
        if actual != expected:
            raise ValueError(f"Source hash changed for {filename}.")
        hashes[filename] = actual
    return hashes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.reset_index().to_json(orient="records", double_precision=15))


if __name__ == "__main__":
    main()

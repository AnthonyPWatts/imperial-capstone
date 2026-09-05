"""Confirm the predeclared two-seed deep-XGBoost average from caches only."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
import hashlib
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
NORMALISED_DIR = PROJECT_DIR / ".runtime" / "normalised-top-common-screen"
GENERALISATION_DIR = PROJECT_DIR / ".runtime" / "generalisation-diagnosis"
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "deep-xgboost-two-seed-variance-check"
FOLD_SEED = 20260905
SEED_20260824 = 20260824
SEED_20260905 = 20260905
FOLDS = 5
ITERATIONS = 600
CACHE_VERSION = 1
EXPECTED_FOLD_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)
EXPECTED_GENERALISATION_FINGERPRINT = (
    "0341c8deaf52c4af835b414d28e565c1a6b11899021a5aa7fea283a6850c26d8"
)
EXPECTED_SOURCE_SHA256 = {
    "TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    "TrainingSetLabels.csv": (
        "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
    ),
}
EXPECTED_PRODUCER_SHA256 = {
    "run_normalised_top_common_screen.py": (
        "270c735f5307c2db05082c2c7f09b1d35cd09878c5a16139dfb458da2dab8f31"
    ),
    "run_generalisation_diagnosis.py": (
        "a5f71248b4d3a180b8113c5517459ebbe24582f4a43312b00c471bb8f0b1026a"
    ),
}
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from data_preparation import remove_known_redundant_columns
from deep_xgboost_seed_average import FOLD_WIN_GATE
from deep_xgboost_seed_average import MEAN_ACCURACY_GAIN_GATE
from deep_xgboost_seed_average import REPAIR_RECALL_DELTA_GATE
from deep_xgboost_seed_average import WORST_FOLD_DELTA_GATE
from deep_xgboost_seed_average import confirm_seed_average
from deep_xgboost_seed_average import rebuild_and_validate_evaluation
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from final_model import validate_probabilities
from fresh_rf_family_evaluation import make_full_labelled_partition
from gpu_model_evaluation import CLASS_LABELS
from gpu_model_evaluation import XGBOOST_VARIANTS
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import build_candidate_evaluation
from model_evaluation import CandidateEvaluation
from modelling_data import ModellingData
from source_data_validation import validate_aligned_ids
from top_common_identity_evaluation import TOP_COMMON_VALUES


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = _validate_locked_sources()
    modelling = _load_labelled_modelling_data()
    partitioned = make_full_labelled_partition(
        modelling,
        cross_validation_seed=FOLD_SEED,
    )
    if partitioned.cross_validation_fingerprint != EXPECTED_FOLD_FINGERPRINT:
        raise ValueError("Locked fresh-fold membership changed.")
    cross_validation = make_cross_validation(partitioned)
    recipes = {
        "seed_20260824": _canonical_recipe(SEED_20260824),
        "seed_20260905": _canonical_recipe(SEED_20260905),
        "probability_average": {
            "operation": "row-wise arithmetic mean of aligned class probabilities",
            "weights": {
                "seed_20260824": 0.5,
                "seed_20260905": 0.5,
            },
            "class_order": list(CLASS_LABELS),
            "selection": "single predeclared weight; no search",
        },
    }
    seed_20260824, hashes_20260824 = _load_seed_20260824(
        partitioned,
        cross_validation,
        source_hashes,
    )
    seed_20260905, hashes_20260905 = _load_seed_20260905(
        partitioned,
        cross_validation,
    )
    confirmation = confirm_seed_average(
        partitioned,
        cross_validation,
        {
            "seed_20260824": seed_20260824,
            "seed_20260905": seed_20260905,
        },
    )

    summary_path = OUTPUT_DIR / "comparison-summary.csv"
    folds_path = OUTPUT_DIR / "fold-paired-deltas.csv"
    evaluation_path = OUTPUT_DIR / "seed-average-evaluation.joblib"
    confirmation.summary.to_csv(summary_path)
    confirmation.fold_deltas.to_csv(folds_path, index=False)
    evidence_hashes = {
        "seed_20260824": hashes_20260824,
        "seed_20260905": hashes_20260905,
    }
    cache_metadata = {
        "cache_version": CACHE_VERSION,
        "labelled_rows": len(partitioned.y_development),
        "folds": FOLDS,
        "fold_seed": FOLD_SEED,
        "fold_fingerprint": partitioned.cross_validation_fingerprint,
        "recipes": recipes,
        "source_sha256": source_hashes,
        "evidence_sha256": evidence_hashes,
        "competition_predictions_generated": False,
    }
    joblib.dump(
        {
            "cache_version": CACHE_VERSION,
            "metadata": cache_metadata,
            "evaluation": confirmation.average,
        },
        evaluation_path,
        compress=3,
    )
    _validate_output_cache(
        joblib.load(evaluation_path),
        cache_metadata,
        partitioned,
        cross_validation,
    )
    output_hashes = {
        path.name: _sha256(path)
        for path in (summary_path, folds_path, evaluation_path)
    }
    result = {
        "protocol": {
            "cache_only": True,
            "models_fitted": 0,
            "labelled_rows": len(partitioned.y_development),
            "folds": FOLDS,
            "fold_seed": FOLD_SEED,
            "fold_fingerprint": partitioned.cross_validation_fingerprint,
            "comparison": "only the fixed 50:50 probability average",
            "historical_local_subset": "not reconstructed or consulted",
            "competition_predictions_generated": False,
        },
        "gate": {
            "minimum_mean_accuracy_gain": MEAN_ACCURACY_GAIN_GATE,
            "minimum_fold_wins": FOLD_WIN_GATE,
            "minimum_worst_fold_delta": WORST_FOLD_DELTA_GATE,
            "minimum_repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
            "must_pass_against_both_seeds": True,
        },
        "recipes": recipes,
        "source_sha256": source_hashes,
        "evidence_sha256": evidence_hashes,
        "output_sha256": output_hashes,
        "comparison_summary": _json_records(confirmation.summary),
        "fold_deltas": _json_records(confirmation.fold_deltas),
        "gate_failures": {
            comparison: list(failures)
            for comparison, failures in confirmation.gate_failures.items()
        },
        "passes_gate": confirmation.passes_gate_against_both,
        "stable_positive_near_miss": confirmation.stable_positive_near_miss,
        "verdict": confirmation.verdict,
        "near_miss_definition": (
            "positive mean accuracy against both seeds with every non-mean "
            "gate satisfied; only the +0.10pp mean-gain threshold fails"
        ),
        "selected_candidate": (
            "probability_average"
            if confirmation.passes_gate_against_both
            else None
        ),
        "promotion_decision": (
            "promote"
            if confirmation.passes_gate_against_both
            else "do_not_promote"
        ),
        "competition_predictions_generated": False,
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


def _load_seed_20260824(partitioned, cross_validation, source_hashes):
    result_path = NORMALISED_DIR / "result.json"
    evaluation_path = (
        NORMALISED_DIR / "raw-top-50-identities-evaluation.joblib"
    )
    result = _read_json(result_path)
    expected_recipe = _normalised_screen_recipe()
    screen_source_hashes = {
        filename: source_hashes[f"data/{filename}"]
        for filename in EXPECTED_SOURCE_SHA256
    }
    _validate_subset(
        result,
        {
            "protocol": {
                "labelled_rows": len(partitioned.y_development),
                "folds": FOLDS,
                "fold_seed": FOLD_SEED,
                "fold_fingerprint": EXPECTED_FOLD_FINGERPRINT,
                "model_seed": SEED_20260824,
                "iterations": ITERATIONS,
                "top_k": TOP_COMMON_VALUES,
                "competition_predictions_generated": False,
            },
            "source_sha256": screen_source_hashes,
            "recipes": {"raw_top_50_identities": expected_recipe},
        },
        context="seed-20260824 result",
    )
    payload = joblib.load(evaluation_path)
    if not isinstance(payload, dict) or set(payload) != {"recipe", "evaluation"}:
        raise ValueError("Seed-20260824 aggregate cache schema changed.")
    if payload["recipe"] != expected_recipe:
        raise ValueError("Seed-20260824 aggregate recipe changed.")
    evaluation = payload["evaluation"]
    if not isinstance(evaluation, CandidateEvaluation):
        raise ValueError("Seed-20260824 cache has no CandidateEvaluation.")

    probabilities = np.full((len(partitioned.y_development), len(CLASS_LABELS)), np.nan)
    hashes = {
        "result.json": _sha256(result_path),
        evaluation_path.name: _sha256(evaluation_path),
    }
    for fold, (_, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        path = NORMALISED_DIR / f"raw-top-50-identities-fold-{fold}.joblib"
        fold_payload = joblib.load(path)
        expected_metadata = {
            "cache_version": 1,
            "candidate": "raw_top_50_identities",
            "recipe": expected_recipe,
            "source_sha256": screen_source_hashes,
            "labelled_rows": len(partitioned.y_development),
            "folds": FOLDS,
            "fold_seed": FOLD_SEED,
            "fold_fingerprint": EXPECTED_FOLD_FINGERPRINT,
            "validation_fold": fold,
        }
        _validate_normalised_fold(
            fold_payload,
            expected_metadata,
            partitioned.development_ids.iloc[validation_positions].to_numpy(),
            path,
        )
        probabilities[validation_positions] = fold_payload["probabilities"]
        hashes[path.name] = _sha256(path)
    validate_probabilities(probabilities, len(partitioned.y_development))
    if not np.array_equal(
        probabilities,
        evaluation.out_of_fold_probabilities.to_numpy(),
    ):
        raise ValueError("Seed-20260824 aggregate OOF differs from fold caches.")
    rebuilt = rebuild_and_validate_evaluation(
        evaluation,
        partitioned,
        cross_validation,
        evidence_name="seed_20260824",
    )
    raw_summary = next(
        row
        for row in result["candidate_summary"]
        if row["candidate"] == "raw_top_50_identities"
    )
    _require_close(
        raw_summary["mean_accuracy"],
        rebuilt.metric_summary.loc["accuracy", "mean"],
        "seed-20260824 result accuracy",
    )
    return rebuilt, hashes


def _load_seed_20260905(partitioned, cross_validation):
    result_path = GENERALISATION_DIR / "result.json"
    predictions_path = GENERALISATION_DIR / "oof-predictions.csv"
    result = _read_json(result_path)
    _validate_subset(
        result,
        {
            "seed": SEED_20260905,
            "folds": FOLDS,
            "iterations": ITERATIONS,
            "model": "XGBoost archived depth 17 [top-50 deferred identities]",
            "fold_fingerprint": EXPECTED_GENERALISATION_FINGERPRINT,
            "selection": "none; fixed archived specification",
        },
        context="seed-20260905 result",
    )
    expected_generalisation_fingerprint = hashlib.sha256(
        pd.DataFrame(
            {
                "id": partitioned.development_ids,
                "fold": partitioned.validation_folds,
            }
        ).to_csv(index=False).encode("utf-8")
    ).hexdigest()
    if expected_generalisation_fingerprint != EXPECTED_GENERALISATION_FINGERPRINT:
        raise ValueError("Generalisation fold fingerprint changed.")

    probabilities = np.full((len(partitioned.y_development), len(CLASS_LABELS)), np.nan)
    diagnostics = []
    hashes = {"result.json": _sha256(result_path)}
    for fold, (_, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        path = GENERALISATION_DIR / f"fold-{fold}.joblib"
        payload = joblib.load(path)
        expected_ids = partitioned.development_ids.iloc[
            validation_positions
        ].to_numpy()
        _validate_generalisation_fold(payload, expected_ids, path)
        probabilities[validation_positions] = payload["probabilities"]
        diagnostics.append(
            {
                "validation_fold": fold,
                "selected_iterations": ITERATIONS,
                "fit_seconds": float(payload["fit_seconds"]),
            }
        )
        hashes[path.name] = _sha256(path)
    validate_probabilities(probabilities, len(partitioned.y_development))
    predictions = pd.read_csv(predictions_path)
    _validate_generalisation_predictions(predictions, probabilities, partitioned)
    hashes[predictions_path.name] = _sha256(predictions_path)
    evaluation = build_candidate_evaluation(
        model_name=(
            "XGBoost archived depth 17 "
            "[raw top-50 deferred identities; seed 20260905]"
        ),
        partitioned_data=partitioned,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=diagnostics,
    )
    all_labelled = next(
        row for row in result["accuracy"] if row["group"] == "all_labelled"
    )
    _require_close(
        all_labelled["accuracy"],
        evaluation.metric_summary.loc["accuracy", "mean"],
        "seed-20260905 result accuracy",
    )
    return evaluation, hashes


def _validate_normalised_fold(payload, metadata, expected_ids, path):
    if not isinstance(payload, dict) or set(payload) != {
        "metadata",
        "validation_ids",
        "probabilities",
        "diagnostics",
    }:
        raise ValueError(f"Malformed seed-20260824 fold cache: {path}.")
    if payload["metadata"] != metadata:
        raise ValueError(f"Stale seed-20260824 fold metadata: {path}.")
    if not np.array_equal(payload["validation_ids"], expected_ids):
        raise ValueError(f"Misaligned seed-20260824 validation IDs: {path}.")
    validate_probabilities(payload["probabilities"], len(expected_ids))
    diagnostics = payload["diagnostics"]
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("validation_fold") != metadata["validation_fold"]
        or diagnostics.get("selected_iterations") != ITERATIONS
        or diagnostics.get("inner_fit_rows") != 47_520
        or diagnostics.get("inner_stop_rows") != 0
    ):
        raise ValueError(f"Invalid seed-20260824 diagnostics: {path}.")


def _validate_generalisation_fold(payload, expected_ids, path):
    if not isinstance(payload, dict) or set(payload) != {
        "seed",
        "iterations",
        "fold_fingerprint",
        "validation_ids",
        "probabilities",
        "fit_seconds",
    }:
        raise ValueError(f"Malformed seed-20260905 fold cache: {path}.")
    if (
        payload["seed"] != SEED_20260905
        or payload["iterations"] != ITERATIONS
        or payload["fold_fingerprint"] != EXPECTED_GENERALISATION_FINGERPRINT
    ):
        raise ValueError(f"Stale seed-20260905 fold metadata: {path}.")
    if not np.array_equal(payload["validation_ids"], expected_ids):
        raise ValueError(f"Misaligned seed-20260905 validation IDs: {path}.")
    validate_probabilities(payload["probabilities"], len(expected_ids))
    if (
        not isinstance(payload["fit_seconds"], (int, float))
        or not np.isfinite(payload["fit_seconds"])
        or payload["fit_seconds"] <= 0
    ):
        raise ValueError(f"Invalid seed-20260905 timing: {path}.")


def _validate_generalisation_predictions(predictions, probabilities, partitioned):
    probability_columns = [f"probability_{label}" for label in CLASS_LABELS]
    expected_columns = [
        "id",
        "fold",
        "historical_local",
        "actual",
        "predicted",
        *probability_columns,
        "correct",
    ]
    if list(predictions.columns) != expected_columns:
        raise ValueError("Seed-20260905 OOF CSV schema changed.")
    if not predictions["id"].equals(partitioned.development_ids):
        raise ValueError("Seed-20260905 OOF CSV IDs are misaligned.")
    if not predictions["fold"].equals(partitioned.validation_folds.astype("int64")):
        raise ValueError("Seed-20260905 OOF CSV folds are misaligned.")
    if not predictions["actual"].equals(partitioned.y_development):
        raise ValueError("Seed-20260905 OOF CSV labels are misaligned.")
    if not np.allclose(
        predictions[probability_columns].to_numpy(),
        probabilities,
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError("Seed-20260905 OOF CSV probabilities changed.")
    expected_predictions = np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]
    if not np.array_equal(predictions["predicted"].to_numpy(), expected_predictions):
        raise ValueError("Seed-20260905 OOF CSV predictions changed.")
    expected_correct = expected_predictions == partitioned.y_development.to_numpy()
    if not np.array_equal(predictions["correct"].to_numpy(), expected_correct):
        raise ValueError("Seed-20260905 OOF CSV correctness flags changed.")
    if predictions["historical_local"].dtype != bool:
        raise ValueError("Seed-20260905 subgroup marker is not Boolean.")


def _validate_output_cache(payload, metadata, partitioned, cross_validation):
    if not isinstance(payload, dict) or set(payload) != {
        "cache_version",
        "metadata",
        "evaluation",
    }:
        raise ValueError("Two-seed output cache schema changed.")
    if payload["cache_version"] != CACHE_VERSION or payload["metadata"] != metadata:
        raise ValueError("Two-seed output cache metadata changed.")
    evaluation = payload["evaluation"]
    if not isinstance(evaluation, CandidateEvaluation):
        raise ValueError("Two-seed output cache has no CandidateEvaluation.")
    rebuild_and_validate_evaluation(
        evaluation,
        partitioned,
        cross_validation,
        evidence_name="probability_average",
    )


def _canonical_recipe(seed):
    return {
        "family": "XGBoost GPU",
        "variant": "archived depth 17",
        "variant_parameters": dict(XGBOOST_VARIANTS["archived depth 17"]),
        "model_seed": seed,
        "iterations": ITERATIONS,
        "preprocessor": (
            "top_common_identity_evaluation."
            "make_top_common_identity_preprocessor"
        ),
        "identity_fields": list(DEFERRED_HIGH_CARDINALITY_FEATURES),
        "top_k": TOP_COMMON_VALUES,
        "normalise": False,
        "class_order": list(CLASS_LABELS),
    }


def _normalised_screen_recipe():
    spec = replace(
        make_xgboost_spec(variant="archived depth 17", seed=SEED_20260824),
        name=(
            "XGBoost archived depth 17 "
            "[raw top-50 deferred identities; seed 20260824]"
        ),
        feature_policy="accepted_plus_raw_top_50_deferred_identities",
    )
    return {
        "spec": asdict(spec),
        "variant_parameters": dict(XGBOOST_VARIANTS[spec.variant]),
        "iterations": ITERATIONS,
        "identity_fields": list(DEFERRED_HIGH_CARDINALITY_FEATURES),
        "top_k": TOP_COMMON_VALUES,
        "normalise": False,
        "normaliser": "raw string values with missing sentinel",
    }


def _load_labelled_modelling_data():
    raw = pd.read_csv(DATA_DIR / "TrainingSetValues.csv")
    labels = pd.read_csv(DATA_DIR / "TrainingSetLabels.csv")
    validate_aligned_ids(raw, labels)
    if list(labels.columns) != ["id", "status_group"]:
        raise ValueError("Training label schema changed.")
    if set(labels["status_group"]) != set(CLASS_LABELS):
        raise ValueError("Training target classes changed.")
    prepared = remove_known_redundant_columns(raw)
    identifiers = prepared.pop("id").reset_index(drop=True)
    predictors = prepared.reset_index(drop=True)
    target = labels["status_group"].reset_index(drop=True)
    if predictors.shape != (59_400, 36) or target.shape != (59_400,):
        raise ValueError("Labelled modelling shape changed.")
    return ModellingData(
        original_ids=identifiers,
        X_original=predictors,
        y_original=target,
        competition_ids=identifiers.iloc[0:0].copy(),
        X_competition=predictors.iloc[0:0].copy(),
    )


def _validate_locked_sources():
    hashes = {}
    for filename, expected in EXPECTED_SOURCE_SHA256.items():
        path = DATA_DIR / filename
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Source data hash changed for {filename}.")
        hashes[f"data/{filename}"] = actual
    for filename, expected in EXPECTED_PRODUCER_SHA256.items():
        path = SCRIPT_DIR / filename
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Evidence producer hash changed for {filename}.")
        hashes[f"scripts/{filename}"] = actual
    return hashes


def _validate_subset(actual, expected, *, context):
    if not isinstance(actual, dict):
        raise ValueError(f"{context} is not a JSON object.")
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if isinstance(expected_value, dict):
            _validate_subset(
                actual_value,
                expected_value,
                context=f"{context}.{key}",
            )
        elif actual_value != expected_value:
            raise ValueError(f"{context}.{key} changed.")


def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}.")
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_close(actual, expected, context):
    if not np.isclose(actual, expected, rtol=0.0, atol=1e-12):
        raise ValueError(f"{context} changed.")


def _json_records(frame):
    serialisable = (
        frame
        if isinstance(frame.index, pd.RangeIndex)
        else frame.reset_index()
    )
    return json.loads(
        serialisable.to_json(orient="records", double_precision=15)
    )


if __name__ == "__main__":
    main()

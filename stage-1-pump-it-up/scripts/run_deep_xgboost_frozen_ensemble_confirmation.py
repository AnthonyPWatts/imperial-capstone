"""Confirm the two-seed deep bag inside the exact old-fold archive ensemble."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = (
    PROJECT_DIR
    / ".runtime/deep-xgboost-frozen-ensemble-confirmation-20260905"
)
DEVELOPMENT_LABELS_PATH = (
    PROJECT_DIR
    / ".runtime/class-membership-analysis/accepted-development-memberships.csv"
)
DEVELOPMENT_FINGERPRINT = (
    "c8a9e27264e4932140ffe9f19c897229230f9da00126f7daf200b4e37e79f6e6"
)
CROSS_VALIDATION_FINGERPRINT = (
    "bd7da743e9b4498888fa9a7b4166f64266a5f86c5175b7a9805bf3031b554469"
)
SOURCE_SHA256 = {
    "data/TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    ".runtime/class-membership-analysis/accepted-development-memberships.csv": (
        "5e25cac99c1443533f880205c8ae7a14c3d2dc28b78e94cea5630db41357d366"
    ),
    "src/deep_xgboost_frozen_ensemble_confirmation.py": (
        "03557b289fd093041520ffc87a173bad42cc0cac22f1c0f4703ec7c96dbb1e25"
    ),
    "src/deep_xgboost_variance_hedge.py": (
        "4e2be6d9264cd56aef328c2e16188c06c7bfbc3453b8704da8217496e85460b1"
    ),
    "src/gpu_model_evaluation.py": (
        "e6867cac83f1acd5437a34288748578662cb106d219ac00c3cb2f76c7fd884b5"
    ),
    "src/top_common_identity_evaluation.py": (
        "cc2719cbebd9e776c3e410a10b5eaa017e19f1fb702d4303c2498fdc48904c85"
    ),
    "src/deep_archive_confirmation.py": (
        "aa156cd3b69c8be2d86608244355ed6c99b9d1cf4d0819417160a35bbd636b3d"
    ),
    "src/locked_screen_evidence.py": (
        "50b57a22c9ccf533c4b580fcc01847fe3a750978e530430b0d2d3cd7c9ace55e"
    ),
    "src/model_evaluation.py": (
        "2c8d668654bfb9a6bbe04b17e1fec1f9d9ce16af647d891b7807d96c77988c20"
    ),
    "src/data_preparation.py": (
        "2aeda2f9229275ddf9bb207422d4ae4aacf3c05c46e16321147d1fa5c6e496f6"
    ),
    "src/data_partitioning.py": (
        "ccf43b4bca8cd542ec100783f6aa7bbb23dc15946ec29880604bed97d1cd3de3"
    ),
}
COMPONENT_PATHS = {
    "baseline": (
        PROJECT_DIR / ".runtime/geography-screen/frozen-baseline.joblib",
        "c6fe80cef3fd8153512e03fc7663b00d310cf7ad2b12b48e53c4dd17af07add8",
    ),
    "deep_xgboost": (
        PROJECT_DIR
        / ".runtime/archived-deep-xgboost-screen/"
        "depth-17-seed-20260824.joblib",
        "d8c1829424e558a6628b1224202ca9b8c548ad1780a00e3d36da2ed15ea8e924",
    ),
    "frequency_random_forest": (
        PROJECT_DIR
        / ".runtime/categorical-frequency-forest-screen/"
        "all-categorical-occurrence-counts.joblib",
        "42ccea3a7a39103213a1b5ef77e8a07d191c98cd1ba1fbdf15a5da68cd539af6",
    ),
    "spatial": (
        PROJECT_DIR
        / ".runtime/spatial-height-imputation-screen/ten-neighbour-height.joblib",
        "51fc0ade338c50f05ca281d6bc872eb22f04c5191fda7e7d466ece2336859a29",
    ),
    "identity_catboost": (
        PROJECT_DIR
        / ".runtime/catboost-identity-screen/complete-deferred-identities.joblib",
        "69ebe1c08fe6de95007734dcc46c5014adbfa18b43c38db30a1a8719c24230da",
    ),
}
SEED_20260824 = 20260824
SEED_20260905 = 20260905
ITERATIONS = 600

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import CROSS_VALIDATION_FOLDS
from data_partitioning import CROSS_VALIDATION_SEED
from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from data_preparation import remove_known_redundant_columns
from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from deep_xgboost_frozen_ensemble_confirmation import (
    compare_deep_seed_average_ensemble,
)
from deep_xgboost_frozen_ensemble_confirmation import (
    CONFIRMATION_CACHE_VERSION,
)
from deep_xgboost_frozen_ensemble_confirmation import ENSEMBLE_FOLD_WIN_GATE
from deep_xgboost_frozen_ensemble_confirmation import (
    ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
)
from deep_xgboost_frozen_ensemble_confirmation import (
    ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
)
from deep_xgboost_frozen_ensemble_confirmation import (
    ENSEMBLE_WORST_FOLD_DELTA_GATE,
)
from deep_xgboost_frozen_ensemble_confirmation import (
    validate_deep_average_component_cache,
)
from deep_xgboost_frozen_ensemble_confirmation import validate_seed_evaluation_cache
from deep_xgboost_frozen_ensemble_confirmation import validate_seed_fold_cache
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import XGBOOST_VARIANTS
from locked_screen_evidence import recompute_and_validate_oof_evaluation
from model_evaluation import build_candidate_evaluation
from top_common_identity_evaluation import make_top_common_identity_preprocessor
from top_common_identity_evaluation import TOP_COMMON_VALUES


def main() -> None:
    """Fit only missing seed-20260905 folds and write exact-ensemble evidence."""

    source_hashes = _validate_sources()
    partitioned = _load_frozen_development_partition()
    incumbent_components, seed_20260824, component_hashes = (
        _load_incumbent_components(partitioned)
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    recipe = _recipe_signature(SEED_20260905)
    seed_20260905, fold_records, aggregate_record = _load_or_fit_seed_20260905(
        partitioned,
        recipe,
        source_hashes,
        component_hashes,
    )
    comparison = compare_deep_seed_average_ensemble(
        partitioned,
        incumbent_components,
        seed_20260824,
        seed_20260905,
    )
    deep_average_record = _load_or_write_deep_average_component(
        comparison.deep_seed_average,
        partitioned,
        source_hashes,
        component_hashes,
        fold_records,
        aggregate_record,
    )
    summary_path = OUTPUT_DIR / "candidate-summary.csv"
    folds_path = OUTPUT_DIR / "fold-paired-deltas.csv"
    _write_csv_without_overwrite(summary_path, comparison.summary)
    _write_csv_without_overwrite(
        folds_path,
        comparison.fold_deltas,
        index=False,
    )
    row = comparison.summary.iloc[0]
    passes = bool(row["passes_gate"])
    result = {
        "screen": "deep-XGBoost seed average exact frozen-ensemble confirmation",
        "status": (
            "passes_exact_ensemble_gate"
            if passes
            else "comparison_complete_candidate_failed_gate"
        ),
        "passes_gate": passes,
        "competition_refit_authorised": passes,
        "competition_generator_run": False,
        "candidate": "deep_seed_probability_average_at_0.33",
        "protocol": {
            "development_rows": len(partitioned.y_development),
            "development_fingerprint": partitioned.development_fingerprint,
            "cross_validation_folds": CROSS_VALIDATION_FOLDS,
            "cross_validation_seed": CROSS_VALIDATION_SEED,
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "deep_component_weight": DEEP_ARCHIVE_WEIGHTS["deep_xgboost"],
            "seed_weights_inside_deep_component": {
                "seed_20260824": 0.5,
                "seed_20260905": 0.5,
            },
            "models_fitted": fold_records["newly_fitted_folds"],
            "reused_seed_20260824_cache": True,
            "historical_local_test_membership_reconstructed": False,
            "historical_local_test_labels_opened": False,
            "historical_local_test_scored": False,
            "TrainingSetLabels.csv_opened": False,
            "competition_data_opened": False,
            "competition_predictions_generated": False,
        },
        "gate_thresholds": {
            "mean_accuracy_gain": ENSEMBLE_MEAN_ACCURACY_GAIN_GATE,
            "minimum_fold_wins": ENSEMBLE_FOLD_WIN_GATE,
            "worst_fold_delta": ENSEMBLE_WORST_FOLD_DELTA_GATE,
            "minimum_repair_recall_delta": (
                ENSEMBLE_REPAIR_RECALL_DELTA_GATE
            ),
        },
        "recipes": {
            "seed_20260824": _recipe_signature(SEED_20260824),
            "seed_20260905": recipe,
            "probability_average": {
                "operation": (
                    "row-wise arithmetic mean of aligned class probabilities"
                ),
                "weights": {
                    "seed_20260824": 0.5,
                    "seed_20260905": 0.5,
                },
                "selection": "single predeclared weight; no search",
            },
            "ensemble_weights": dict(DEEP_ARCHIVE_WEIGHTS),
            "only_replaced_component": "deep_xgboost",
        },
        "source_sha256": source_hashes,
        "incumbent_component_sha256": component_hashes,
        "seed_20260905_fold_caches": fold_records,
        "seed_20260905_aggregate_cache": aggregate_record,
        "deep_seed_average_component_cache": deep_average_record,
        "output_sha256": {
            summary_path.name: _sha256(summary_path),
            folds_path.name: _sha256(folds_path),
        },
        "result": _records(comparison.summary)[0],
        "fold_deltas": _records(comparison.fold_deltas),
    }
    _write_json_without_overwrite(OUTPUT_DIR / "result.json", result)
    print(_format_summary(comparison.summary), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def _load_frozen_development_partition() -> PartitionedData:
    labels = pd.read_csv(
        DEVELOPMENT_LABELS_PATH,
        usecols=["id", "actual_class"],
    )
    if len(labels) != 47_520 or labels["id"].duplicated().any():
        raise ValueError("Frozen development label rows changed.")
    if labels.isna().any().any() or set(labels["actual_class"]) != set(
        CLASS_LABELS
    ):
        raise ValueError("Frozen development labels changed.")

    raw_values = pd.read_csv(DATA_DIR / "TrainingSetValues.csv")
    if raw_values["id"].duplicated().any():
        raise ValueError("Training feature identifiers are not unique.")
    indexed = raw_values.set_index("id", drop=False)
    missing = set(labels["id"]).difference(indexed.index)
    if missing:
        raise ValueError("Frozen development IDs are absent from training values.")
    selected = indexed.loc[labels["id"]].reset_index(drop=True)
    prepared = remove_known_redundant_columns(selected)
    development_ids = prepared.pop("id").reset_index(drop=True)
    X = prepared.reset_index(drop=True)
    y = labels["actual_class"].rename("status_group").reset_index(drop=True)
    folds = _make_folds(X, y)
    development_fingerprint = _fingerprint_ids(development_ids)
    fold_fingerprint = _fingerprint_folds(development_ids, folds)
    if X.shape != (47_520, 36) or y.shape != (47_520,):
        raise ValueError("Frozen development modelling shape changed.")
    if development_fingerprint != DEVELOPMENT_FINGERPRINT:
        raise ValueError("Frozen development membership changed.")
    if fold_fingerprint != CROSS_VALIDATION_FINGERPRINT:
        raise ValueError("Frozen development folds changed.")
    return PartitionedData(
        development_ids=development_ids,
        X_development=X,
        y_development=y,
        local_test_ids=development_ids.iloc[0:0].copy(),
        X_local_test=X.iloc[0:0].copy(),
        y_local_test=y.iloc[0:0].copy(),
        validation_folds=folds,
        development_fingerprint=development_fingerprint,
        local_test_fingerprint="not-opened-for-deep-ensemble-confirmation",
        cross_validation_fingerprint=fold_fingerprint,
    )


def _load_incumbent_components(partitioned: PartitionedData):
    loaded = {}
    hashes = {}
    for name, (path, expected_hash) in COMPONENT_PATHS.items():
        _require_hash(path, expected_hash)
        loaded[name] = joblib.load(path)
        hashes[name] = expected_hash
    seed_20260824 = loaded["deep_xgboost"]
    if (
        seed_20260824.model_name
        != "XGBoost archived depth 17 [top-50 deferred identities; seed 20260824]"
    ):
        raise ValueError("Pinned seed-20260824 model identity changed.")
    if not seed_20260824.diagnostics["selected_iterations"].eq(ITERATIONS).all():
        raise ValueError("Pinned seed-20260824 iterations changed.")
    evaluations = {
        "deep_xgboost": seed_20260824,
        "spatial_xgboost": loaded["spatial"].xgboost,
        "random_forest": loaded["baseline"].random_forest,
        "frequency_random_forest": loaded[
            "frequency_random_forest"
        ].random_forest,
        "spatial_random_forest": loaded["spatial"].random_forest,
        "identity_catboost": loaded["identity_catboost"].evaluation,
    }
    cross_validation = make_cross_validation(partitioned)
    replayed = {
        name: recompute_and_validate_oof_evaluation(
            evaluation,
            partitioned,
            cross_validation,
            candidate=f"incumbent {name}",
        )
        for name, evaluation in evaluations.items()
    }
    components = {
        name: evaluation.out_of_fold_probabilities.to_numpy(dtype="float64")
        for name, evaluation in replayed.items()
    }
    return components, replayed["deep_xgboost"], hashes


def _load_or_fit_seed_20260905(
    partitioned: PartitionedData,
    recipe: dict[str, object],
    source_hashes: dict[str, str],
    component_hashes: dict[str, str],
):
    probabilities = np.full(
        (len(partitioned.y_development), len(CLASS_LABELS)),
        np.nan,
    )
    diagnostics = []
    cache_hashes = {}
    fitted_folds = 0
    spec = replace(
        make_xgboost_spec(variant="archived depth 17", seed=SEED_20260905),
        name=(
            "XGBoost archived depth 17 "
            "[top-50 deferred identities; seed 20260905]"
        ),
        feature_policy="accepted_plus_top_50_deferred_identities",
    )
    cross_validation = make_cross_validation(partitioned)
    for fold, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        metadata = {
            "cache_version": CONFIRMATION_CACHE_VERSION,
            "candidate": "deep_xgboost_seed_20260905",
            "recipe": recipe,
            "development_rows": 47_520,
            "outer_training_rows": len(training_positions),
            "validation_rows": len(validation_positions),
            "development_fingerprint": DEVELOPMENT_FINGERPRINT,
            "cross_validation_folds": CROSS_VALIDATION_FOLDS,
            "cross_validation_seed": CROSS_VALIDATION_SEED,
            "cross_validation_fingerprint": CROSS_VALIDATION_FINGERPRINT,
            "validation_fold": fold,
            "source_sha256": source_hashes,
            "incumbent_component_sha256": component_hashes,
            "historical_local_test_labels_opened": False,
            "competition_data_opened": False,
        }
        path = OUTPUT_DIR / f"seed-20260905-fold-{fold}.joblib"
        expected_ids = partitioned.development_ids.iloc[
            validation_positions
        ].to_numpy()
        if path.exists():
            payload = joblib.load(path)
            print(f"Loaded {path}.", flush=True)
        else:
            fold_probabilities, seconds = fit_gpu_candidate_probabilities(
                spec,
                partitioned.X_development.iloc[training_positions],
                partitioned.y_development.iloc[training_positions],
                partitioned.X_development.iloc[validation_positions],
                iterations=ITERATIONS,
                preprocessor_factory=make_top_common_identity_preprocessor,
            )
            payload = {
                "cache_version": CONFIRMATION_CACHE_VERSION,
                "metadata": metadata,
                "validation_ids": expected_ids.copy(),
                "probabilities": fold_probabilities,
                "fit_and_predict_seconds": seconds,
            }
            _dump_new_cache(path, payload)
            fitted_folds += 1
            print(
                f"Completed seed-20260905 fold {fold}/{CROSS_VALIDATION_FOLDS} "
                f"in {seconds:.1f} seconds.",
                flush=True,
            )
        payload = validate_seed_fold_cache(payload, metadata, expected_ids)
        probabilities[validation_positions] = payload["probabilities"]
        diagnostics.append(
            {
                "validation_fold": fold,
                "selected_iterations": ITERATIONS,
                "inner_fit_rows": len(training_positions),
                "inner_stop_rows": 0,
                "stopping_seconds": 0.0,
                "refit_predict_seconds": float(
                    payload["fit_and_predict_seconds"]
                ),
                "total_seconds": float(payload["fit_and_predict_seconds"]),
            }
        )
        cache_hashes[path.name] = _sha256(path)
    validate_probabilities(probabilities, len(partitioned.y_development))
    evaluation = build_candidate_evaluation(
        model_name=spec.name,
        partitioned_data=partitioned,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=diagnostics,
    )
    aggregate_metadata = {
        "cache_version": CONFIRMATION_CACHE_VERSION,
        "candidate": "deep_xgboost_seed_20260905",
        "recipe": recipe,
        "development_rows": 47_520,
        "development_fingerprint": DEVELOPMENT_FINGERPRINT,
        "cross_validation_folds": CROSS_VALIDATION_FOLDS,
        "cross_validation_seed": CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": CROSS_VALIDATION_FINGERPRINT,
        "source_sha256": source_hashes,
        "incumbent_component_sha256": component_hashes,
        "fold_cache_sha256": cache_hashes,
        "historical_local_test_labels_opened": False,
        "competition_data_opened": False,
    }
    aggregate_path = OUTPUT_DIR / "seed-20260905-evaluation.joblib"
    if aggregate_path.exists():
        aggregate_payload = joblib.load(aggregate_path)
    else:
        aggregate_payload = {
            "cache_version": CONFIRMATION_CACHE_VERSION,
            "metadata": aggregate_metadata,
            "evaluation": evaluation,
        }
        _dump_new_cache(aggregate_path, aggregate_payload)
    evaluation = validate_seed_evaluation_cache(
        aggregate_payload,
        aggregate_metadata,
        partitioned,
    )
    if not np.array_equal(
        evaluation.out_of_fold_probabilities.to_numpy(),
        probabilities,
    ):
        raise ValueError("Seed-20260905 aggregate differs from its fold caches.")
    records = {
        "newly_fitted_folds": fitted_folds,
        "reused_folds": CROSS_VALIDATION_FOLDS - fitted_folds,
        "sha256": cache_hashes,
    }
    aggregate_record = {
        "file": aggregate_path.name,
        "sha256": _sha256(aggregate_path),
        "metadata": aggregate_metadata,
    }
    return evaluation, records, aggregate_record


def _load_or_write_deep_average_component(
    evaluation,
    partitioned: PartitionedData,
    source_hashes: dict[str, str],
    component_hashes: dict[str, str],
    fold_records: dict[str, object],
    aggregate_record: dict[str, object],
) -> dict[str, object]:
    metadata = {
        "cache_version": CONFIRMATION_CACHE_VERSION,
        "candidate": "deep_seed_probability_average_component",
        "probability_average": {
            "seed_20260824": 0.5,
            "seed_20260905": 0.5,
        },
        "recipes": {
            "seed_20260824": _recipe_signature(SEED_20260824),
            "seed_20260905": _recipe_signature(SEED_20260905),
        },
        "development_rows": 47_520,
        "development_fingerprint": DEVELOPMENT_FINGERPRINT,
        "cross_validation_folds": CROSS_VALIDATION_FOLDS,
        "cross_validation_seed": CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": CROSS_VALIDATION_FINGERPRINT,
        "source_sha256": source_hashes,
        "input_sha256": {
            "seed_20260824": component_hashes["deep_xgboost"],
            "seed_20260905_aggregate": aggregate_record["sha256"],
            "seed_20260905_folds": fold_records["sha256"],
        },
        "historical_local_test_labels_opened": False,
        "competition_data_opened": False,
    }
    path = OUTPUT_DIR / "deep-seed-average-component-evaluation.joblib"
    if path.exists():
        payload = joblib.load(path)
    else:
        payload = {
            "cache_version": CONFIRMATION_CACHE_VERSION,
            "metadata": metadata,
            "evaluation": evaluation,
        }
        _dump_new_cache(path, payload)
    replayed = validate_deep_average_component_cache(
        payload,
        metadata,
        partitioned,
    )
    if not np.array_equal(
        replayed.out_of_fold_probabilities.to_numpy(),
        evaluation.out_of_fold_probabilities.to_numpy(),
    ):
        raise ValueError("Deep seed-average component cache changed.")
    return {
        "file": path.name,
        "sha256": _sha256(path),
        "metadata": metadata,
    }


def _recipe_signature(seed: int) -> dict[str, object]:
    if seed not in {SEED_20260824, SEED_20260905}:
        raise ValueError("Deep confirmation seed is outside the locked pair.")
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
        "fit_scope": (
            "each old frozen outer fold trains on 38,016 development rows"
        ),
    }


def _validate_sources() -> dict[str, str]:
    paths = {
        "data/TrainingSetValues.csv": DATA_DIR / "TrainingSetValues.csv",
        (
            ".runtime/class-membership-analysis/"
            "accepted-development-memberships.csv"
        ): DEVELOPMENT_LABELS_PATH,
        **{
            relative: STAGE_DIR / relative
            for relative in SOURCE_SHA256
            if relative.startswith("src/")
        },
    }
    hashes = {}
    if set(paths) != set(SOURCE_SHA256):
        raise ValueError("Frozen confirmation source map changed.")
    for name, path in paths.items():
        _require_hash(path, SOURCE_SHA256[name])
        hashes[name] = SOURCE_SHA256[name]
    return hashes


def _make_folds(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    splitter = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=CROSS_VALIDATION_SEED,
    )
    assignments = np.zeros(len(y), dtype=np.int8)
    for fold, (_, positions) in enumerate(splitter.split(X, y), start=1):
        assignments[positions] = fold
    return pd.Series(assignments, name="validation_fold")


def _write_csv_without_overwrite(
    path: Path,
    frame: pd.DataFrame,
    *,
    index: bool = True,
) -> None:
    expected = frame.to_csv(index=index)
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(f"Refusing to overwrite different CSV: {path}.")
        return
    path.write_text(expected, encoding="utf-8")


def _write_json_without_overwrite(path: Path, value: dict[str, object]) -> None:
    expected = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(f"Refusing to overwrite different JSON: {path}.")
        return
    path.write_text(expected, encoding="utf-8")


def _dump_new_cache(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {path}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary cache exists: {temporary}.")
    joblib.dump(payload, temporary, compress=3)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {path}.")
    temporary.rename(path)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    serialisable = (
        frame
        if isinstance(frame.index, pd.RangeIndex)
        else frame.reset_index()
    )
    return json.loads(
        serialisable.to_json(orient="records", double_precision=15)
    )


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in display.columns:
        if "accuracy" in column or "recall" in column or "delta" in column:
            display[column] = display[column].map(
                lambda value: f"{value:.4%}" if pd.notna(value) else ""
            )
    return display.to_string()


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 changed for {path}: {actual}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_ids(ids: pd.Series) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(value) for value in ids):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint_folds(ids: pd.Series, folds: pd.Series) -> str:
    digest = hashlib.sha256()
    memberships = sorted(
        zip((str(value) for value in ids), folds, strict=True),
        key=lambda item: item[0],
    )
    for identifier, fold in memberships:
        digest.update(f"{identifier}:{fold}\n".encode("utf-8"))
    return digest.hexdigest()


if __name__ == "__main__":
    main()

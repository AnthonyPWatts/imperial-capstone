"""Confirm two locked RF substitutions inside the exact deep archive."""

from __future__ import annotations

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
RF_SCREEN_DIR = (
    PROJECT_DIR / ".runtime" / "fresh-rf-family-screen-seed-20260905"
)
OUTPUT_DIR = (
    PROJECT_DIR / ".runtime" / "locked-rf-ensemble-confirmation-20260905"
)
DEVELOPMENT_LABELS_PATH = (
    PROJECT_DIR
    / ".runtime"
    / "class-membership-analysis"
    / "accepted-development-memberships.csv"
)
DEVELOPMENT_FINGERPRINT = (
    "c8a9e27264e4932140ffe9f19c897229230f9da00126f7daf200b4e37e79f6e6"
)
CROSS_VALIDATION_FINGERPRINT = (
    "bd7da743e9b4498888fa9a7b4166f64266a5f86c5175b7a9805bf3031b554469"
)
SOURCE_HASHES = {
    "TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    "accepted-development-memberships.csv": (
        "5e25cac99c1443533f880205c8ae7a14c3d2dc28b78e94cea5630db41357d366"
    ),
}
COMPONENT_PATHS = {
    "baseline": (
        PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib",
        "c6fe80cef3fd8153512e03fc7663b00d310cf7ad2b12b48e53c4dd17af07add8",
    ),
    "deep_xgboost": (
        PROJECT_DIR
        / ".runtime"
        / "archived-deep-xgboost-screen"
        / "depth-17-seed-20260824.joblib",
        "d8c1829424e558a6628b1224202ca9b8c548ad1780a00e3d36da2ed15ea8e924",
    ),
    "frequency_random_forest": (
        PROJECT_DIR
        / ".runtime"
        / "categorical-frequency-forest-screen"
        / "all-categorical-occurrence-counts.joblib",
        "42ccea3a7a39103213a1b5ef77e8a07d191c98cd1ba1fbdf15a5da68cd539af6",
    ),
    "spatial": (
        PROJECT_DIR
        / ".runtime"
        / "spatial-height-imputation-screen"
        / "ten-neighbour-height.joblib",
        "51fc0ade338c50f05ca281d6bc872eb22f04c5191fda7e7d466ece2336859a29",
    ),
    "identity_catboost": (
        PROJECT_DIR
        / ".runtime"
        / "catboost-identity-screen"
        / "complete-deferred-identities.joblib",
        "69ebe1c08fe6de95007734dcc46c5014adbfa18b43c38db30a1a8719c24230da",
    ),
}
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import CROSS_VALIDATION_FOLDS
from data_partitioning import CROSS_VALIDATION_SEED
from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from data_preparation import remove_known_redundant_columns
from deep_archive_confirmation import DEEP_ARCHIVE_WEIGHTS
from final_model import CLASS_LABELS
from gpu_model_evaluation import evaluate_gpu_candidate
from locked_architecture_candidates import sha256_file
from locked_architecture_candidates import write_json_without_overwrite
from locked_rf_ensemble_confirmation import ACCURACY_FIRST_RF_CHALLENGER
from locked_rf_ensemble_confirmation import CONFIRMATION_CACHE_VERSION
from locked_rf_ensemble_confirmation import CONFIRMATION_CANDIDATES
from locked_rf_ensemble_confirmation import ENSEMBLE_FOLD_WIN_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_MEAN_ACCURACY_GAIN_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_REPAIR_RECALL_DELTA_GATE
from locked_rf_ensemble_confirmation import ENSEMBLE_WORST_FOLD_DELTA_GATE
from locked_rf_ensemble_confirmation import FORMAL_RF_CHALLENGER
from locked_rf_ensemble_confirmation import (
    compare_locked_rf_ensemble_substitutions,
)
from locked_rf_ensemble_confirmation import load_locked_rf_screen_candidates
from locked_rf_ensemble_confirmation import rf_recipe_signature
from locked_rf_ensemble_confirmation import validate_confirmation_cache_payload
from locked_screen_evidence import recompute_and_validate_oof_evaluation


def main() -> None:
    screen = load_locked_rf_screen_candidates(RF_SCREEN_DIR)
    partitioned = _load_frozen_development_partition()
    components, component_hashes = _load_incumbent_components(partitioned)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    replacements = {}
    cache_records = {}
    for candidate in CONFIRMATION_CANDIDATES:
        metadata = _cache_metadata(
            candidate,
            screen.specs[candidate],
            screen.hashes,
            component_hashes,
        )
        cache_path = OUTPUT_DIR / f"{_slug(candidate)}.joblib"
        evaluation = _load_or_evaluate(
            cache_path,
            metadata,
            partitioned,
            candidate,
            screen.specs[candidate],
        )
        replacements[candidate] = (
            evaluation.out_of_fold_probabilities.to_numpy(dtype="float64")
        )
        cache_records[candidate] = {
            "file": cache_path.name,
            "sha256": sha256_file(cache_path),
            "metadata": metadata,
        }

    comparison = compare_locked_rf_ensemble_substitutions(
        partitioned,
        components,
        replacements,
    )
    summary = comparison.summary.copy()
    summary.insert(
        0,
        "fresh_screen_role",
        ["formal balanced gate winner", "accuracy-first guardrail exception"],
    )
    screen_summary = pd.read_csv(
        RF_SCREEN_DIR / "candidate-summary.csv",
        index_col="candidate",
    )
    for column in (
        "mean_accuracy_delta",
        "fold_wins_vs_incumbent",
        "worst_fold_delta",
        "repair_recall_delta",
        "passes_gate",
    ):
        summary[f"fresh_component_{column}"] = screen_summary.loc[
            summary.index,
            column,
        ]
    passing_candidates = summary.index[summary["passes_gate"]].tolist()
    _write_csv_without_overwrite(OUTPUT_DIR / "candidate-summary.csv", summary)
    _write_csv_without_overwrite(
        OUTPUT_DIR / "fold-paired-deltas.csv",
        comparison.fold_deltas,
        index=False,
    )

    result = {
        "screen": "locked RF exact-ensemble secondary confirmation",
        "status": (
            "passing_candidates_require_review"
            if passing_candidates
            else "comparison_complete_no_candidate_passed"
        ),
        "selection_deferred": bool(passing_candidates),
        "competition_refit_authorised": bool(passing_candidates),
        "passing_candidates": passing_candidates,
        "candidates": list(CONFIRMATION_CANDIDATES),
        "candidate_roles": {
            FORMAL_RF_CHALLENGER: "formal balanced gate winner",
            ACCURACY_FIRST_RF_CHALLENGER: (
                "accuracy-first guardrail exception; failed the fresh "
                "standalone repair-recall gate"
            ),
        },
        "protocol": {
            "development_rows": len(partitioned.y_development),
            "development_fingerprint": partitioned.development_fingerprint,
            "cross_validation_folds": CROSS_VALIDATION_FOLDS,
            "cross_validation_seed": CROSS_VALIDATION_SEED,
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "random_forest_weight": DEEP_ARCHIVE_WEIGHTS["random_forest"],
            "historical_local_test_labels_opened": False,
            "historical_local_test_scored": False,
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
        "source_sha256": SOURCE_HASHES,
        "fresh_screen_sha256": screen.hashes,
        "incumbent_component_sha256": component_hashes,
        "candidate_caches": cache_records,
        "incumbent_mean_accuracy": float(
            comparison.incumbent_evaluation.metric_summary.loc[
                "accuracy", "mean"
            ]
        ),
        "results": _records(summary),
    }
    write_json_without_overwrite(OUTPUT_DIR / "result.json", result)
    print(_format_summary(summary), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def _load_frozen_development_partition() -> PartitionedData:
    values_path = DATA_DIR / "TrainingSetValues.csv"
    _require_hash(values_path, SOURCE_HASHES[values_path.name])
    _require_hash(
        DEVELOPMENT_LABELS_PATH,
        SOURCE_HASHES[DEVELOPMENT_LABELS_PATH.name],
    )
    labels = pd.read_csv(
        DEVELOPMENT_LABELS_PATH,
        usecols=["id", "actual_class"],
    )
    if len(labels) != 47_520 or labels["id"].duplicated().any():
        raise ValueError("Frozen development label rows changed.")
    if set(labels["actual_class"]) != set(CLASS_LABELS):
        raise ValueError("Frozen development labels changed.")

    raw_values = pd.read_csv(values_path)
    if raw_values["id"].duplicated().any():
        raise ValueError("Training feature identifiers are not unique.")
    indexed = raw_values.set_index("id", drop=False)
    missing = set(labels["id"]).difference(indexed.index)
    if missing:
        raise ValueError("Frozen development IDs are absent from training values.")
    selected = indexed.loc[labels["id"]].reset_index(drop=True)
    prepared = remove_known_redundant_columns(selected)
    development_ids = prepared.pop("id").reset_index(drop=True)
    y = labels["actual_class"].rename("status_group").reset_index(drop=True)
    X = prepared.reset_index(drop=True)
    folds = _make_folds(X, y)
    development_fingerprint = _fingerprint_ids(development_ids)
    cross_validation_fingerprint = _fingerprint_folds(
        development_ids,
        folds,
    )
    if development_fingerprint != DEVELOPMENT_FINGERPRINT:
        raise ValueError("Frozen development membership changed.")
    if cross_validation_fingerprint != CROSS_VALIDATION_FINGERPRINT:
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
        local_test_fingerprint="not-opened-for-secondary-confirmation",
        cross_validation_fingerprint=cross_validation_fingerprint,
    )


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


def _load_incumbent_components(
    partitioned: PartitionedData,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    loaded = {}
    hashes = {}
    for name, (path, expected_hash) in COMPONENT_PATHS.items():
        _require_hash(path, expected_hash)
        loaded[name] = joblib.load(path)
        hashes[name] = expected_hash
    evaluations = {
        "deep_xgboost": loaded["deep_xgboost"],
        "spatial_xgboost": loaded["spatial"].xgboost,
        "random_forest": loaded["baseline"].random_forest,
        "frequency_random_forest": loaded["frequency_random_forest"].random_forest,
        "spatial_random_forest": loaded["spatial"].random_forest,
        "identity_catboost": loaded["identity_catboost"].evaluation,
    }
    cross_validation = make_cross_validation(partitioned)
    components = {}
    for name, evaluation in evaluations.items():
        recomputed = recompute_and_validate_oof_evaluation(
            evaluation,
            partitioned,
            cross_validation,
            candidate=f"incumbent {name}",
        )
        components[name] = recomputed.out_of_fold_probabilities.to_numpy(
            dtype="float64"
        )
    return components, hashes


def _load_or_evaluate(
    path: Path,
    metadata: dict[str, object],
    partitioned: PartitionedData,
    candidate: str,
    spec,
):
    if path.exists():
        payload = joblib.load(path)
        print(f"Loaded {path}.", flush=True)
    else:
        evaluation = evaluate_gpu_candidate(
            spec,
            partitioned,
            make_cross_validation(partitioned),
        )
        payload = {
            "cache_version": CONFIRMATION_CACHE_VERSION,
            "metadata": metadata,
            "evaluation": evaluation,
        }
        joblib.dump(payload, path, compress=3)
    return validate_confirmation_cache_payload(
        payload,
        metadata,
        partitioned,
        candidate=candidate,
    )


def _cache_metadata(
    candidate,
    spec,
    screen_hashes,
    component_hashes,
) -> dict[str, object]:
    return {
        "candidate": candidate,
        "recipe": rf_recipe_signature(spec),
        "development_rows": 47_520,
        "development_fingerprint": DEVELOPMENT_FINGERPRINT,
        "cross_validation_folds": CROSS_VALIDATION_FOLDS,
        "cross_validation_seed": CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": CROSS_VALIDATION_FINGERPRINT,
        "source_sha256": SOURCE_HASHES,
        "fresh_screen_sha256": screen_hashes,
        "incumbent_component_sha256": component_hashes,
        "deep_archive_weights": DEEP_ARCHIVE_WEIGHTS,
        "historical_local_test_labels_opened": False,
        "competition_data_opened": False,
    }


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


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.reset_index().to_json(orient="records"))


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
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA-256 changed for {path}: {actual}.")


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


def _slug(value: str) -> str:
    return "-".join(value.casefold().split()).replace(".", "-")


if __name__ == "__main__":
    main()

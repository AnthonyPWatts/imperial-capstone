"""Run the locked full-labelled-data spatial-grid CatBoost comparison."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "fresh-spatial-grid-catboost"
CROSS_VALIDATION_SEED = 20260905
CROSS_VALIDATION_FOLDS = 5
CACHE_VERSION = 1
EXPECTED_CROSS_VALIDATION_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_evaluation import CatBoostIdentityTrial
from catboost_identity_evaluation import DEFERRED_IDENTITY_FEATURES
from catboost_identity_evaluation import evaluate_complete_identity_catboost
from catboost_identity_evaluation import make_complete_identity_catboost_spec
from data_partitioning import PartitionedData
from gpu_model_evaluation import median_selected_iterations
from gpu_model_evaluation import CATBOOST_VARIANTS
from locked_architecture_candidates import sha256_file
from locked_architecture_candidates import write_json_without_overwrite
from modelling_data import ModellingData
from modelling_data import prepare_modelling_data
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_ACCURACY
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_FOLD_WINS
from spatial_grid_catboost_evaluation import SPATIAL_GRID_GATE_WORST_FOLD
from spatial_grid_catboost_evaluation import evaluate_spatial_grid_catboost
from spatial_grid_catboost_evaluation import make_spatial_grid_catboost_spec
from spatial_grid_catboost_evaluation import spatial_grid_policy_signature
from spatial_grid_catboost_evaluation import (
    summarise_spatial_grid_catboost_comparison,
)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = _load_modelling_data()
    partitioned_data, cross_validation = _make_full_data_design(modelling_data)
    identity_recipe = _model_recipe(make_complete_identity_catboost_spec())
    grid_recipe = _model_recipe(make_spatial_grid_catboost_spec())
    identity_policy = _identity_policy_signature()
    grid_policy = spatial_grid_policy_signature()
    identity = _load_or_evaluate(
        OUTPUT_DIR / "complete-identity-catboost.joblib",
        candidate="complete_identity_catboost",
        expected_model_name="CatBoost d8 [complete deferred identities]",
        expected_recipe=identity_recipe,
        feature_policy_signature=identity_policy,
        evaluator=evaluate_complete_identity_catboost,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
    )
    spatial_grid = _load_or_evaluate(
        OUTPUT_DIR / "spatial-grid-catboost.joblib",
        candidate="spatial_grid_catboost",
        expected_model_name="CatBoost d8 [complete identities plus spatial grids]",
        expected_recipe=grid_recipe,
        feature_policy_signature=grid_policy,
        evaluator=evaluate_spatial_grid_catboost,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
    )

    candidate_summary, fold_deltas = (
        summarise_spatial_grid_catboost_comparison(
            identity.evaluation,
            spatial_grid.evaluation,
        )
    )
    candidate_summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    fold_deltas.to_csv(OUTPUT_DIR / "fold-deltas.csv")
    passes_gate = bool(
        candidate_summary.loc["spatial_grid_catboost", "passes_gate"]
    )
    result = {
        "protocol": {
            "labelled_rows": len(partitioned_data.y_development),
            "cross_validation_folds": CROSS_VALIDATION_FOLDS,
            "cross_validation_seed": CROSS_VALIDATION_SEED,
            "cross_validation_fingerprint": (
                partitioned_data.cross_validation_fingerprint
            ),
            "selection_basis": "paired OOF accuracy across all labelled rows",
            "historical_local_subset": "not created or consulted",
            "competition_predictions_generated": False,
        },
        "gate": {
            "minimum_accuracy_change": SPATIAL_GRID_GATE_ACCURACY,
            "minimum_fold_wins": SPATIAL_GRID_GATE_FOLD_WINS,
            "minimum_worst_fold_change": SPATIAL_GRID_GATE_WORST_FOLD,
        },
        "candidate_summary": _json_records(candidate_summary),
        "fold_deltas": _json_records(fold_deltas),
        "full_data_refit_iterations": {
            "complete_identity_catboost": median_selected_iterations(
                identity.evaluation
            ),
            "spatial_grid_catboost": median_selected_iterations(
                spatial_grid.evaluation
            ),
        },
        "model_recipes": {
            "complete_identity_catboost": identity_recipe,
            "spatial_grid_catboost": grid_recipe,
        },
        "feature_policy_signatures": {
            "complete_identity_catboost": identity_policy,
            "spatial_grid_catboost": grid_policy,
        },
        "cache_protocol_sha256": {
            "complete_identity_catboost": sha256_file(
                _protocol_path(OUTPUT_DIR / "complete-identity-catboost.joblib")
            ),
            "spatial_grid_catboost": sha256_file(
                _protocol_path(OUTPUT_DIR / "spatial-grid-catboost.joblib")
            ),
        },
        "passes_gate": passes_gate,
        "selected_candidate": (
            "spatial_grid_catboost"
            if passes_gate
            else "complete_identity_catboost"
        ),
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(_format_summary(candidate_summary), flush=True)
    print(f"\nFold deltas\n{_format_fold_deltas(fold_deltas)}", flush=True)


def _load_modelling_data() -> ModellingData:
    return prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )


def _make_full_data_design(
    modelling_data: ModellingData,
) -> tuple[PartitionedData, PredefinedSplit]:
    X = modelling_data.X_original.reset_index(drop=True)
    y = modelling_data.y_original.reset_index(drop=True)
    ids = modelling_data.original_ids.reset_index(drop=True)
    splitter = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=CROSS_VALIDATION_SEED,
    )
    assignments = np.zeros(len(y), dtype=np.int8)
    for fold_number, (_, validation_positions) in enumerate(
        splitter.split(X, y),
        start=1,
    ):
        assignments[validation_positions] = fold_number
    folds = pd.Series(assignments, name="validation_fold")
    fold_fingerprint = _fingerprint_folds(ids, folds)
    if fold_fingerprint != EXPECTED_CROSS_VALIDATION_FINGERPRINT:
        raise ValueError(
            "Fresh full-data cross-validation membership changed; "
            f"expected {EXPECTED_CROSS_VALIDATION_FINGERPRINT}, "
            f"found {fold_fingerprint}."
        )
    partitioned_data = PartitionedData(
        development_ids=ids,
        X_development=X,
        y_development=y,
        local_test_ids=ids.iloc[0:0].copy(),
        X_local_test=X.iloc[0:0].copy(),
        y_local_test=y.iloc[0:0].copy(),
        validation_folds=folds,
        development_fingerprint=_fingerprint_ids(ids),
        local_test_fingerprint="not-applicable-full-data-design",
        cross_validation_fingerprint=fold_fingerprint,
    )
    return partitioned_data, PredefinedSplit(test_fold=assignments - 1)


def _load_or_evaluate(
    path: Path,
    *,
    candidate: str,
    expected_model_name: str,
    expected_recipe: dict[str, object],
    feature_policy_signature: dict[str, object],
    evaluator: Callable[[PartitionedData, object], CatBoostIdentityTrial],
    partitioned_data: PartitionedData,
    cross_validation: PredefinedSplit,
) -> CatBoostIdentityTrial:
    expected_metadata = {
        "cache_version": CACHE_VERSION,
        "candidate": candidate,
        "labelled_rows": len(partitioned_data.y_development),
        "cross_validation_folds": CROSS_VALIDATION_FOLDS,
        "cross_validation_seed": CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": (
            partitioned_data.cross_validation_fingerprint
        ),
        "model_name": expected_model_name,
    }
    if path.exists():
        payload = joblib.load(path)
        trial = _validate_cache(
            payload,
            expected_metadata=expected_metadata,
            path=path,
        )
        print(f"Loaded {path}.", flush=True)
    else:
        trial = evaluator(partitioned_data, cross_validation)
        payload = {**expected_metadata, "trial": trial}
        _validate_cache(payload, expected_metadata=expected_metadata, path=path)
        joblib.dump(payload, path, compress=3)
        print(f"Cached {path}.", flush=True)
    _ensure_protocol_sidecar(
        path,
        expected_metadata=expected_metadata,
        model_recipe=expected_recipe,
        feature_policy_signature=feature_policy_signature,
    )
    return trial


def _validate_cache(
    payload: object,
    *,
    expected_metadata: dict[str, object],
    path: Path,
) -> CatBoostIdentityTrial:
    if not isinstance(payload, dict):
        raise ValueError(f"Cached CatBoost evaluation is malformed: {path}")
    actual_metadata = {
        key: payload.get(key) for key in expected_metadata
    }
    if actual_metadata != expected_metadata:
        raise ValueError(f"Cached CatBoost evaluation is incompatible: {path}")
    trial = payload.get("trial")
    if not isinstance(trial, CatBoostIdentityTrial):
        raise ValueError(f"Cached CatBoost trial is malformed: {path}")
    evaluation = trial.evaluation
    if (
        evaluation.cross_validation_fingerprint
        != expected_metadata["cross_validation_fingerprint"]
        or evaluation.model_name != expected_metadata["model_name"]
        or len(evaluation.fold_metrics) != CROSS_VALIDATION_FOLDS
        or len(evaluation.out_of_fold_probabilities)
        != expected_metadata["labelled_rows"]
    ):
        raise ValueError(f"Cached CatBoost trial evidence is incompatible: {path}")
    return trial


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
    for identifier, fold_number in memberships:
        digest.update(f"{identifier}:{fold_number}\n".encode("utf-8"))
    return digest.hexdigest()


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(
        frame.reset_index().to_json(orient="records", double_precision=15)
    )


def _model_recipe(spec: object) -> dict[str, object]:
    return {
        "spec": asdict(spec),
        "variant_parameters": dict(CATBOOST_VARIANTS[spec.variant]),
    }


def _identity_policy_signature() -> dict[str, object]:
    return {
        "version": 1,
        "feature_engineer": "engineer_complete_identity_catboost_features",
        "deferred_identity_features": list(DEFERRED_IDENTITY_FEATURES),
    }


def _ensure_protocol_sidecar(
    cache_path: Path,
    *,
    expected_metadata: dict[str, object],
    model_recipe: dict[str, object],
    feature_policy_signature: dict[str, object],
) -> None:
    protocol = {
        "protocol_version": 2,
        "migration": "strict v1 cache validation; no model refit",
        "cache_sha256": sha256_file(cache_path),
        "validated_v1_metadata": expected_metadata,
        "model_recipe": model_recipe,
        "feature_policy_signature": feature_policy_signature,
    }
    write_json_without_overwrite(_protocol_path(cache_path), protocol)


def _protocol_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".protocol-v2.json")


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in (
        "mean_accuracy",
        "accuracy_change_vs_identity",
        "worst_fold_change_vs_identity",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


def _format_fold_deltas(fold_deltas: pd.DataFrame) -> str:
    display = fold_deltas.copy()
    for column in (
        "complete_identity_accuracy",
        "spatial_grid_accuracy",
        "spatial_grid_change",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


if __name__ == "__main__":
    main()

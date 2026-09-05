"""Run fresh five-fold OOF diagnosis on every labelled Pump It Up row."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "generalisation-diagnosis"
FOLD_CACHE_DIR = OUTPUT_DIR / "strict-fold-cache-v1"
SEED = 20260905
FOLDS = 5
ITERATIONS = 600
SOURCE_SHA256 = {
    "TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    "TrainingSetLabels.csv": (
        "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
    ),
    "TestSetValues.csv": (
        "a222110d5606910953607efa5112eafb1d6c30a483c4cbcd0b92c8306125c9b5"
    ),
}
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import partition_modelling_data
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from generalisation_diagnosis import make_fold_cache_payload
from generalisation_diagnosis import stratified_accuracy_difference_interval
from generalisation_diagnosis import summarise_predictions
from generalisation_diagnosis import validate_fold_cache_payload
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import XGBOOST_VARIANTS
from modelling_data import prepare_modelling_data
from top_common_identity_evaluation import TOP_COMMON_VALUES
from top_common_identity_evaluation import make_top_common_identity_preprocessor


def main(*, cache_only: bool = False) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = _validate_source_hashes()
    modelling = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    historical_local_ids = set(partition_modelling_data(modelling).local_test_ids)
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    assignments = np.zeros(len(modelling.y_original), dtype=np.int8)
    splits = list(splitter.split(modelling.X_original, modelling.y_original))
    for fold, (_, validation_positions) in enumerate(splits, start=1):
        assignments[validation_positions] = fold
    fold_fingerprint = hashlib.sha256(
        pd.DataFrame({"id": modelling.original_ids, "fold": assignments})
        .to_csv(index=False)
        .encode("utf-8")
    ).hexdigest()

    spec = make_xgboost_spec(variant="archived depth 17", seed=SEED)
    model_recipe = _model_recipe(spec)
    probabilities = np.full(
        (len(modelling.y_original), len(CLASS_LABELS)),
        np.nan,
    )
    diagnostics = []
    for fold, (training_positions, validation_positions) in enumerate(splits, start=1):
        cache_path = FOLD_CACHE_DIR / f"fold-{fold}.joblib"
        expected_ids = modelling.original_ids.iloc[validation_positions].to_numpy()
        metadata = _fold_cache_metadata(
            fold=fold,
            training_rows=len(training_positions),
            validation_rows=len(validation_positions),
            labelled_rows=len(modelling.y_original),
            fold_fingerprint=fold_fingerprint,
            source_hashes=source_hashes,
            model_recipe=model_recipe,
        )
        if cache_path.exists():
            cached = joblib.load(cache_path)
            fold_probabilities, fit_seconds = validate_fold_cache_payload(
                cached,
                metadata,
                expected_ids,
                cache_name=str(cache_path),
            )
            print(f"Loaded {cache_path}.", flush=True)
        else:
            if cache_only:
                raise FileNotFoundError(
                    f"Cache-only replay requires existing fold cache: {cache_path}"
                )
            started = time.perf_counter()
            fold_probabilities, fit_seconds = fit_gpu_candidate_probabilities(
                spec,
                modelling.X_original.iloc[training_positions],
                modelling.y_original.iloc[training_positions],
                modelling.X_original.iloc[validation_positions],
                iterations=ITERATIONS,
                preprocessor_factory=make_top_common_identity_preprocessor,
            )
            cached = make_fold_cache_payload(
                metadata,
                expected_ids,
                fold_probabilities,
                fit_seconds,
            )
            joblib.dump(cached, cache_path, compress=3)
            fold_probabilities, fit_seconds = validate_fold_cache_payload(
                joblib.load(cache_path),
                metadata,
                expected_ids,
                cache_name=str(cache_path),
            )
            print(
                f"Completed fold {fold}/{FOLDS} in {time.perf_counter() - started:.1f}s.",
                flush=True,
            )
        probabilities[validation_positions] = fold_probabilities
        diagnostics.append(
            {
                "fold": fold,
                "training_rows": len(training_positions),
                "validation_rows": len(validation_positions),
                "fit_seconds": fit_seconds,
            }
        )
    validate_probabilities(probabilities, len(modelling.y_original))

    predicted = np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]
    predictions = pd.DataFrame(
        {
            "id": modelling.original_ids,
            "fold": assignments,
            "historical_local": modelling.original_ids.isin(historical_local_ids),
            "actual": modelling.y_original,
            "predicted": predicted,
            **{
                f"probability_{label}": probabilities[:, index]
                for index, label in enumerate(CLASS_LABELS)
            },
        }
    )
    predictions["correct"] = predictions["actual"].eq(predictions["predicted"])
    accuracy, recall = summarise_predictions(predictions)
    contrast = stratified_accuracy_difference_interval(
        predictions, seed=SEED, draws=20_000
    )
    result = {
        "seed": SEED,
        "folds": FOLDS,
        "iterations": ITERATIONS,
        "model": "XGBoost archived depth 17 [top-50 deferred identities]",
        "fold_fingerprint": fold_fingerprint,
        "selection": "none; fixed archived specification",
        "historical_local_role": "post-hoc subgroup membership only",
        "accuracy": accuracy.to_dict(orient="records"),
        "accuracy_difference": contrast,
    }
    predictions.to_csv(OUTPUT_DIR / "oof-predictions.csv", index=False)
    accuracy.to_csv(OUTPUT_DIR / "accuracy-summary.csv", index=False)
    recall.to_csv(OUTPUT_DIR / "class-recall-summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUTPUT_DIR / "fold-diagnostics.csv", index=False)
    (OUTPUT_DIR / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def _fold_cache_metadata(
    *,
    fold: int,
    training_rows: int,
    validation_rows: int,
    labelled_rows: int,
    fold_fingerprint: str,
    source_hashes: dict[str, str],
    model_recipe: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment": "full-label generalisation diagnosis",
        "source_sha256": source_hashes,
        "model_recipe": model_recipe,
        "labelled_rows": labelled_rows,
        "cross_validation": {
            "folds": FOLDS,
            "seed": SEED,
            "fingerprint": fold_fingerprint,
            "validation_fold": fold,
            "training_rows": training_rows,
            "validation_rows": validation_rows,
        },
    }


def _model_recipe(spec) -> dict[str, object]:
    return {
        "spec": asdict(spec),
        "variant_parameters": dict(XGBOOST_VARIANTS[spec.variant]),
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


def _validate_source_hashes() -> dict[str, str]:
    actual = {
        filename: _sha256(DATA_DIR / filename)
        for filename in SOURCE_SHA256
    }
    if actual != SOURCE_SHA256:
        raise ValueError("Generalisation-diagnosis source data hashes changed.")
    return actual


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Refuse to fit if any strict fold cache is absent.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(cache_only=_parse_args().cache_only)

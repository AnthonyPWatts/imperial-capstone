"""Run the one locked full-data CatBoost refit-scale comparison."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
import sys

import joblib
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / ".runtime" / "fresh-catboost-refit-scale"
BASELINE_DIR = PROJECT_DIR / ".runtime" / "fresh-spatial-grid-catboost"
BASELINE_CACHE = BASELINE_DIR / "complete-identity-catboost.joblib"
BASELINE_PROTOCOL = BASELINE_DIR / "complete-identity-catboost.protocol-v2.json"
FOLD_SEED = 20260905
FOLDS = 5
FOLD_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)
BASELINE_CACHE_SHA256 = (
    "5e1883c5dfa2c5e36d8937eaa41636f32d2c1f1f6ca200a6b5eed488d6066bbd"
)
BASELINE_PROTOCOL_SHA256 = (
    "89910e4356d2eb7891e090947eee750694b48247ca5b3df5a8a9863126a105a8"
)
SOURCE_SHA256 = {
    "TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    "TrainingSetLabels.csv": (
        "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
    ),
}
FIT_TIME_CODE_SHA256 = {
    "src/catboost_identity_evaluation.py": (
        "b078bb5132cf994c4078aa50d16108841903a0715cd1061d19a57272432e2e9f"
    ),
    "src/catboost_refit_calibration.py": (
        "af98042fb106f84f84addefef5c01969d271600c5c74057df0a8aa0ba513622c"
    ),
    "src/gpu_model_evaluation.py": (
        "e6867cac83f1acd5437a34288748578662cb106d219ac00c3cb2f76c7fd884b5"
    ),
    "src/locked_screen_evidence.py": (
        "c003883ab0428a02b3d2f173b838df67ca568d76f68d06fab7c98e40fa5f1a56"
    ),
    "src/model_evaluation.py": (
        "2c8d668654bfb9a6bbe04b17e1fec1f9d9ce16af647d891b7807d96c77988c20"
    ),
    "scripts/run_catboost_refit_scale_screen.py": (
        "16d6977a8b7261a2cd3d0284d379c81624eccedb74ef4d83681db142a2489c04"
    ),
}
FIT_LOGIC_PATHS = (
    "src/catboost_identity_evaluation.py",
    "src/catboost_refit_calibration.py",
    "src/gpu_model_evaluation.py",
    "src/model_evaluation.py",
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from catboost_identity_evaluation import DEFERRED_IDENTITY_FEATURES
from catboost_identity_evaluation import make_complete_identity_catboost_spec
from catboost_refit_calibration import BASELINE_CANDIDATE
from catboost_refit_calibration import CACHE_VERSION
from catboost_refit_calibration import CORRECTED_CANDIDATE
from catboost_refit_calibration import FOLD_WIN_GATE
from catboost_refit_calibration import INNER_FIT_SHARE
from catboost_refit_calibration import MEAN_ACCURACY_GAIN_GATE
from catboost_refit_calibration import REPAIR_RECALL_DELTA_GATE
from catboost_refit_calibration import WORST_FOLD_DELTA_GATE
from catboost_refit_calibration import evaluate_corrected_refits
from catboost_refit_calibration import load_baseline_evidence
from catboost_refit_calibration import make_refit_recipe
from catboost_refit_calibration import paired_prediction_counts
from catboost_refit_calibration import summarise_refit_calibration
from data_partitioning import make_cross_validation
from fresh_rf_family_evaluation import make_full_labelled_partition
from gpu_model_evaluation import CATBOOST_VARIANTS
from locked_architecture_candidates import sha256_file
from modelling_data import prepare_modelling_data


def main() -> None:
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
    if partitioned.cross_validation_fingerprint != FOLD_FINGERPRINT:
        raise ValueError("Fresh full-data fold membership changed.")
    cross_validation = make_cross_validation(partitioned)

    baseline_spec = make_complete_identity_catboost_spec()
    baseline_metadata = {
        "cache_version": 1,
        "candidate": "complete_identity_catboost",
        "labelled_rows": len(partitioned.y_development),
        "cross_validation_folds": FOLDS,
        "cross_validation_seed": FOLD_SEED,
        "cross_validation_fingerprint": FOLD_FINGERPRINT,
        "model_name": baseline_spec.name,
    }
    baseline = load_baseline_evidence(
        BASELINE_CACHE,
        BASELINE_PROTOCOL,
        partitioned,
        cross_validation,
        expected_cache_sha256=BASELINE_CACHE_SHA256,
        expected_protocol_sha256=BASELINE_PROTOCOL_SHA256,
        expected_metadata=baseline_metadata,
        expected_recipe={
            "spec": asdict(baseline_spec),
            "variant_parameters": dict(CATBOOST_VARIANTS[baseline_spec.variant]),
        },
        expected_feature_policy={
            "version": 1,
            "feature_engineer": "engineer_complete_identity_catboost_features",
            "deferred_identity_features": list(DEFERRED_IDENTITY_FEATURES),
        },
    )
    corrected_spec = replace(
        baseline_spec,
        name="CatBoost d8 [complete identities; 1/0.9 refit scale]",
    )
    recipe = make_refit_recipe(corrected_spec, baseline.selected_iterations)
    metadata = {
        "cache_version": CACHE_VERSION,
        "experiment": "fresh_catboost_refit_scale",
        "labelled_rows": len(partitioned.y_development),
        "cross_validation_folds": FOLDS,
        "cross_validation_seed": FOLD_SEED,
        "cross_validation_fingerprint": FOLD_FINGERPRINT,
        "development_fingerprint": partitioned.development_fingerprint,
        "source_sha256": source_hashes,
        "code_sha256": _code_hashes(),
        "baseline_provenance": baseline.provenance,
        "recipe": recipe,
    }
    corrected = evaluate_corrected_refits(
        corrected_spec,
        partitioned,
        cross_validation,
        selected_iterations=baseline.selected_iterations,
        cache_directory=OUTPUT_DIR,
        common_metadata=metadata,
    )

    summary, folds = summarise_refit_calibration(
        baseline.evaluation,
        corrected,
    )
    counts = paired_prediction_counts(
        baseline.evaluation,
        corrected,
        partitioned,
        cross_validation,
    )
    folds = folds.join(counts)
    folds["baseline_selected_iterations"] = list(
        baseline.selected_iterations.values()
    )
    folds["corrected_refit_iterations"] = recipe[
        "corrected_refit_iterations_by_fold"
    ]
    summary.loc[CORRECTED_CANDIDATE, "net_additional_correct"] = int(
        folds["net_additional_correct"].sum()
    )
    summary.loc[CORRECTED_CANDIDATE, "prediction_disagreements"] = int(
        folds["prediction_disagreements"].sum()
    )
    summary.loc[
        BASELINE_CANDIDATE,
        ["net_additional_correct", "prediction_disagreements"],
    ] = 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "candidate-summary.csv")
    folds.to_csv(OUTPUT_DIR / "fold-deltas.csv")
    evaluation_path = OUTPUT_DIR / "corrected-evaluation.joblib"
    joblib.dump(
        {"metadata": metadata, "evaluation": corrected},
        evaluation_path,
        compress=3,
    )
    passes = bool(summary.loc[CORRECTED_CANDIDATE, "passes_gate"])
    result = {
        "protocol": {
            "labelled_rows": len(partitioned.y_development),
            "cross_validation_folds": FOLDS,
            "cross_validation_seed": FOLD_SEED,
            "cross_validation_fingerprint": FOLD_FINGERPRINT,
            "selection_basis": "one predeclared paired full-data OOF comparison",
            "iteration_correction": "ceil(selected_iterations / 0.9)",
            "inner_fit_share": INNER_FIT_SHARE,
            "baseline_stopping_fits_run": 0,
            "new_early_stopping": False,
            "parameter_variants": 0,
            "historical_local_subset": "not reconstructed or consulted",
            "competition_predictions_generated": False,
        },
        "gate": {
            "minimum_mean_accuracy_gain": MEAN_ACCURACY_GAIN_GATE,
            "minimum_fold_wins": FOLD_WIN_GATE,
            "minimum_worst_fold_delta": WORST_FOLD_DELTA_GATE,
            "minimum_mean_repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
        },
        "source_sha256": source_hashes,
        "code_sha256": metadata["code_sha256"],
        "baseline_provenance": baseline.provenance,
        "recipe": recipe,
        "candidate_summary": _records(summary),
        "fold_deltas": _records(folds),
        "corrected_evaluation_sha256": sha256_file(evaluation_path),
        "passes_gate": passes,
        "selected_candidate": (
            CORRECTED_CANDIDATE if passes else BASELINE_CANDIDATE
        ),
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2), flush=True)


def _validate_source_hashes() -> dict[str, str]:
    hashes = {
        filename: sha256_file(DATA_DIR / filename)
        for filename in SOURCE_SHA256
    }
    if hashes != SOURCE_SHA256:
        raise ValueError("The labelled source hashes changed.")
    return hashes


def _code_hashes() -> dict[str, str]:
    # Cache metadata records the code that performed the expensive fit. Keep
    # that historical provenance immutable while allowing validation/runner
    # hardening which cannot alter the cached probability values.
    current_fit_logic = {
        path: sha256_file(STAGE_DIR / path) for path in FIT_LOGIC_PATHS
    }
    expected_fit_logic = {
        path: FIT_TIME_CODE_SHA256[path] for path in FIT_LOGIC_PATHS
    }
    if current_fit_logic != expected_fit_logic:
        raise ValueError("CatBoost refit model-fitting code changed.")
    return dict(FIT_TIME_CODE_SHA256)


def _records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(
        frame.reset_index().to_json(orient="records", double_precision=15)
    )


if __name__ == "__main__":
    main()

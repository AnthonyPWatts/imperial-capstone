"""Run the predeclared equal ten-seed deep-XGBoost fresh-fold variance test."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import replace
import hashlib
from io import StringIO
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
OUTPUT_DIR = PROJECT_DIR / ".runtime/deep-xgboost-ten-seed-bag-fresh-20260905"
NORMALISED_DIR = PROJECT_DIR / ".runtime/normalised-top-common-screen"
GENERALISATION_DIR = PROJECT_DIR / ".runtime/generalisation-diagnosis"
STRICT_SEED_05_DIR = GENERALISATION_DIR / "strict-fold-cache-v1"
TWO_SEED_DIR = PROJECT_DIR / ".runtime/deep-xgboost-two-seed-variance-check"
FOLD_SEED = 20260905
FOLDS = 5
ITERATIONS = 600
FOLD_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)
GENERALISATION_FINGERPRINT = (
    "0341c8deaf52c4af835b414d28e565c1a6b11899021a5aa7fea283a6850c26d8"
)
DATA_SHA256 = {
    "TrainingSetValues.csv": (
        "d8ebe40f49fe749a851c8ea28601115cd92442f7b20025fbc33881595ae75f5d"
    ),
    "TrainingSetLabels.csv": (
        "ae9b4f893e8e89df3a2187d38cade75c61dfb1d1e156ecd7615fee14fdbe0a24"
    ),
}
CURRENT_VALIDATION_SOURCE_SHA256 = {
    "src/deep_xgboost_ten_seed_bag.py": (
        "2e7e91845e93067d580139b4aa3729d35976ca12d392444cc90de2629b8bb357"
    ),
    "src/deep_xgboost_seed_average.py": (
        "4ad3a1c31f17305e2d2e6d189a5ca7abfa92926c4d96c86c19ac56e05772ea3b"
    ),
    "src/gpu_model_evaluation.py": (
        "e6867cac83f1acd5437a34288748578662cb106d219ac00c3cb2f76c7fd884b5"
    ),
    "src/top_common_identity_evaluation.py": (
        "cc2719cbebd9e776c3e410a10b5eaa017e19f1fb702d4303c2498fdc48904c85"
    ),
    "src/fresh_rf_family_evaluation.py": (
        "856d6b7fb969b06a2e2c044cd46a0ed6b2da89daae76fff9efe7e8f1368d616b"
    ),
    "src/data_preparation.py": (
        "2aeda2f9229275ddf9bb207422d4ae4aacf3c05c46e16321147d1fa5c6e496f6"
    ),
    "src/model_evaluation.py": (
        "2c8d668654bfb9a6bbe04b17e1fec1f9d9ce16af647d891b7807d96c77988c20"
    ),
    "src/generalisation_diagnosis.py": (
        "d03e3dfc4e55da61d3e1b0e8c7d531d494fa2dbdb31addfbe34479d42146e651"
    ),
    "scripts/run_generalisation_diagnosis.py": (
        "a5f71248b4d3a180b8113c5517459ebbe24582f4a43312b00c471bb8f0b1026a"
    ),
}
PINNED_EVIDENCE_SHA256 = {
    "normalised_result": (
        "1fe96a3a0a7cf0dd397e91c6b5c3fc0ba66da61ffb703679880eba625a018524"
    ),
    "seed_20260824_aggregate": (
        "15a8952324a7642b4ea9d15afc3991a3cab38db1b3698aee80c5036122d769b8"
    ),
    "seed_20260824_folds": {
        1: "4e7bac767c76eaa9f839d5bff11c6b76eb001da65f0753eba2694b12b097d049",
        2: "b42570ec2db57ad2c8e6110b2df115adf0b47efc879f99598ae5daeabcaea9bc",
        3: "dfe69aa2022e1ff5d7e48007ce3abd12987d5121cdc9b89141c4497fcf0eb063",
        4: "eb1302fc818d1eaf33bfe43d077e9f18ad4c444e85cdc54834ac4e27ae5593b2",
        5: "c13ac2854c95ade0459a3e57206b2c632bf4bd066465b429486f589229581e33",
    },
    "generalisation_result": (
        "2548340c693280c635d03769420706798fa949698ca3575359d3449eeee2b34c"
    ),
    "seed_20260905_migration": (
        "ceebc91fa8a74e64b5ca6cfd6da9c9c750b84fa578d3f7f1ab00d51e6955ffcd"
    ),
    "seed_20260905_strict_folds": {
        1: "9c8673641681b41fc1e4a9b027d8295382936e948624f636f956fe386a25329c",
        2: "d9cf26175191512b4635027c3809c488542fd8dbd399faddaa0659d5faf2788c",
        3: "2a156a644aaa3229839e251279c250cb5a76a92e6003cb9c0e156a312f6c5251",
        4: "cc59d280cb27ad449240a61743c8de9f3b4f21afc892fcd2ef31eda04794220d",
        5: "d3e1664af63e4d5004324b4175794d0ab292371d5f9c191cc02444e2aa658e5c",
    },
    "two_seed_result": (
        "d35f93283950816d776448d4b8cd7bd552edd0cf768e5f05d145d4203c657c0a"
    ),
    "two_seed_aggregate": (
        "ad76fd787ddb27d44c0efac3652a5a1bba1035d5d0f752417ad0e8044285188b"
    ),
}
FIT_TIME_TWO_SEED_SHA256 = {
    "result.json": (
        "e71917a2a46c3630ff7b4398c38dcd4a1dc80783dd931334fa3e658e7e5bbe9d"
    ),
    "seed-average-evaluation.joblib": (
        "e8b3403e9e9d4ec7b587fb27c7ba966b9fdd463f6bab262febd6e46aea0bd1ae"
    ),
}
INITIAL_FITTED_FOLDS = 40
GENERALISATION_FIT_SOURCE_SHA256 = {
    "TrainingSetValues.csv": DATA_SHA256["TrainingSetValues.csv"],
    "TrainingSetLabels.csv": DATA_SHA256["TrainingSetLabels.csv"],
    "TestSetValues.csv": (
        "a222110d5606910953607efa5112eafb1d6c30a483c4cbcd0b92c8306125c9b5"
    ),
}

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_partitioning import make_cross_validation
from data_preparation import remove_known_redundant_columns
from deep_xgboost_seed_average import FOLD_WIN_GATE
from deep_xgboost_seed_average import MEAN_ACCURACY_GAIN_GATE
from deep_xgboost_seed_average import REPAIR_RECALL_DELTA_GATE
from deep_xgboost_seed_average import WORST_FOLD_DELTA_GATE
from deep_xgboost_seed_average import rebuild_and_validate_evaluation
from deep_xgboost_ten_seed_bag import confirm_equal_ten_seed_bag
from deep_xgboost_ten_seed_bag import FIT_SEEDS
from deep_xgboost_ten_seed_bag import LOCKED_SEEDS
from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from fresh_rf_family_evaluation import make_full_labelled_partition
from generalisation_diagnosis import make_fold_cache_payload
from generalisation_diagnosis import validate_fold_cache_payload
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from gpu_model_evaluation import XGBOOST_VARIANTS
from model_evaluation import build_candidate_evaluation
from model_evaluation import CandidateEvaluation
from modelling_data import ModellingData
from source_data_validation import validate_aligned_ids
from top_common_identity_evaluation import make_top_common_identity_preprocessor
from top_common_identity_evaluation import TOP_COMMON_VALUES
import run_generalisation_diagnosis as generalisation_runner


def main() -> None:
    """Fit the eight missing seeds, then reveal only the locked bag comparisons."""

    if INITIAL_FITTED_FOLDS != len(FIT_SEEDS) * FOLDS:
        raise ValueError("Initial ten-seed fitted-fold count changed.")
    source_hashes = _validate_current_sources()
    partitioned = _load_full_labelled_partition()
    seed_20260824, seed_24_hashes = _load_seed_20260824(partitioned)
    seed_20260905, seed_05_hashes = _load_seed_20260905(partitioned)
    two_seed_bag, two_seed_hashes = _load_two_seed_bag(partitioned)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evidence_hashes = {
        "seed_20260824": seed_24_hashes,
        "seed_20260905": seed_05_hashes,
        "fixed_two_seed_bag": two_seed_hashes,
    }
    fit_time_evidence_hashes = _fit_time_baseline_evidence_hashes(
        evidence_hashes
    )
    evaluations = {
        20260824: seed_20260824,
        20260905: seed_20260905,
    }
    fitted_cache_hashes = {}
    newly_fitted = 0
    for seed in FIT_SEEDS:
        evaluation, hashes, fitted = _load_or_fit_seed(
            seed,
            partitioned,
            source_hashes,
            fit_time_evidence_hashes,
        )
        evaluations[seed] = evaluation
        fitted_cache_hashes[str(seed)] = hashes
        newly_fitted += fitted
    confirmation = confirm_equal_ten_seed_bag(
        partitioned,
        make_cross_validation(partitioned),
        evaluations,
        two_seed_bag,
    )
    summary_path = OUTPUT_DIR / "comparison-summary.csv"
    folds_path = OUTPUT_DIR / "fold-paired-deltas.csv"
    evaluation_path = OUTPUT_DIR / "ten-seed-bag-evaluation.joblib"
    _write_csv(summary_path, confirmation.summary)
    _write_csv(folds_path, confirmation.fold_deltas, index=False)
    aggregate_metadata = {
        "cache_version": 1,
        "candidate": "equal_ten_seed_raw_top_50_deep_xgboost_bag",
        "locked_seeds": list(LOCKED_SEEDS),
        "seed_weight": 0.1,
        "recipe_by_seed": {
            str(seed): _model_recipe(seed) for seed in LOCKED_SEEDS
        },
        "labelled_rows": len(partitioned.y_development),
        "folds": FOLDS,
        "fold_seed": FOLD_SEED,
        "fold_fingerprint": FOLD_FINGERPRINT,
        "data_sha256": DATA_SHA256,
        "current_validation_source_sha256": source_hashes,
        "baseline_evidence_sha256": evidence_hashes,
        "fit_time_baseline_evidence_sha256": fit_time_evidence_hashes,
        "new_seed_fold_cache_sha256": fitted_cache_hashes,
        "competition_data_opened": False,
        "competition_predictions_generated": False,
    }
    aggregate_payload = {
        "cache_version": 1,
        "metadata": aggregate_metadata,
        "evaluation": confirmation.ten_seed_bag,
    }
    _write_or_validate_aggregate(
        evaluation_path,
        aggregate_payload,
        partitioned,
    )
    passes = confirmation.passes_gate_against_both
    result = {
        "screen": "locked equal ten-seed deep-XGBoost fresh-fold variance test",
        "status": (
            "passes_fresh_gate_pending_old_fold_exact_confirmation"
            if passes
            else "fails_fresh_gate_stop"
        ),
        "passes_gate_against_both": passes,
        "old_fold_confirmation_authorised": passes,
        "competition_refit_authorised": False,
        "stop_rule": (
            "proceed only to old-fold exact-ensemble confirmation"
            if passes
            else "stop; do not run old-fold or competition fits"
        ),
        "protocol": {
            "locked_seeds": list(LOCKED_SEEDS),
            "equal_seed_weight": 0.1,
            "labelled_rows": len(partitioned.y_development),
            "folds": FOLDS,
            "fold_seed": FOLD_SEED,
            "fold_fingerprint": FOLD_FINGERPRINT,
            "models_fitted": INITIAL_FITTED_FOLDS,
            "models_fitted_for_initial_evidence": INITIAL_FITTED_FOLDS,
            "models_fitted_this_run": newly_fitted,
            "reused_seeds": [20260824, 20260905],
            "individual_seed_scores_computed_for_selection": False,
            "prefixes_or_bag_sizes_screened": False,
            "weights_screened": False,
            "comparisons": [
                "ten_seed_bag_vs_seed_20260824",
                "ten_seed_bag_vs_fixed_two_seed_bag",
            ],
            "historical_local_subset": "not reconstructed or consulted",
            "competition_data_opened": False,
            "competition_predictions_generated": False,
        },
        "gate": {
            "minimum_mean_accuracy_gain": MEAN_ACCURACY_GAIN_GATE,
            "minimum_fold_wins": FOLD_WIN_GATE,
            "minimum_worst_fold_delta": WORST_FOLD_DELTA_GATE,
            "minimum_repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
            "must_pass_against_both_baselines": True,
        },
        "recipe": {
            "per_seed": {
                str(seed): _model_recipe(seed) for seed in LOCKED_SEEDS
            },
            "bag": {
                "operation": "equal arithmetic mean of ten probability matrices",
                "seed_weight": 0.1,
                "selection": "one predeclared seed list, size and weight only",
            },
        },
        "data_sha256": DATA_SHA256,
        "current_validation_source_sha256": source_hashes,
        "historical_baseline_evidence_sha256": evidence_hashes,
        "fit_time_baseline_evidence_sha256": fit_time_evidence_hashes,
        "new_seed_fold_cache_sha256": fitted_cache_hashes,
        "output_sha256": {
            summary_path.name: _sha256(summary_path),
            folds_path.name: _sha256(folds_path),
            evaluation_path.name: _sha256(evaluation_path),
        },
        "comparison_summary": _records(confirmation.summary),
        "fold_deltas": _records(confirmation.fold_deltas),
    }
    _write_json(OUTPUT_DIR / "result.json", result)
    print(_format_summary(confirmation.summary), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def _load_full_labelled_partition():
    values = pd.read_csv(DATA_DIR / "TrainingSetValues.csv")
    labels = pd.read_csv(DATA_DIR / "TrainingSetLabels.csv")
    validate_aligned_ids(values, labels)
    if list(labels.columns) != ["id", "status_group"]:
        raise ValueError("Training label schema changed.")
    if set(labels["status_group"]) != set(CLASS_LABELS):
        raise ValueError("Training target classes changed.")
    prepared = remove_known_redundant_columns(values)
    identifiers = prepared.pop("id").reset_index(drop=True)
    X = prepared.reset_index(drop=True)
    y = labels["status_group"].reset_index(drop=True)
    empty_ids = identifiers.iloc[0:0].copy()
    empty_X = X.iloc[0:0].copy()
    modelling = ModellingData(
        original_ids=identifiers,
        X_original=X,
        y_original=y,
        competition_ids=empty_ids,
        X_competition=empty_X,
    )
    partitioned = make_full_labelled_partition(
        modelling,
        cross_validation_seed=FOLD_SEED,
    )
    if partitioned.cross_validation_fingerprint != FOLD_FINGERPRINT:
        raise ValueError("Locked fresh-fold membership changed.")
    return partitioned


def _load_seed_20260824(partitioned):
    result_path = NORMALISED_DIR / "result.json"
    aggregate_path = NORMALISED_DIR / "raw-top-50-identities-evaluation.joblib"
    _require_hash(result_path, PINNED_EVIDENCE_SHA256["normalised_result"])
    _require_hash(
        aggregate_path,
        PINNED_EVIDENCE_SHA256["seed_20260824_aggregate"],
    )
    recipe = _normalised_seed_24_recipe()
    probabilities = np.full((59_400, len(CLASS_LABELS)), np.nan)
    hashes = {
        "result.json": PINNED_EVIDENCE_SHA256["normalised_result"],
        aggregate_path.name: PINNED_EVIDENCE_SHA256["seed_20260824_aggregate"],
    }
    source_hashes = dict(DATA_SHA256)
    cross_validation = make_cross_validation(partitioned)
    for fold, (_, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        path = NORMALISED_DIR / f"raw-top-50-identities-fold-{fold}.joblib"
        _require_hash(
            path,
            PINNED_EVIDENCE_SHA256["seed_20260824_folds"][fold],
        )
        payload = joblib.load(path)
        metadata = {
            "cache_version": 1,
            "candidate": "raw_top_50_identities",
            "recipe": recipe,
            "source_sha256": source_hashes,
            "labelled_rows": 59_400,
            "folds": FOLDS,
            "fold_seed": FOLD_SEED,
            "fold_fingerprint": FOLD_FINGERPRINT,
            "validation_fold": fold,
        }
        expected_ids = partitioned.development_ids.iloc[
            validation_positions
        ].to_numpy()
        _validate_seed_24_fold(payload, metadata, expected_ids, path)
        probabilities[validation_positions] = payload["probabilities"]
        hashes[path.name] = PINNED_EVIDENCE_SHA256[
            "seed_20260824_folds"
        ][fold]
    aggregate = joblib.load(aggregate_path)
    if not isinstance(aggregate, dict) or set(aggregate) != {"recipe", "evaluation"}:
        raise ValueError("Seed-20260824 aggregate schema changed.")
    if aggregate["recipe"] != recipe:
        raise ValueError("Seed-20260824 aggregate recipe changed.")
    evaluation = rebuild_and_validate_evaluation(
        aggregate["evaluation"],
        partitioned,
        cross_validation,
        evidence_name="seed_20260824",
    )
    if not np.array_equal(
        evaluation.out_of_fold_probabilities.to_numpy(),
        probabilities,
    ):
        raise ValueError("Seed-20260824 aggregate differs from fold caches.")
    return evaluation, hashes


def _load_seed_20260905(partitioned):
    result_path = GENERALISATION_DIR / "result.json"
    migration_path = STRICT_SEED_05_DIR / "migration.json"
    _require_hash(result_path, PINNED_EVIDENCE_SHA256["generalisation_result"])
    _require_hash(
        migration_path,
        PINNED_EVIDENCE_SHA256["seed_20260905_migration"],
    )
    migration = _read_json(migration_path)
    if migration.get("model_fitting_performed") is not False:
        raise ValueError("Seed-20260905 strict-cache migration refitted models.")
    spec = make_xgboost_spec(variant="archived depth 17", seed=20260905)
    recipe = generalisation_runner._model_recipe(spec)
    cross_validation = make_cross_validation(partitioned)
    generalisation_fingerprint = hashlib.sha256(
        pd.DataFrame(
            {
                "id": partitioned.development_ids,
                "fold": partitioned.validation_folds,
            }
        ).to_csv(index=False).encode("utf-8")
    ).hexdigest()
    if generalisation_fingerprint != GENERALISATION_FINGERPRINT:
        raise ValueError("Seed-20260905 historical fold fingerprint changed.")
    probabilities = np.full((59_400, len(CLASS_LABELS)), np.nan)
    diagnostics = []
    hashes = {
        "result.json": PINNED_EVIDENCE_SHA256["generalisation_result"],
        "migration.json": PINNED_EVIDENCE_SHA256["seed_20260905_migration"],
    }
    for fold, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        path = STRICT_SEED_05_DIR / f"fold-{fold}.joblib"
        _require_hash(
            path,
            PINNED_EVIDENCE_SHA256["seed_20260905_strict_folds"][fold],
        )
        metadata = generalisation_runner._fold_cache_metadata(
            fold=fold,
            training_rows=len(training_positions),
            validation_rows=len(validation_positions),
            labelled_rows=59_400,
            fold_fingerprint=GENERALISATION_FINGERPRINT,
            source_hashes=GENERALISATION_FIT_SOURCE_SHA256,
            model_recipe=recipe,
        )
        expected_ids = partitioned.development_ids.iloc[
            validation_positions
        ].to_numpy()
        fold_probabilities, seconds = validate_fold_cache_payload(
            joblib.load(path),
            metadata,
            expected_ids,
            cache_name=str(path),
        )
        probabilities[validation_positions] = fold_probabilities
        diagnostics.append(
            {
                "validation_fold": fold,
                "selected_iterations": ITERATIONS,
                "fit_seconds": seconds,
            }
        )
        hashes[path.name] = PINNED_EVIDENCE_SHA256[
            "seed_20260905_strict_folds"
        ][fold]
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
    return evaluation, hashes


def _load_two_seed_bag(partitioned):
    result_path = TWO_SEED_DIR / "result.json"
    aggregate_path = TWO_SEED_DIR / "seed-average-evaluation.joblib"
    _require_hash(result_path, PINNED_EVIDENCE_SHA256["two_seed_result"])
    _require_hash(aggregate_path, PINNED_EVIDENCE_SHA256["two_seed_aggregate"])
    payload = joblib.load(aggregate_path)
    if not isinstance(payload, dict) or set(payload) != {
        "cache_version",
        "metadata",
        "evaluation",
    }:
        raise ValueError("Fixed two-seed aggregate cache schema changed.")
    if payload["cache_version"] != 1:
        raise ValueError("Fixed two-seed aggregate cache version changed.")
    evaluation = rebuild_and_validate_evaluation(
        payload["evaluation"],
        partitioned,
        make_cross_validation(partitioned),
        evidence_name="fixed_two_seed_bag",
    )
    return evaluation, {
        result_path.name: PINNED_EVIDENCE_SHA256["two_seed_result"],
        aggregate_path.name: PINNED_EVIDENCE_SHA256["two_seed_aggregate"],
    }


def _fit_time_baseline_evidence_hashes(current_evidence_hashes):
    """Retain the exact provenance embedded in the immutable seed-fold caches.

    The two-seed cache was replayed after its producer contract was hardened.
    Its probabilities and metrics are unchanged, but its serialised metadata
    has new hashes.  The 40 already-fitted seed-fold caches must therefore keep
    validating against the two historical hashes present at fit time, while
    aggregate outputs separately record the current replayed artefacts.
    """

    if not isinstance(current_evidence_hashes, dict) or set(
        current_evidence_hashes
    ) != {"seed_20260824", "seed_20260905", "fixed_two_seed_bag"}:
        raise ValueError("Ten-seed current baseline evidence set changed.")
    current_two_seed = current_evidence_hashes["fixed_two_seed_bag"]
    expected_current = {
        "result.json": PINNED_EVIDENCE_SHA256["two_seed_result"],
        "seed-average-evaluation.joblib": PINNED_EVIDENCE_SHA256[
            "two_seed_aggregate"
        ],
    }
    if current_two_seed != expected_current:
        raise ValueError("Ten-seed current two-seed evidence hashes changed.")
    retained = {
        key: dict(value)
        for key, value in current_evidence_hashes.items()
    }
    retained["fixed_two_seed_bag"] = dict(FIT_TIME_TWO_SEED_SHA256)
    return retained


def _load_or_fit_seed(seed, partitioned, source_hashes, evidence_hashes):
    if seed not in FIT_SEEDS:
        raise ValueError(f"Seed {seed} is not one of the eight missing seeds.")
    spec = replace(
        make_xgboost_spec(variant="archived depth 17", seed=seed),
        name=f"XGBoost archived depth 17 [raw top-50 identities; seed {seed}]",
        feature_policy="accepted_plus_raw_top_50_deferred_identities",
    )
    recipe = _model_recipe(seed)
    probabilities = np.full((59_400, len(CLASS_LABELS)), np.nan)
    diagnostics = []
    hashes = {}
    fitted = 0
    cross_validation = make_cross_validation(partitioned)
    for fold, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        metadata = {
            "cache_version": 1,
            "experiment": "locked equal ten-seed deep-XGBoost bag",
            "locked_seeds": list(LOCKED_SEEDS),
            "candidate_seed": seed,
            "recipe": recipe,
            "data_sha256": DATA_SHA256,
            "current_training_source_sha256": source_hashes,
            "baseline_evidence_sha256": evidence_hashes,
            "labelled_rows": 59_400,
            "cross_validation": {
                "folds": FOLDS,
                "seed": FOLD_SEED,
                "fingerprint": FOLD_FINGERPRINT,
                "validation_fold": fold,
                "training_rows": len(training_positions),
                "validation_rows": len(validation_positions),
            },
            "individual_seed_score_used_for_selection": False,
            "competition_data_opened": False,
        }
        path = OUTPUT_DIR / f"seed-{seed}-fold-{fold}.joblib"
        expected_ids = partitioned.development_ids.iloc[
            validation_positions
        ].to_numpy()
        if path.exists():
            fold_probabilities, seconds = validate_fold_cache_payload(
                joblib.load(path),
                metadata,
                expected_ids,
                cache_name=str(path),
            )
            print(f"Loaded seed {seed} fold {fold}/{FOLDS}.", flush=True)
        else:
            fold_probabilities, seconds = fit_gpu_candidate_probabilities(
                spec,
                partitioned.X_development.iloc[training_positions],
                partitioned.y_development.iloc[training_positions],
                partitioned.X_development.iloc[validation_positions],
                iterations=ITERATIONS,
                preprocessor_factory=make_top_common_identity_preprocessor,
            )
            payload = make_fold_cache_payload(
                metadata,
                expected_ids,
                fold_probabilities,
                seconds,
            )
            _dump_new_cache(path, payload)
            fold_probabilities, seconds = validate_fold_cache_payload(
                joblib.load(path),
                metadata,
                expected_ids,
                cache_name=str(path),
            )
            fitted += 1
            print(
                f"Completed seed {seed} fold {fold}/{FOLDS} in {seconds:.1f}s.",
                flush=True,
            )
        probabilities[validation_positions] = fold_probabilities
        diagnostics.append(
            {
                "validation_fold": fold,
                "selected_iterations": ITERATIONS,
                "fit_seconds": seconds,
            }
        )
        hashes[path.name] = _sha256(path)
    validate_probabilities(probabilities, 59_400)
    evaluation = build_candidate_evaluation(
        model_name=spec.name,
        partitioned_data=partitioned,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=diagnostics,
    )
    return evaluation, hashes, fitted


def _model_recipe(seed):
    spec = replace(
        make_xgboost_spec(variant="archived depth 17", seed=seed),
        name=f"XGBoost archived depth 17 [raw top-50 identities; seed {seed}]",
        feature_policy="accepted_plus_raw_top_50_deferred_identities",
    )
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


def _normalised_seed_24_recipe():
    spec = replace(
        make_xgboost_spec(variant="archived depth 17", seed=20260824),
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


def _validate_seed_24_fold(payload, metadata, expected_ids, path):
    if not isinstance(payload, dict) or set(payload) != {
        "metadata",
        "validation_ids",
        "probabilities",
        "diagnostics",
    }:
        raise ValueError(f"Seed-20260824 fold schema changed: {path}.")
    if payload["metadata"] != metadata:
        raise ValueError(f"Seed-20260824 fold metadata changed: {path}.")
    if not np.array_equal(payload["validation_ids"], expected_ids):
        raise ValueError(f"Seed-20260824 fold IDs changed: {path}.")
    validate_probabilities(payload["probabilities"], len(expected_ids))
    diagnostics = payload["diagnostics"]
    if (
        not isinstance(diagnostics, dict)
        or diagnostics.get("validation_fold") != metadata["validation_fold"]
        or diagnostics.get("selected_iterations") != ITERATIONS
    ):
        raise ValueError(f"Seed-20260824 fold diagnostics changed: {path}.")


def _validate_current_sources():
    hashes = {}
    for name, expected in DATA_SHA256.items():
        path = DATA_DIR / name
        _require_hash(path, expected)
        hashes[f"data/{name}"] = expected
    for relative, expected in CURRENT_VALIDATION_SOURCE_SHA256.items():
        _require_hash(STAGE_DIR / relative, expected)
        hashes[relative] = expected
    return hashes


def _write_or_validate_aggregate(path, payload, partitioned):
    if path.exists():
        cached = joblib.load(path)
    else:
        _dump_new_cache(path, payload)
        cached = joblib.load(path)
    if not isinstance(cached, dict) or set(cached) != {
        "cache_version",
        "metadata",
        "evaluation",
    }:
        raise ValueError("Ten-seed aggregate cache schema changed.")
    if cached["cache_version"] != 1 or cached["metadata"] != payload["metadata"]:
        raise ValueError("Ten-seed aggregate cache metadata changed.")
    if not isinstance(cached["evaluation"], CandidateEvaluation):
        raise ValueError("Ten-seed aggregate cache has no evaluation.")
    replayed = rebuild_and_validate_evaluation(
        cached["evaluation"],
        partitioned,
        make_cross_validation(partitioned),
        evidence_name="ten_seed_bag",
    )
    if not np.array_equal(
        replayed.out_of_fold_probabilities.to_numpy(),
        payload["evaluation"].out_of_fold_probabilities.to_numpy(),
    ):
        raise ValueError("Ten-seed aggregate probabilities changed.")


def _dump_new_cache(path, payload):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {path}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale temporary cache exists: {temporary}.")
    joblib.dump(payload, temporary, compress=3)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {path}.")
    temporary.rename(path)


def _write_csv(path, frame, *, index=True):
    expected = frame.to_csv(index=index)
    if path.exists():
        existing_frame = pd.read_csv(path)
        expected_frame = pd.read_csv(StringIO(expected))
        if not existing_frame.equals(expected_frame):
            raise FileExistsError(f"Refusing to overwrite different CSV: {path}.")
        return
    path.write_text(expected, encoding="utf-8", newline="")


def _write_json(path, value):
    expected = json.dumps(value, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != expected:
            raise FileExistsError(f"Refusing to overwrite different JSON: {path}.")
        return
    path.write_text(expected, encoding="utf-8")


def _records(frame):
    serialisable = frame if isinstance(frame.index, pd.RangeIndex) else frame.reset_index()
    return json.loads(serialisable.to_json(orient="records", double_precision=15))


def _format_summary(summary):
    display = summary.copy()
    for column in display.columns:
        if "accuracy" in column or "recall" in column or "delta" in column:
            display[column] = display[column].map(
                lambda value: f"{value:.4%}" if pd.notna(value) else ""
            )
    return display.to_string()


def _read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}.")
    return value


def _require_hash(path, expected):
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 changed for {path}: {actual}.")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

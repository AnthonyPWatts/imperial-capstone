"""Confirm locked Random Forest substitutions inside the exact deep archive."""

from __future__ import annotations

from dataclasses import asdict as _asdict
from dataclasses import dataclass as _dataclass
import json as _json
from pathlib import Path as _Path
import re as _re
from typing import Mapping as _Mapping

import joblib as _joblib
import numpy as _np
import pandas as _pd

from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from deep_archive_confirmation import blend_deep_archive
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from final_model import CLASS_LABELS
from final_model import validate_probabilities
from fresh_rf_family_evaluation import CURRENT_RF_VARIANT
from fresh_rf_family_evaluation import FOLD_WIN_GATE
from fresh_rf_family_evaluation import FRESH_CROSS_VALIDATION_SEED
from fresh_rf_family_evaluation import LOCKED_MODEL_SEED
from fresh_rf_family_evaluation import LOCKED_RF_VARIANTS
from fresh_rf_family_evaluation import MEAN_ACCURACY_GAIN_GATE
from fresh_rf_family_evaluation import REPAIR_RECALL_DELTA_GATE
from fresh_rf_family_evaluation import RF_SCREEN_CACHE_VERSION
from fresh_rf_family_evaluation import WORST_FOLD_DELTA_GATE
from fresh_rf_family_evaluation import make_locked_rf_specs
from fresh_rf_family_evaluation import paired_fold_deltas
from fresh_rf_family_evaluation import summarise_rf_screen
from gpu_model_evaluation import GpuCandidateSpec
from gpu_model_evaluation import resolve_sklearn_tree_parameters
from locked_architecture_candidates import LockedReplacementDecision
from locked_architecture_candidates import RF_SUBSTITUTION
from locked_architecture_candidates import build_locked_substitution_probabilities
from locked_architecture_candidates import sha256_file
from locked_architecture_candidates import validate_screen_contract
from locked_screen_evidence import recompute_and_validate_oof_evaluation
from model_evaluation import build_candidate_evaluation
from model_evaluation import CandidateEvaluation
from model_preprocessing import RARE_CATEGORY_MINIMUM


FORMAL_RF_CHALLENGER = "Random Forest features 0.3"
ACCURACY_FIRST_RF_CHALLENGER = "Random Forest features 0.3 leaf 2"
CONFIRMATION_CANDIDATES = (
    FORMAL_RF_CHALLENGER,
    ACCURACY_FIRST_RF_CHALLENGER,
)
FRESH_FOLD_FINGERPRINT = (
    "1599c84e2f85e2813aa7eec0a2fcc9535a3ea2a717c5c849db6b9212dd3812c3"
)
ENSEMBLE_MEAN_ACCURACY_GAIN_GATE = 0.0005
ENSEMBLE_FOLD_WIN_GATE = 3
ENSEMBLE_WORST_FOLD_DELTA_GATE = -0.001
ENSEMBLE_REPAIR_RECALL_DELTA_GATE = -0.01
CONFIRMATION_CACHE_VERSION = 1


@_dataclass(frozen=True)
class LockedRfScreenCandidates:
    """Two predeclared challengers and immutable fresh-screen evidence."""

    result: dict[str, object]
    specs: dict[str, GpuCandidateSpec]
    hashes: dict[str, str]


@_dataclass(frozen=True)
class LockedRfEnsembleComparison:
    """Exact incumbent and candidate ensemble evidence on frozen folds."""

    incumbent_evaluation: CandidateEvaluation
    candidate_evaluations: dict[str, CandidateEvaluation]
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame


@_dataclass(frozen=True)
class FreshRfBagComparison:
    """Fixed equal-weight RF bags compared on fresh all-label folds."""

    evaluations: dict[str, CandidateEvaluation]
    summary: _pd.DataFrame
    fold_deltas: _pd.DataFrame


def load_locked_rf_screen_candidates(
    directory: _Path,
) -> LockedRfScreenCandidates:
    """Validate the final fresh result without reading historical test labels."""

    result_path = directory / "result.json"
    summary_path = directory / "candidate-summary.csv"
    result = _read_json(result_path)
    specs = make_locked_rf_specs()
    validate_screen_contract(
        result,
        {
            "cache_version": RF_SCREEN_CACHE_VERSION,
            "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
            "model_seed": LOCKED_MODEL_SEED,
            "folds": 5,
            "labelled_rows": 59_400,
            "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
            "incumbent": CURRENT_RF_VARIANT,
            "locked_candidates": list(LOCKED_RF_VARIANTS),
            "locked_specs": {
                name: _asdict(spec) for name, spec in specs.items()
            },
            "locked_model_parameters": {
                name: resolve_sklearn_tree_parameters(spec)
                for name, spec in specs.items()
            },
            "gate_thresholds": {
                "mean_accuracy_gain": MEAN_ACCURACY_GAIN_GATE,
                "minimum_fold_wins": FOLD_WIN_GATE,
                "worst_fold_delta": WORST_FOLD_DELTA_GATE,
                "minimum_repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
            },
            "passes_gate": True,
            "passing_candidates": [FORMAL_RF_CHALLENGER],
            "selected_candidate": FORMAL_RF_CHALLENGER,
            "selected_candidate_is_challenger": True,
            "screen_winner": ACCURACY_FIRST_RF_CHALLENGER,
            "best_candidate": ACCURACY_FIRST_RF_CHALLENGER,
            "old_local_test_membership_reconstructed": False,
            "old_local_test_scored_separately": False,
            "competition_predictions_generated": False,
            "external_data_used": False,
        },
        screen="fresh RF",
    )
    summary = _read_summary(summary_path)
    _validate_fresh_summary(summary)
    _validate_result_metrics(result, summary)

    hashes = {
        "result": sha256_file(result_path),
        "summary": sha256_file(summary_path),
    }
    for candidate in LOCKED_RF_VARIANTS:
        path = directory / f"{_slug(candidate)}.joblib"
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[f"cache:{candidate}"] = sha256_file(path)
        if candidate in CONFIRMATION_CANDIDATES:
            _validate_fresh_cache_without_labels(
                _joblib.load(path),
                specs[candidate],
            )
    folds_path = directory / "fold-paired-deltas.csv"
    if not folds_path.is_file():
        raise FileNotFoundError(folds_path)
    hashes["folds"] = sha256_file(folds_path)
    provenance_path = directory / "cache-provenance.json"
    provenance = _read_json(provenance_path)
    _validate_cache_provenance(provenance, directory, hashes)
    hashes["cache_provenance"] = sha256_file(provenance_path)
    return LockedRfScreenCandidates(
        result=result,
        specs={name: specs[name] for name in CONFIRMATION_CANDIDATES},
        hashes=hashes,
    )


def compare_locked_rf_ensemble_substitutions(
    partitioned_data: PartitionedData,
    incumbent_components: _Mapping[str, _np.ndarray],
    replacements: _Mapping[str, _np.ndarray],
) -> LockedRfEnsembleComparison:
    """Score both unchanged-weight substitutions on aligned frozen folds."""

    if set(replacements) != set(CONFIRMATION_CANDIDATES):
        raise ValueError("The ensemble confirmation candidate set changed.")
    cross_validation = make_cross_validation(partitioned_data)
    incumbent_probabilities = blend_deep_archive(dict(incumbent_components))
    incumbent_evaluation = _build_evaluation(
        "seed-20260824 deep archive",
        partitioned_data,
        cross_validation,
        incumbent_probabilities,
    )
    evaluations: dict[str, CandidateEvaluation] = {}
    fold_frames = []
    summary_rows = []
    incumbent_predictions = _predictions(incumbent_probabilities)
    for candidate in CONFIRMATION_CANDIDATES:
        probabilities = build_locked_substitution_probabilities(
            incumbent_components,
            LockedReplacementDecision(False, candidate),
            rf_family=replacements[candidate],
        )[RF_SUBSTITUTION]
        evaluation = _build_evaluation(
            candidate,
            partitioned_data,
            cross_validation,
            probabilities,
        )
        evaluations[candidate] = evaluation
        folds = _paired_folds(
            partitioned_data,
            incumbent_evaluation,
            evaluation,
            incumbent_predictions,
            _predictions(probabilities),
            candidate,
        )
        fold_frames.append(folds)
        summary_rows.append(
            _summary_row(
                candidate,
                incumbent_evaluation,
                evaluation,
                folds,
            )
        )
    return LockedRfEnsembleComparison(
        incumbent_evaluation=incumbent_evaluation,
        candidate_evaluations=evaluations,
        summary=_pd.DataFrame(summary_rows).set_index("candidate"),
        fold_deltas=_pd.concat(fold_frames, ignore_index=True),
    )


def make_equal_rf_probability_bags(
    incumbent: _np.ndarray,
    challengers: _Mapping[str, _np.ndarray],
) -> dict[str, _np.ndarray]:
    """Return only the two predeclared 50:50 probability bags."""

    if set(challengers) != set(CONFIRMATION_CANDIDATES):
        raise ValueError("The RF probability-bag candidate set changed.")
    rows = len(incumbent)
    validate_probabilities(incumbent, rows)
    bags = {}
    for candidate in CONFIRMATION_CANDIDATES:
        validate_probabilities(challengers[candidate], rows)
        bag = 0.5 * incumbent + 0.5 * challengers[candidate]
        validate_probabilities(bag, rows)
        bags[candidate] = bag
    return bags


def compare_fresh_rf_probability_bags(
    partitioned_data: PartitionedData,
    evaluations: _Mapping[str, CandidateEvaluation],
) -> FreshRfBagComparison:
    """Apply the existing fresh gate to fixed 50:50 OOF probability bags."""

    expected = {CURRENT_RF_VARIANT, *CONFIRMATION_CANDIDATES}
    if set(evaluations) != expected:
        raise ValueError("Fresh RF probability-bag evidence set changed.")
    cross_validation = make_cross_validation(partitioned_data)
    replayed = {
        name: recompute_and_validate_oof_evaluation(
            evaluation,
            partitioned_data,
            cross_validation,
            candidate=name,
        )
        for name, evaluation in evaluations.items()
    }
    bags = make_equal_rf_probability_bags(
        replayed[CURRENT_RF_VARIANT].out_of_fold_probabilities.to_numpy(),
        {
            candidate: replayed[candidate]
            .out_of_fold_probabilities.to_numpy()
            for candidate in CONFIRMATION_CANDIDATES
        },
    )
    bag_evaluations = {
        candidate: _build_evaluation(
            candidate,
            partitioned_data,
            cross_validation,
            probabilities,
        )
        for candidate, probabilities in bags.items()
    }
    compared = {
        CURRENT_RF_VARIANT: replayed[CURRENT_RF_VARIANT],
        **bag_evaluations,
    }
    folds = paired_fold_deltas(
        partitioned_data,
        cross_validation,
        compared,
    )
    return FreshRfBagComparison(
        evaluations=compared,
        summary=summarise_rf_screen(compared, folds),
        fold_deltas=folds,
    )


def passes_locked_rf_ensemble_gate(
    *,
    mean_accuracy_delta: float,
    fold_wins: int,
    worst_fold_delta: float,
    repair_recall_delta: float,
) -> bool:
    """Apply the predeclared conservative exact-ensemble gate."""

    return (
        _at_least(mean_accuracy_delta, ENSEMBLE_MEAN_ACCURACY_GAIN_GATE)
        and fold_wins >= ENSEMBLE_FOLD_WIN_GATE
        and _at_least(worst_fold_delta, ENSEMBLE_WORST_FOLD_DELTA_GATE)
        and _at_least(
            repair_recall_delta,
            ENSEMBLE_REPAIR_RECALL_DELTA_GATE,
        )
    )


def rf_recipe_signature(spec: GpuCandidateSpec) -> dict[str, object]:
    """Return the complete fixed estimator and preprocessing recipe."""

    return {
        "spec": _asdict(spec),
        "resolved_model_parameters": resolve_sklearn_tree_parameters(spec),
        "preprocessing_contract": {
            "factory": "model_preprocessing.make_initial_preprocessor",
            "feature_policy": spec.feature_policy,
            "model_features": list(MODEL_FEATURES),
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "rare_category_minimum": RARE_CATEGORY_MINIMUM,
        },
    }


def validate_confirmation_cache_payload(
    payload: object,
    expected_metadata: _Mapping[str, object],
    partitioned_data: PartitionedData,
    *,
    candidate: str,
) -> CandidateEvaluation:
    """Reject stale recipes/evidence and replay cached OOF metrics."""

    if not isinstance(payload, dict) or set(payload) != {
        "cache_version",
        "metadata",
        "evaluation",
    }:
        raise ValueError(f"{candidate} confirmation cache schema changed.")
    if payload["cache_version"] != CONFIRMATION_CACHE_VERSION:
        raise ValueError(f"{candidate} confirmation cache version changed.")
    if payload["metadata"] != dict(expected_metadata):
        raise ValueError(f"{candidate} confirmation cache metadata is stale.")
    evaluation = payload["evaluation"]
    if not isinstance(evaluation, CandidateEvaluation):
        raise ValueError(f"{candidate} confirmation cache has no evaluation.")
    return recompute_and_validate_oof_evaluation(
        evaluation,
        partitioned_data,
        make_cross_validation(partitioned_data),
        candidate=candidate,
    )


def _validate_fresh_summary(summary: _pd.DataFrame) -> None:
    if set(summary.index) != set(LOCKED_RF_VARIANTS):
        raise ValueError("Fresh RF summary candidate set changed.")
    recomputed = {}
    for candidate in LOCKED_RF_VARIANTS:
        row = summary.loc[candidate]
        passes = (
            candidate != CURRENT_RF_VARIANT
            and _at_least(row["mean_accuracy_delta"], MEAN_ACCURACY_GAIN_GATE)
            and int(row["fold_wins_vs_incumbent"]) >= FOLD_WIN_GATE
            and _at_least(row["worst_fold_delta"], WORST_FOLD_DELTA_GATE)
            and _at_least(
                row["repair_recall_delta"],
                REPAIR_RECALL_DELTA_GATE,
            )
        )
        if bool(row["passes_gate"]) != passes:
            raise ValueError(f"Fresh RF summary gate changed for {candidate!r}.")
        recomputed[candidate] = passes
    if not recomputed[FORMAL_RF_CHALLENGER]:
        raise ValueError("The formal RF challenger no longer passes its gate.")
    accuracy_first = summary.loc[ACCURACY_FIRST_RF_CHALLENGER]
    if recomputed[ACCURACY_FIRST_RF_CHALLENGER]:
        raise ValueError("The accuracy-first exception unexpectedly passes.")
    if not (
        _at_least(
            accuracy_first["mean_accuracy_delta"],
            MEAN_ACCURACY_GAIN_GATE,
        )
        and int(accuracy_first["fold_wins_vs_incumbent"]) >= FOLD_WIN_GATE
        and _at_least(
            accuracy_first["worst_fold_delta"],
            WORST_FOLD_DELTA_GATE,
        )
        and accuracy_first["repair_recall_delta"] < REPAIR_RECALL_DELTA_GATE
    ):
        raise ValueError("The accuracy-first exception failure mode changed.")


def _validate_result_metrics(
    result: dict[str, object],
    summary: _pd.DataFrame,
) -> None:
    formal = summary.loc[FORMAL_RF_CHALLENGER]
    accuracy_first = summary.loc[ACCURACY_FIRST_RF_CHALLENGER]
    expected = {
        "best_mean_accuracy": float(accuracy_first["mean_accuracy"]),
        "best_mean_accuracy_delta": float(
            accuracy_first["mean_accuracy_delta"]
        ),
        "selected_mean_accuracy": float(formal["mean_accuracy"]),
        "selected_mean_accuracy_delta": float(
            formal["mean_accuracy_delta"]
        ),
        "selected_repair_recall_delta": float(
            formal["repair_recall_delta"]
        ),
    }
    for field, expected_value in expected.items():
        if not _np.isclose(result.get(field), expected_value, atol=1e-15):
            raise ValueError(f"Fresh RF result metric {field!r} changed.")


def _validate_cache_provenance(
    provenance: dict[str, object],
    directory: _Path,
    versioned_hashes: dict[str, str],
) -> None:
    validate_screen_contract(
        provenance,
        {
            "metadata_attached_after_training": True,
            "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
        },
        screen="fresh RF cache provenance",
    )
    caches = provenance.get("caches")
    if not isinstance(caches, dict) or set(caches) != set(LOCKED_RF_VARIANTS):
        raise ValueError("Fresh RF cache provenance candidate set changed.")
    for candidate in LOCKED_RF_VARIANTS:
        record = caches[candidate]
        if not isinstance(record, dict):
            raise ValueError("Fresh RF cache provenance record changed.")
        versioned_path = directory / f"{_slug(candidate)}.joblib"
        bare_path = directory / f"{_slug(candidate)}.bare-v0.joblib"
        expected = {
            "bare_cache": bare_path.name,
            "bare_sha256": sha256_file(bare_path),
            "versioned_cache": versioned_path.name,
            "versioned_sha256": versioned_hashes[f"cache:{candidate}"],
        }
        if record != expected:
            raise ValueError(
                f"Fresh RF cache provenance changed for {candidate!r}."
            )


def _validate_fresh_cache_without_labels(
    payload: object,
    spec: GpuCandidateSpec,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Fresh RF cache is unversioned.")
    expected = {
        "cache_version": RF_SCREEN_CACHE_VERSION,
        "candidate_spec": _asdict(spec),
        "resolved_model_parameters": resolve_sklearn_tree_parameters(spec),
        "preprocessing_contract": rf_recipe_signature(spec)[
            "preprocessing_contract"
        ],
        "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": FRESH_FOLD_FINGERPRINT,
        "labelled_rows": 59_400,
    }
    validate_screen_contract(payload, expected, screen="fresh RF cache")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, CandidateEvaluation):
        raise ValueError("Fresh RF cache has no candidate evaluation.")
    probabilities = evaluation.out_of_fold_probabilities
    if tuple(probabilities.columns) != tuple(CLASS_LABELS):
        raise ValueError("Fresh RF cache class columns changed.")
    validate_probabilities(probabilities.to_numpy(), 59_400)


def _build_evaluation(
    name: str,
    partitioned_data: PartitionedData,
    cross_validation,
    probabilities: _np.ndarray,
) -> CandidateEvaluation:
    validate_probabilities(probabilities, len(partitioned_data.y_development))
    return build_candidate_evaluation(
        model_name=name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=[
            {"validation_fold": fold_number}
            for fold_number in range(1, 6)
        ],
    )


def _paired_folds(
    partitioned_data: PartitionedData,
    incumbent: CandidateEvaluation,
    candidate: CandidateEvaluation,
    incumbent_predictions: _np.ndarray,
    candidate_predictions: _np.ndarray,
    candidate_name: str,
) -> _pd.DataFrame:
    rows = []
    for fold_number, (_, positions) in enumerate(
        make_cross_validation(partitioned_data).split(),
        start=1,
    ):
        actual = partitioned_data.y_development.iloc[positions].to_numpy()
        incumbent_correct = incumbent_predictions[positions] == actual
        candidate_correct = candidate_predictions[positions] == actual
        rows.append(
            {
                "candidate": candidate_name,
                "validation_fold": fold_number,
                "rows": len(positions),
                "incumbent_accuracy": incumbent.fold_metrics.loc[
                    fold_number, "accuracy"
                ],
                "candidate_accuracy": candidate.fold_metrics.loc[
                    fold_number, "accuracy"
                ],
                "accuracy_delta": (
                    candidate.fold_metrics.loc[fold_number, "accuracy"]
                    - incumbent.fold_metrics.loc[fold_number, "accuracy"]
                ),
                "incumbent_repair_recall": incumbent.fold_metrics.loc[
                    fold_number, "recall: functional needs repair"
                ],
                "candidate_repair_recall": candidate.fold_metrics.loc[
                    fold_number, "recall: functional needs repair"
                ],
                "gained_correct": int(
                    (candidate_correct & ~incumbent_correct).sum()
                ),
                "lost_correct": int(
                    (~candidate_correct & incumbent_correct).sum()
                ),
                "net_additional_correct": int(
                    candidate_correct.sum() - incumbent_correct.sum()
                ),
            }
        )
    return _pd.DataFrame(rows)


def _summary_row(
    name: str,
    incumbent: CandidateEvaluation,
    candidate: CandidateEvaluation,
    folds: _pd.DataFrame,
) -> dict[str, object]:
    incumbent_accuracy = incumbent.metric_summary.loc["accuracy", "mean"]
    candidate_accuracy = candidate.metric_summary.loc["accuracy", "mean"]
    repair_metric = "recall: functional needs repair"
    repair_delta = (
        candidate.metric_summary.loc[repair_metric, "mean"]
        - incumbent.metric_summary.loc[repair_metric, "mean"]
    )
    accuracy_delta = candidate_accuracy - incumbent_accuracy
    fold_wins = int(folds["accuracy_delta"].gt(0).sum())
    worst_fold = float(folds["accuracy_delta"].min())
    return {
        "candidate": name,
        "incumbent_mean_accuracy": float(incumbent_accuracy),
        "candidate_mean_accuracy": float(candidate_accuracy),
        "mean_accuracy_delta": float(accuracy_delta),
        "fold_wins": fold_wins,
        "worst_fold_delta": worst_fold,
        "incumbent_repair_recall": float(
            incumbent.metric_summary.loc[repair_metric, "mean"]
        ),
        "candidate_repair_recall": float(
            candidate.metric_summary.loc[repair_metric, "mean"]
        ),
        "repair_recall_delta": float(repair_delta),
        "net_additional_correct": int(folds["net_additional_correct"].sum()),
        "passes_gate": passes_locked_rf_ensemble_gate(
            mean_accuracy_delta=float(accuracy_delta),
            fold_wins=fold_wins,
            worst_fold_delta=worst_fold,
            repair_recall_delta=float(repair_delta),
        ),
    }


def _predictions(probabilities: _np.ndarray) -> _np.ndarray:
    return _np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]


def _read_summary(path: _Path) -> _pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = _pd.read_csv(path, index_col="candidate")
    required = {
        "mean_accuracy",
        "mean_accuracy_delta",
        "fold_wins_vs_incumbent",
        "worst_fold_delta",
        "repair_recall_delta",
        "passes_gate",
    }
    if not required.issubset(summary.columns):
        raise ValueError("Fresh RF summary schema changed.")
    return summary


def _read_json(path: _Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = _json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _slug(value: str) -> str:
    return _re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _at_least(actual: float, threshold: float) -> bool:
    return float(actual) >= threshold - 1e-12

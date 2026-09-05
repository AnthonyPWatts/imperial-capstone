"""Locked fresh-fold evaluation helpers for the Random Forest family."""

from __future__ import annotations

import hashlib as _hashlib
from dataclasses import asdict as _asdict
from typing import Mapping as _Mapping

import numpy as _np
import pandas as _pd
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from gpu_model_evaluation import GpuCandidateSpec
from gpu_model_evaluation import make_sklearn_tree_spec
from gpu_model_evaluation import resolve_sklearn_tree_parameters
from feature_engineering import CATEGORICAL_FEATURES
from feature_engineering import MODEL_FEATURES
from feature_engineering import NUMERIC_FEATURES
from model_evaluation import CandidateEvaluation
from model_evaluation import RANDOM_FOREST_SEED
from model_preprocessing import RARE_CATEGORY_MINIMUM
from modelling_data import ModellingData


FRESH_CROSS_VALIDATION_SEED = 20260905
LOCKED_MODEL_SEED = RANDOM_FOREST_SEED
CURRENT_RF_VARIANT = "Current Random Forest"
LOCKED_RF_VARIANTS = (
    CURRENT_RF_VARIANT,
    "Extra Trees leaf 2",
    "Random Forest features 0.3",
    "Random Forest features 0.3 leaf 2",
)
MEAN_ACCURACY_GAIN_GATE = 0.001
FOLD_WIN_GATE = 3
WORST_FOLD_DELTA_GATE = -0.0025
REPAIR_RECALL_DELTA_GATE = -0.02
RF_SCREEN_CACHE_VERSION = 1


def make_full_labelled_partition(
    modelling_data: ModellingData,
    *,
    cross_validation_seed: int = FRESH_CROSS_VALIDATION_SEED,
) -> PartitionedData:
    """Treat every labelled row symmetrically under one fresh fold assignment.

    The historical local-test membership is not reconstructed. Its fields are
    deliberately empty so downstream evaluation code cannot score that holdout
    as a separate selection dataset.
    """

    development_ids = modelling_data.original_ids.reset_index(drop=True).copy()
    X_development = modelling_data.X_original.reset_index(drop=True).copy()
    y_development = modelling_data.y_original.reset_index(drop=True).copy()
    if not (
        len(development_ids) == len(X_development) == len(y_development)
    ):
        raise ValueError("Full-labelled identifiers, predictors and labels differ in length.")
    if development_ids.duplicated().any():
        raise ValueError("Full-labelled identifiers must be unique.")

    splitter = _StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=cross_validation_seed,
    )
    assignments = _np.zeros(len(y_development), dtype=_np.int8)
    for fold_number, (_, validation_positions) in enumerate(
        splitter.split(X_development, y_development),
        start=1,
    ):
        assignments[validation_positions] = fold_number
    validation_folds = _pd.Series(assignments, name="validation_fold")
    if set(validation_folds) != set(range(1, CROSS_VALIDATION_FOLDS + 1)):
        raise ValueError("Fresh cross-validation did not assign every expected fold.")

    return PartitionedData(
        development_ids=development_ids,
        X_development=X_development,
        y_development=y_development,
        local_test_ids=development_ids.iloc[0:0].copy(),
        X_local_test=X_development.iloc[0:0].copy(),
        y_local_test=y_development.iloc[0:0].copy(),
        validation_folds=validation_folds,
        development_fingerprint=_fingerprint_ids(development_ids),
        local_test_fingerprint="not-created-for-full-labelled-screen",
        cross_validation_fingerprint=_fingerprint_folds(
            development_ids,
            validation_folds,
        ),
    )


def make_locked_rf_specs() -> dict[str, GpuCandidateSpec]:
    """Return exactly the incumbent and three predeclared challenger specs."""

    return {
        variant: make_sklearn_tree_spec(
            variant=variant,
            seed=LOCKED_MODEL_SEED,
        )
        for variant in LOCKED_RF_VARIANTS
    }


def make_rf_cache_payload(
    spec: GpuCandidateSpec,
    partitioned_data: PartitionedData,
    evaluation: CandidateEvaluation,
) -> dict[str, object]:
    """Attach a complete, versioned experiment contract to cached evidence."""

    _validate_evaluations(
        partitioned_data,
        {spec.variant: evaluation},
    )
    if evaluation.model_name != spec.name:
        raise ValueError("Evaluation model name does not match its candidate spec.")
    return {
        "cache_version": RF_SCREEN_CACHE_VERSION,
        "candidate_spec": _asdict(spec),
        "resolved_model_parameters": resolve_sklearn_tree_parameters(spec),
        "preprocessing_contract": _preprocessing_contract(spec),
        "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": (
            partitioned_data.cross_validation_fingerprint
        ),
        "labelled_rows": len(partitioned_data.y_development),
        "evaluation": evaluation,
    }


def validate_rf_cache_payload(
    payload: object,
    spec: GpuCandidateSpec,
    partitioned_data: PartitionedData,
) -> CandidateEvaluation:
    """Return cached evidence only when every locked contract still matches."""

    if not isinstance(payload, dict):
        raise ValueError("RF cache is unversioned; refusing stale evidence.")
    expected_metadata = {
        "cache_version": RF_SCREEN_CACHE_VERSION,
        "candidate_spec": _asdict(spec),
        "resolved_model_parameters": resolve_sklearn_tree_parameters(spec),
        "preprocessing_contract": _preprocessing_contract(spec),
        "cross_validation_seed": FRESH_CROSS_VALIDATION_SEED,
        "cross_validation_fingerprint": (
            partitioned_data.cross_validation_fingerprint
        ),
        "labelled_rows": len(partitioned_data.y_development),
    }
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise ValueError(f"RF cache metadata mismatch for {key!r}.")
    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, CandidateEvaluation):
        raise ValueError("RF cache does not contain a CandidateEvaluation.")
    _validate_evaluations(partitioned_data, {spec.variant: evaluation})
    if evaluation.model_name != spec.name:
        raise ValueError("Cached model name does not match its candidate spec.")
    return evaluation


def paired_fold_deltas(
    partitioned_data: PartitionedData,
    cross_validation: object,
    evaluations: _Mapping[str, CandidateEvaluation],
    *,
    incumbent_name: str = CURRENT_RF_VARIANT,
) -> _pd.DataFrame:
    """Return fold-wise paired correctness changes against the incumbent."""

    if incumbent_name not in evaluations:
        raise ValueError(f"Missing incumbent evaluation: {incumbent_name!r}.")
    incumbent = evaluations[incumbent_name]
    _validate_evaluations(partitioned_data, evaluations)
    incumbent_predictions = incumbent.out_of_fold_probabilities.idxmax(
        axis="columns"
    ).to_numpy()

    rows: list[dict[str, int | float | str]] = []
    for fold_number, (_, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        actual = partitioned_data.y_development.iloc[
            validation_positions
        ].to_numpy()
        incumbent_correct = (
            incumbent_predictions[validation_positions] == actual
        )
        for candidate_name, candidate in evaluations.items():
            if candidate_name == incumbent_name:
                continue
            candidate_predictions = candidate.out_of_fold_probabilities.idxmax(
                axis="columns"
            ).to_numpy()[validation_positions]
            candidate_correct = candidate_predictions == actual
            gained = int((candidate_correct & ~incumbent_correct).sum())
            lost = int((~candidate_correct & incumbent_correct).sum())
            disagreements = int(
                (candidate_predictions != incumbent_predictions[validation_positions]).sum()
            )
            rows.append(
                {
                    "candidate": candidate_name,
                    "validation_fold": fold_number,
                    "rows": len(validation_positions),
                    "incumbent_accuracy": float(incumbent_correct.mean()),
                    "candidate_accuracy": float(candidate_correct.mean()),
                    "accuracy_delta": float(
                        candidate_correct.mean() - incumbent_correct.mean()
                    ),
                    "prediction_disagreements": disagreements,
                    "gained_correct": gained,
                    "lost_correct": lost,
                    "net_additional_correct": gained - lost,
                }
            )
    return _pd.DataFrame(rows).sort_values(
        ["candidate", "validation_fold"],
        ignore_index=True,
    )


def summarise_rf_screen(
    evaluations: _Mapping[str, CandidateEvaluation],
    paired_deltas: _pd.DataFrame,
    *,
    incumbent_name: str = CURRENT_RF_VARIANT,
) -> _pd.DataFrame:
    """Summarise mean metrics and fold robustness for the locked screen."""

    incumbent = evaluations[incumbent_name]
    incumbent_accuracy = float(
        incumbent.metric_summary.loc["accuracy", "mean"]
    )
    incumbent_repair_recall = float(
        incumbent.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    rows = []
    for candidate_name, evaluation in evaluations.items():
        candidate_pairs = paired_deltas.loc[
            paired_deltas["candidate"].eq(candidate_name)
        ]
        if candidate_name == incumbent_name:
            wins = losses = net_correct = 0
            worst_delta = best_delta = 0.0
        else:
            wins = int(candidate_pairs["accuracy_delta"].gt(0).sum())
            losses = int(candidate_pairs["accuracy_delta"].lt(0).sum())
            net_correct = int(candidate_pairs["net_additional_correct"].sum())
            worst_delta = float(candidate_pairs["accuracy_delta"].min())
            best_delta = float(candidate_pairs["accuracy_delta"].max())
        mean_accuracy = float(evaluation.metric_summary.loc["accuracy", "mean"])
        repair_recall = float(
            evaluation.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        repair_recall_delta = repair_recall - incumbent_repair_recall
        mean_accuracy_delta = mean_accuracy - incumbent_accuracy
        rows.append(
            {
                "candidate": candidate_name,
                "mean_accuracy": mean_accuracy,
                "mean_accuracy_delta": mean_accuracy_delta,
                "accuracy_std": float(
                    evaluation.metric_summary.loc["accuracy", "std"]
                ),
                "fold_wins_vs_incumbent": wins,
                "fold_losses_vs_incumbent": losses,
                "worst_fold_delta": worst_delta,
                "best_fold_delta": best_delta,
                "net_additional_correct": net_correct,
                "repair_recall": repair_recall,
                "repair_recall_delta": repair_recall_delta,
                "passes_gate": (
                    candidate_name != incumbent_name
                    and mean_accuracy_delta >= MEAN_ACCURACY_GAIN_GATE
                    and wins >= FOLD_WIN_GATE
                    and worst_delta >= WORST_FOLD_DELTA_GATE
                    and repair_recall_delta >= REPAIR_RECALL_DELTA_GATE
                ),
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("candidate")
        .sort_values(["mean_accuracy", "candidate"], ascending=[False, True])
    )


def _validate_evaluations(
    partitioned_data: PartitionedData,
    evaluations: _Mapping[str, CandidateEvaluation],
) -> None:
    expected_index = partitioned_data.y_development.index
    for name, evaluation in evaluations.items():
        if evaluation.cross_validation_fingerprint != (
            partitioned_data.cross_validation_fingerprint
        ):
            raise ValueError(f"{name!r} uses a different cross-validation design.")
        if not evaluation.out_of_fold_probabilities.index.equals(expected_index):
            raise ValueError(f"{name!r} has misaligned out-of-fold rows.")


def _preprocessing_contract(spec: GpuCandidateSpec) -> dict[str, object]:
    return {
        "factory": "model_preprocessing.make_initial_preprocessor",
        "feature_policy": spec.feature_policy,
        "model_features": list(MODEL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "rare_category_minimum": RARE_CATEGORY_MINIMUM,
    }


def _fingerprint_ids(ids: _pd.Series) -> str:
    digest = _hashlib.sha256()
    for value in sorted(str(value) for value in ids):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _fingerprint_folds(ids: _pd.Series, folds: _pd.Series) -> str:
    digest = _hashlib.sha256()
    memberships = sorted(
        zip((str(value) for value in ids), folds, strict=True),
        key=lambda item: item[0],
    )
    for identifier, fold_number in memberships:
        digest.update(f"{identifier}:{fold_number}\n".encode("utf-8"))
    return digest.hexdigest()

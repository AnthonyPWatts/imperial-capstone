"""Leakage-safe random oversampling for the Pump It Up minority class.

The helpers in this module deliberately operate on one frozen development-fold
training partition at a time.  Validation rows, the reserved local test, and
the unlabelled competition rows never enter the resampling API.
"""

from __future__ import annotations

from collections.abc import Callable as _Callable
from dataclasses import dataclass as _dataclass
import hashlib as _hashlib
import json as _json
from pathlib import Path as _Path
import time as _time

import numpy as _np
import pandas as _pd
from sklearn.metrics import accuracy_score as _accuracy_score
from sklearn.metrics import balanced_accuracy_score as _balanced_accuracy_score
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import f1_score as _f1_score
from sklearn.metrics import recall_score as _recall_score
from sklearn.pipeline import Pipeline as _Pipeline

from data_partitioning import PartitionedData
from model_evaluation import make_random_forest_pipeline


CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)
MINORITY_CLASS = "functional needs repair"
OVERSAMPLING_SEED = 20260821

BASELINE_NAME = "unbalanced baseline"
CLASS_WEIGHTED_NAME = "balanced class weights"
OVERSAMPLED_NAME = "random oversampling"
STRATEGY_NAMES = (BASELINE_NAME, CLASS_WEIGHTED_NAME, OVERSAMPLED_NAME)

_METRIC_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "minority_recall",
    "minority_f1",
)


@_dataclass(frozen=True)
class OversampledTrainingData:
    """One training partition plus traceable minority-class replicas."""

    X: _pd.DataFrame
    y: _pd.Series
    lineage: _pd.DataFrame
    seed: int
    target_class: str
    target_count: int
    counts_before: _pd.Series
    counts_after: _pd.Series


@_dataclass(frozen=True)
class OversamplingEvaluation:
    """Out-of-fold comparison of the three imbalance interventions."""

    cross_validation_fingerprint: str
    fold_metrics: _pd.DataFrame
    metric_summary: _pd.DataFrame
    diagnostics: _pd.DataFrame
    fold_confusions: _pd.DataFrame
    confusion_counts: dict[str, _pd.DataFrame]
    confusion_recall: dict[str, _pd.DataFrame]
    out_of_fold_predictions: _pd.DataFrame


def random_oversample_minority(
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    source_ids: _pd.Series,
    *,
    target_class: str = MINORITY_CLASS,
    target_count: int | None = None,
    seed: int = OVERSAMPLING_SEED,
) -> OversampledTrainingData:
    """Replicate minority rows up to a transparent, deterministic count.

    By default, the minority class is raised to the count of the second-largest
    class.  This gives it material representation without duplicating it all the
    way to the majority count.  Original rows remain first and in source order;
    sampled replicas follow them.  ``source_id`` is lineage only and is never
    inserted into ``X``.
    """

    _validate_training_inputs(X_training, y_training, source_ids, target_class)
    counts_before = _ordered_counts(y_training)
    if not counts_before[target_class] < counts_before.drop(target_class).min():
        raise ValueError(
            f"{target_class!r} is not the unique minority class in this training set."
        )

    if target_count is None:
        target_count = int(counts_before.sort_values(ascending=False).iloc[1])
    if isinstance(target_count, bool) or not isinstance(target_count, (int, _np.integer)):
        raise TypeError("target_count must be an integer or None.")
    target_count = int(target_count)
    minority_count = int(counts_before[target_class])
    if target_count < minority_count:
        raise ValueError(
            "target_count cannot be smaller than the existing minority-class count."
        )

    minority_positions = _np.flatnonzero(y_training.to_numpy() == target_class)
    replica_count = target_count - minority_count
    rng = _np.random.default_rng(seed)
    sampled_positions = rng.choice(
        minority_positions,
        size=replica_count,
        replace=True,
    )
    all_positions = _np.concatenate(
        [_np.arange(len(X_training), dtype="int64"), sampled_positions]
    )

    X_oversampled = X_training.iloc[all_positions].reset_index(drop=True)
    y_oversampled = y_training.iloc[all_positions].reset_index(drop=True)
    y_oversampled.name = y_training.name
    source_id_values = source_ids.iloc[all_positions].reset_index(drop=True)
    is_replica = _np.arange(len(all_positions)) >= len(X_training)
    lineage = _pd.DataFrame(
        {
            "oversampled_row_id": _np.arange(len(all_positions), dtype="int64"),
            "source_id": source_id_values,
            "source_position": all_positions,
            "is_replica": is_replica,
        }
    )
    lineage["replica_number_for_source"] = (
        lineage.groupby("source_id", sort=False).cumcount()
    )

    counts_after = _ordered_counts(y_oversampled)
    result = OversampledTrainingData(
        X=X_oversampled,
        y=y_oversampled,
        lineage=lineage,
        seed=seed,
        target_class=target_class,
        target_count=target_count,
        counts_before=counts_before,
        counts_after=counts_after,
    )
    _validate_oversampled_result(
        result,
        X_training=X_training,
        y_training=y_training,
        source_ids=source_ids,
    )
    return result


def evaluate_oversampling_strategies(
    partitioned_data: PartitionedData,
    cross_validation: object,
    *,
    pipeline_factory: _Callable[[], _Pipeline] = make_random_forest_pipeline,
    seed: int = OVERSAMPLING_SEED,
) -> OversamplingEvaluation:
    """Compare baseline, class weighting and resampling on frozen folds.

    The existing project Random Forest pipeline is the default candidate.  Its
    preprocessing and model are re-fitted from scratch inside every fold.
    """

    development_rows = len(partitioned_data.y_development)
    predictions_by_strategy = {
        name: _np.full(development_rows, None, dtype=object)
        for name in STRATEGY_NAMES
    }
    fold_metric_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    aggregate_confusions = {
        name: _np.zeros((len(CLASS_LABELS), len(CLASS_LABELS)), dtype="int64")
        for name in STRATEGY_NAMES
    }

    split_count = 0
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        split_count += 1
        _validate_fold_positions(
            training_positions,
            validation_positions,
            development_rows=development_rows,
        )
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        ids_training = partitioned_data.development_ids.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        y_validation = partitioned_data.y_development.iloc[validation_positions]

        oversampled = random_oversample_minority(
            X_training,
            y_training,
            ids_training,
            seed=seed + fold_number,
        )
        configurations = (
            (BASELINE_NAME, X_training, y_training, None),
            (CLASS_WEIGHTED_NAME, X_training, y_training, "balanced"),
            (OVERSAMPLED_NAME, oversampled.X, oversampled.y, None),
        )
        for strategy, X_fit, y_fit, class_weight in configurations:
            pipeline = pipeline_factory()
            _validate_pipeline_contract(pipeline)
            if class_weight is not None:
                pipeline.set_params(classifier__class_weight=class_weight)

            started = _time.perf_counter()
            pipeline.fit(X_fit, y_fit)
            predictions = pipeline.predict(X_validation)
            elapsed = _time.perf_counter() - started
            predictions_by_strategy[strategy][validation_positions] = predictions

            fold_confusion = _confusion_matrix(
                y_validation,
                predictions,
                labels=list(CLASS_LABELS),
            )
            aggregate_confusions[strategy] += fold_confusion
            fold_metric_rows.append(
                {
                    "strategy": strategy,
                    "validation_fold": fold_number,
                    **_score_predictions(y_validation, predictions),
                }
            )
            diagnostic_rows.append(
                {
                    "strategy": strategy,
                    "validation_fold": fold_number,
                    "fit_rows": len(X_fit),
                    "validation_rows": len(X_validation),
                    "minority_fit_rows": int(y_fit.eq(MINORITY_CLASS).sum()),
                    "unique_source_rows": len(X_training),
                    "replica_rows": len(X_fit) - len(X_training),
                    "seed": seed + fold_number,
                    "elapsed_seconds": elapsed,
                }
            )
            for actual_position, actual_label in enumerate(CLASS_LABELS):
                for predicted_position, predicted_label in enumerate(CLASS_LABELS):
                    confusion_rows.append(
                        {
                            "strategy": strategy,
                            "validation_fold": fold_number,
                            "actual": actual_label,
                            "predicted": predicted_label,
                            "rows": int(
                                fold_confusion[actual_position, predicted_position]
                            ),
                        }
                    )
        print(
            f"Completed oversampling comparison fold {fold_number}: "
            f"{len(X_training):,} original and {len(oversampled.X):,} "
            "oversampled training rows."
        )

    if split_count == 0:
        raise ValueError("Cross-validation supplied no folds.")
    prediction_frame = _pd.DataFrame(
        predictions_by_strategy,
        index=partitioned_data.y_development.index.copy(),
    )
    if prediction_frame.isna().any().any():
        raise ValueError("Cross-validation did not predict every development row.")

    fold_metrics = _pd.DataFrame(fold_metric_rows).set_index(
        ["strategy", "validation_fold"]
    )
    metric_summary = (
        fold_metrics.loc[:, list(_METRIC_COLUMNS)]
        .groupby(level="strategy", sort=False)
        .agg(["mean", "std", "min", "max"])
    )
    diagnostics = _pd.DataFrame(diagnostic_rows).set_index(
        ["strategy", "validation_fold"]
    )
    fold_confusions = _pd.DataFrame(confusion_rows).set_index(
        ["strategy", "validation_fold", "actual", "predicted"]
    )
    confusion_counts = {
        strategy: _confusion_frame(values)
        for strategy, values in aggregate_confusions.items()
    }
    confusion_recall = {
        strategy: frame.div(frame.sum(axis="columns"), axis="index")
        for strategy, frame in confusion_counts.items()
    }
    _validate_evaluation(
        partitioned_data,
        fold_metrics=fold_metrics,
        diagnostics=diagnostics,
        confusion_counts=confusion_counts,
        predictions=prediction_frame,
        split_count=split_count,
    )
    return OversamplingEvaluation(
        cross_validation_fingerprint=(
            partitioned_data.cross_validation_fingerprint
        ),
        fold_metrics=fold_metrics,
        metric_summary=metric_summary,
        diagnostics=diagnostics,
        fold_confusions=fold_confusions,
        confusion_counts=confusion_counts,
        confusion_recall=confusion_recall,
        out_of_fold_predictions=prediction_frame,
    )


def write_oversampled_fold_artifacts(
    partitioned_data: PartitionedData,
    cross_validation: object,
    output_dir: str | _Path,
    *,
    seed: int = OVERSAMPLING_SEED,
) -> dict[str, object]:
    """Write a development-refit version, fold training versions and a manifest.

    The full-development variant is suitable for fitting before the sealed
    local-test evaluation, after model selection.  It is not a true final
    all-labelled refit because local-test rows remain excluded.  Fold-specific
    versions remain the only artefacts suitable for cross-validation.  All
    CSVs are gzip-compressed and contain development sources only.
    """

    destination = _Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    local_test_id_set = set(partitioned_data.local_test_ids)
    development_id_set = set(partitioned_data.development_ids)
    local_test_overlap = development_id_set.intersection(local_test_id_set)
    if local_test_overlap:
        raise ValueError("Development source IDs overlap the frozen local test IDs.")

    full_development = random_oversample_minority(
        partitioned_data.X_development,
        partitioned_data.y_development,
        partitioned_data.development_ids,
        seed=seed,
    )
    full_development_path = (
        destination / "development-training-oversampled-with-lineage.csv.gz"
    )
    full_development_frame = _combined_artifact_frame(full_development)
    full_development_frame.to_csv(
        full_development_path,
        index=False,
        compression="gzip",
    )
    _validate_combined_written_artifact(
        full_development_path,
        expected_rows=len(full_development.X),
        expected_feature_columns=list(full_development.X.columns),
        expected_counts=full_development.counts_after,
        forbidden_source_ids=local_test_id_set,
    )
    fold_manifests: list[dict[str, object]] = []

    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        fold_dir = destination / f"fold-{fold_number:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        result = random_oversample_minority(
            partitioned_data.X_development.iloc[training_positions],
            partitioned_data.y_development.iloc[training_positions],
            partitioned_data.development_ids.iloc[training_positions],
            seed=seed + fold_number,
        )
        row_ids = result.lineage["oversampled_row_id"]
        values = result.X.copy()
        values.insert(0, "oversampled_row_id", row_ids)
        labels = _pd.DataFrame(
            {
                "oversampled_row_id": row_ids,
                "status_group": result.y,
            }
        )
        paths = {
            "values": fold_dir / "training-values-oversampled.csv.gz",
            "labels": fold_dir / "training-labels-oversampled.csv.gz",
            "lineage": fold_dir / "training-lineage.csv.gz",
        }
        values.to_csv(paths["values"], index=False, compression="gzip")
        labels.to_csv(paths["labels"], index=False, compression="gzip")
        result.lineage.to_csv(paths["lineage"], index=False, compression="gzip")
        _validate_written_artifact(
            paths,
            expected_rows=len(result.X),
            expected_feature_columns=list(result.X.columns),
            expected_counts=result.counts_after,
        )
        fold_manifests.append(
            {
                "validation_fold": fold_number,
                "seed": result.seed,
                "target_class": result.target_class,
                "target_count": result.target_count,
                "training_rows_before": len(training_positions),
                "training_rows_after": len(result.X),
                "validation_rows_untouched": len(validation_positions),
                "counts_before": _counts_as_dict(result.counts_before),
                "counts_after": _counts_as_dict(result.counts_after),
                "validation_counts_untouched": _counts_as_dict(
                    _ordered_counts(
                        partitioned_data.y_development.iloc[validation_positions]
                    )
                ),
                "files": {
                    name: {
                        "path": path.relative_to(destination).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                    for name, path in paths.items()
                },
            }
        )

    manifest: dict[str, object] = {
        "experiment": "08-minority-class-oversampling",
        "method": "random oversampling with replacement",
        "policy": "raise functional needs repair to the second-largest class",
        "base_seed": seed,
        "cross_validation_fingerprint": (
            partitioned_data.cross_validation_fingerprint
        ),
        "development_fingerprint": partitioned_data.development_fingerprint,
        "frozen_local_test_fingerprint": partitioned_data.local_test_fingerprint,
        "frozen_local_test_opened_or_resampled": False,
        "local_test_id_overlap": len(local_test_overlap),
        "feature_columns": list(partitioned_data.X_development.columns),
        "final_refit_development_version": {
            "scope": "complete development partition after frozen local-test split",
            "cross_validation_use": "not permitted; use fold-specific versions",
            "seed": full_development.seed,
            "target_class": full_development.target_class,
            "target_count": full_development.target_count,
            "rows_before": len(partitioned_data.X_development),
            "rows_after": len(full_development.X),
            "counts_before": _counts_as_dict(full_development.counts_before),
            "counts_after": _counts_as_dict(full_development.counts_after),
            "file": {
                "path": full_development_path.relative_to(destination).as_posix(),
                "bytes": full_development_path.stat().st_size,
                "sha256": _sha256_file(full_development_path),
            },
        },
        "folds": fold_manifests,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        _json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def comparison_table(evaluation: OversamplingEvaluation) -> _pd.DataFrame:
    """Return the mean fold metrics in a compact strategy-by-metric table."""

    return evaluation.metric_summary.xs("mean", axis="columns", level=1).loc[
        list(STRATEGY_NAMES), list(_METRIC_COLUMNS)
    ]


def _validate_training_inputs(
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    source_ids: _pd.Series,
    target_class: str,
) -> None:
    if len(X_training) != len(y_training) or len(y_training) != len(source_ids):
        raise ValueError("X, y and source_ids must contain the same number of rows.")
    if len(X_training) == 0:
        raise ValueError("The training partition is empty.")
    if X_training.columns.has_duplicates:
        raise ValueError("Training feature names must be unique.")
    if source_ids.isna().any() or source_ids.duplicated().any():
        raise ValueError("Source identifiers must be complete and unique.")
    if y_training.isna().any() or set(y_training) != set(CLASS_LABELS):
        raise ValueError(
            "The training target must be complete and contain all three classes."
        )
    if target_class not in CLASS_LABELS:
        raise ValueError(f"Unknown target class: {target_class!r}.")


def _validate_oversampled_result(
    result: OversampledTrainingData,
    *,
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
    source_ids: _pd.Series,
) -> None:
    if list(result.X.columns) != list(X_training.columns):
        raise ValueError("Oversampling changed the feature schema.")
    if not result.X.dtypes.equals(X_training.dtypes):
        raise ValueError("Oversampling changed feature dtypes.")
    if not result.X.iloc[: len(X_training)].reset_index(drop=True).equals(
        X_training.reset_index(drop=True)
    ):
        raise ValueError("Oversampling changed or reordered original feature rows.")
    if not result.y.iloc[: len(y_training)].reset_index(drop=True).equals(
        y_training.reset_index(drop=True)
    ):
        raise ValueError("Oversampling changed or reordered original labels.")
    if not result.lineage["source_id"].iloc[: len(source_ids)].reset_index(
        drop=True
    ).equals(source_ids.reset_index(drop=True)):
        raise ValueError("Original source lineage changed.")
    if result.lineage["oversampled_row_id"].duplicated().any():
        raise ValueError("Oversampled row identifiers are not unique.")
    replicas = result.lineage["is_replica"]
    if replicas.iloc[: len(X_training)].any() or not replicas.iloc[len(X_training) :].all():
        raise ValueError("Replica flags do not distinguish original and sampled rows.")
    sampled_labels = result.y.loc[replicas.to_numpy()]
    if not sampled_labels.eq(result.target_class).all():
        raise ValueError("A non-minority row was introduced as a replica.")
    expected_counts = result.counts_before.copy()
    expected_counts[result.target_class] = result.target_count
    if not result.counts_after.equals(expected_counts):
        raise ValueError("Oversampled class counts do not match the requested target.")


def _validate_fold_positions(
    training_positions: _np.ndarray,
    validation_positions: _np.ndarray,
    *,
    development_rows: int,
) -> None:
    if _np.intersect1d(training_positions, validation_positions).size:
        raise ValueError("Training and validation positions overlap.")
    combined = _np.concatenate([training_positions, validation_positions])
    if len(combined) != development_rows or set(combined) != set(range(development_rows)):
        raise ValueError("A fold does not partition the complete development set.")


def _validate_pipeline_contract(pipeline: _Pipeline) -> None:
    if not isinstance(pipeline, _Pipeline):
        raise TypeError("pipeline_factory must return a scikit-learn Pipeline.")
    if "classifier" not in pipeline.named_steps:
        raise ValueError("The pipeline must expose a named 'classifier' step.")
    if "class_weight" not in pipeline.named_steps["classifier"].get_params():
        raise ValueError("The selected classifier does not support class weighting.")


def _validate_evaluation(
    partitioned_data: PartitionedData,
    *,
    fold_metrics: _pd.DataFrame,
    diagnostics: _pd.DataFrame,
    confusion_counts: dict[str, _pd.DataFrame],
    predictions: _pd.DataFrame,
    split_count: int,
) -> None:
    if len(fold_metrics) != len(STRATEGY_NAMES) * split_count:
        raise ValueError("Not every strategy was evaluated on every fold.")
    if not _np.isfinite(fold_metrics.to_numpy()).all():
        raise ValueError("Fold metrics contain non-finite values.")
    if not _np.isfinite(diagnostics.select_dtypes("number").to_numpy()).all():
        raise ValueError("Diagnostics contain non-finite values.")
    if tuple(predictions.columns) != STRATEGY_NAMES:
        raise ValueError("Out-of-fold prediction strategies are incomplete.")
    expected_actual_counts = _ordered_counts(
        partitioned_data.y_development
    ).to_numpy()
    for strategy, frame in confusion_counts.items():
        if int(frame.to_numpy().sum()) != len(partitioned_data.y_development):
            raise ValueError(f"{strategy} confusion matrix does not cover development.")
        if not _np.array_equal(frame.sum(axis="columns"), expected_actual_counts):
            raise ValueError(f"{strategy} confusion rows changed validation prevalence.")


def _validate_written_artifact(
    paths: dict[str, _Path],
    *,
    expected_rows: int,
    expected_feature_columns: list[str],
    expected_counts: _pd.Series,
) -> None:
    values = _pd.read_csv(paths["values"])
    labels = _pd.read_csv(paths["labels"])
    lineage = _pd.read_csv(paths["lineage"])
    if len(values) != expected_rows or len(labels) != expected_rows or len(lineage) != expected_rows:
        raise ValueError("Written fold files have inconsistent row counts.")
    if list(values.columns) != ["oversampled_row_id", *expected_feature_columns]:
        raise ValueError("Written values schema changed.")
    if list(labels.columns) != ["oversampled_row_id", "status_group"]:
        raise ValueError("Written labels schema changed.")
    expected_lineage = [
        "oversampled_row_id",
        "source_id",
        "source_position",
        "is_replica",
        "replica_number_for_source",
    ]
    if list(lineage.columns) != expected_lineage:
        raise ValueError("Written lineage schema changed.")
    if not values["oversampled_row_id"].equals(labels["oversampled_row_id"]):
        raise ValueError("Written values and labels identifiers do not align.")
    if not values["oversampled_row_id"].equals(lineage["oversampled_row_id"]):
        raise ValueError("Written values and lineage identifiers do not align.")
    if values["oversampled_row_id"].duplicated().any():
        raise ValueError("Written oversampled row identifiers are not unique.")
    actual_counts = _ordered_counts(labels["status_group"])
    if not actual_counts.equals(expected_counts):
        raise ValueError("Written label counts differ from the verified in-memory data.")


def _combined_artifact_frame(
    result: OversampledTrainingData,
) -> _pd.DataFrame:
    metadata = result.lineage.loc[
        :,
        [
            "oversampled_row_id",
            "source_id",
            "source_position",
            "is_replica",
            "replica_number_for_source",
        ],
    ].copy()
    metadata["status_group"] = result.y
    return _pd.concat(
        [metadata.reset_index(drop=True), result.X.reset_index(drop=True)],
        axis="columns",
    )


def _validate_combined_written_artifact(
    path: _Path,
    *,
    expected_rows: int,
    expected_feature_columns: list[str],
    expected_counts: _pd.Series,
    forbidden_source_ids: set[object],
) -> None:
    written = _pd.read_csv(path)
    metadata_columns = [
        "oversampled_row_id",
        "source_id",
        "source_position",
        "is_replica",
        "replica_number_for_source",
        "status_group",
    ]
    if len(written) != expected_rows:
        raise ValueError("Written development artefact has the wrong row count.")
    if list(written.columns) != [*metadata_columns, *expected_feature_columns]:
        raise ValueError("Written development artefact schema changed.")
    if written["oversampled_row_id"].duplicated().any():
        raise ValueError("Written development oversampled IDs are not unique.")
    if set(written["source_id"]).intersection(forbidden_source_ids):
        raise ValueError("A frozen local-test source ID reached the written artefact.")
    actual_counts = _ordered_counts(written["status_group"])
    if not actual_counts.equals(expected_counts):
        raise ValueError("Written development label counts changed.")


def _score_predictions(
    y_true: _pd.Series,
    predictions: _np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": _accuracy_score(y_true, predictions),
        "balanced_accuracy": _balanced_accuracy_score(y_true, predictions),
        "macro_f1": _f1_score(
            y_true,
            predictions,
            labels=list(CLASS_LABELS),
            average="macro",
            zero_division=0,
        ),
        "minority_recall": _recall_score(
            y_true,
            predictions,
            labels=[MINORITY_CLASS],
            average="macro",
            zero_division=0,
        ),
        "minority_f1": _f1_score(
            y_true,
            predictions,
            labels=[MINORITY_CLASS],
            average="macro",
            zero_division=0,
        ),
    }


def _ordered_counts(target: _pd.Series) -> _pd.Series:
    return target.value_counts().reindex(CLASS_LABELS, fill_value=0).astype("int64")


def _counts_as_dict(counts: _pd.Series) -> dict[str, int]:
    return {str(label): int(value) for label, value in counts.items()}


def _confusion_frame(values: _np.ndarray) -> _pd.DataFrame:
    return _pd.DataFrame(
        values,
        index=_pd.Index(CLASS_LABELS, name="actual"),
        columns=_pd.Index(CLASS_LABELS, name="predicted"),
    )


def _sha256_file(path: _Path) -> str:
    digest = _hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

"""Evaluate bounded training-only outlier filters on the frozen folds."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import time as _time
from typing import Callable as _Callable

import numpy as _np
import pandas as _pd
from sklearn.ensemble import IsolationForest as _IsolationForest

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from feature_engineering import engineer_initial_features
from gpu_model_evaluation import CLASS_LABELS
from gpu_model_evaluation import fit_gpu_candidate_probabilities
from gpu_model_evaluation import make_xgboost_spec
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_evaluation import make_random_forest_pipeline
from model_evaluation import evaluate_weighted_soft_vote
from probability_diagnostics import probability_quality_summary


XGBOOST_WEIGHT = 0.55
RANDOM_FOREST_WEIGHT = 0.45
OUTLIER_RANDOM_SEED = 20260822
MAXIMUM_REMOVAL_SHARE = 0.02
SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_WORST_FOLD = -0.0025
SELECTION_GATE_REPAIR_RECALL = -0.02

_CONTINUOUS_SUPPORT_FEATURES = (
    "amount_tsh",
    "gps_height",
    "longitude",
    "latitude",
    "num_private",
    "population",
    "days_since_recorded",
    "pump_age_at_recording",
)
_LOG_SUPPORT_FEATURES = (
    "amount_tsh",
    "num_private",
    "population",
)


@_dataclass(frozen=True)
class OutlierFilterPolicy:
    """One fixed rule for removing rows from an outer-training fold."""

    key: str
    label: str
    rationale: str
    kind: str
    contamination: float = 0.0
    minimum_duplicate_rows: int = 0
    minimum_duplicate_purity: float = 0.0


@_dataclass(frozen=True)
class TrainingFilterResult:
    """The retained training-row mask and auditable removal counts."""

    retained: _pd.Series
    diagnostics: dict[str, int | float]


@_dataclass(frozen=True)
class OutlierFilterTrial:
    """Component predictions and removal evidence for one filter policy."""

    policy: OutlierFilterPolicy
    xgboost: CandidateEvaluation
    random_forest: CandidateEvaluation
    blend: CandidateEvaluation
    removal_diagnostics: _pd.DataFrame


OUTLIER_FILTER_POLICIES = {
    policy.key: policy
    for policy in (
        OutlierFilterPolicy(
            key="physical_contradictions",
            label="Unambiguous physical contradictions",
            rationale=(
                "Remove future construction years and negative non-negative "
                "measurements from the current training fold"
            ),
            kind="physical_contradictions",
        ),
        OutlierFilterPolicy(
            key="strict_duplicate_conflicts",
            label="Strict accepted-feature duplicate conflicts",
            rationale=(
                "Within exact accepted-feature duplicate groups of at least "
                "three rows, remove labels outside a two-thirds consensus"
            ),
            kind="duplicate_conflicts",
            minimum_duplicate_rows=3,
            minimum_duplicate_purity=2.0 / 3.0,
        ),
        OutlierFilterPolicy(
            key="isolation_forest_005",
            label="Isolation Forest 0.5%",
            rationale=(
                "Remove the most isolated 0.5% of robustly scaled numeric "
                "training support without consulting labels"
            ),
            kind="isolation_forest",
            contamination=0.005,
        ),
        OutlierFilterPolicy(
            key="isolation_forest_010",
            label="Isolation Forest 1.0%",
            rationale=(
                "Remove the most isolated 1.0% of robustly scaled numeric "
                "training support without consulting labels"
            ),
            kind="isolation_forest",
            contamination=0.010,
        ),
    )
}


def fit_training_filter(
    policy: OutlierFilterPolicy,
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
) -> TrainingFilterResult:
    """Fit one policy on training rows and return rows retained for modelling."""

    _validate_training_inputs(X_training, y_training)
    if policy.kind == "physical_contradictions":
        flagged = _physical_contradiction_mask(X_training)
    elif policy.kind == "duplicate_conflicts":
        flagged = _duplicate_conflict_mask(policy, X_training, y_training)
    elif policy.kind == "isolation_forest":
        flagged = _isolation_forest_mask(policy, X_training)
    else:
        raise ValueError(f"Unknown outlier-filter kind: {policy.kind!r}.")

    retained = ~flagged
    _validate_retained_rows(retained, y_training)
    removed_target = y_training.loc[~retained]
    diagnostics: dict[str, int | float] = {
        "training_rows_before": len(X_training),
        "training_rows_after": int(retained.sum()),
        "removed_rows": int((~retained).sum()),
        "removed_share": float((~retained).mean()),
    }
    for label in CLASS_LABELS:
        key = label.replace(" ", "_")
        diagnostics[f"removed_{key}_rows"] = int(removed_target.eq(label).sum())
        class_rows = int(y_training.eq(label).sum())
        diagnostics[f"removed_{key}_share"] = (
            float(removed_target.eq(label).sum() / class_rows)
            if class_rows
            else 0.0
        )
    return TrainingFilterResult(retained=retained, diagnostics=diagnostics)


def evaluate_outlier_filter(
    policy: OutlierFilterPolicy,
    partitioned_data: PartitionedData,
    cross_validation: object,
    accepted_xgboost: CandidateEvaluation,
    *,
    xgboost_fitter: _Callable[..., tuple[_np.ndarray, float]] = (
        fit_gpu_candidate_probabilities
    ),
    random_forest_factory: _Callable[[], object] = make_random_forest_pipeline,
) -> OutlierFilterTrial:
    """Refit the accepted components after filtering only outer-training rows."""

    _validate_accepted_xgboost(partitioned_data, accepted_xgboost)
    xgboost_probabilities = _empty_probability_matrix(partitioned_data)
    forest_probabilities = _empty_probability_matrix(partitioned_data)
    xgboost_diagnostics = []
    forest_diagnostics = []
    removal_rows = []
    xgboost_spec = make_xgboost_spec(variant="depth 8 child 1")

    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]
        filtered = fit_training_filter(policy, X_training, y_training)
        X_fit = X_training.loc[filtered.retained]
        y_fit = y_training.loc[filtered.retained]
        selected_iterations = int(
            accepted_xgboost.diagnostics.loc[
                fold_number,
                "selected_iterations",
            ]
        )

        fold_started = _time.perf_counter()
        xgboost_fold, xgboost_seconds = xgboost_fitter(
            xgboost_spec,
            X_fit,
            y_fit,
            X_validation,
            iterations=selected_iterations,
        )
        xgboost_probabilities[validation_positions] = xgboost_fold
        xgboost_diagnostics.append(
            {
                "validation_fold": fold_number,
                "selected_iterations": selected_iterations,
                "fit_predict_seconds": xgboost_seconds,
                **filtered.diagnostics,
            }
        )

        forest_started = _time.perf_counter()
        forest = random_forest_factory()
        forest.fit(X_fit, y_fit)
        classifier = forest.named_steps["classifier"]
        forest_fold = _pd.DataFrame(
            forest.predict_proba(X_validation),
            columns=classifier.classes_,
        ).loc[:, list(CLASS_LABELS)]
        forest_seconds = _time.perf_counter() - forest_started
        forest_probabilities[validation_positions] = forest_fold.to_numpy()
        forest_diagnostics.append(
            {
                "validation_fold": fold_number,
                "fit_predict_seconds": forest_seconds,
                **filtered.diagnostics,
            }
        )
        removal_rows.append(
            {"validation_fold": fold_number, **filtered.diagnostics}
        )
        print(
            f"Completed {policy.key} fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS}: removed "
            f"{filtered.diagnostics['removed_rows']} rows in "
            f"{_time.perf_counter() - fold_started:.1f} seconds.",
            flush=True,
        )

    xgboost = build_candidate_evaluation(
        model_name=f"XGBoost [{policy.key}]",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=xgboost_probabilities,
        diagnostic_rows=xgboost_diagnostics,
    )
    random_forest = build_candidate_evaluation(
        model_name=f"Random Forest [{policy.key}]",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=forest_probabilities,
        diagnostic_rows=forest_diagnostics,
    )
    blend = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        xgboost,
        random_forest,
        weights=[XGBOOST_WEIGHT, RANDOM_FOREST_WEIGHT],
        model_name=f"55% XGBoost + 45% Random Forest [{policy.key}]",
    )
    return OutlierFilterTrial(
        policy=policy,
        xgboost=xgboost,
        random_forest=random_forest,
        blend=blend,
        removal_diagnostics=_pd.DataFrame(removal_rows).set_index(
            "validation_fold"
        ),
    )


def summarise_outlier_trials(
    accepted: CandidateEvaluation,
    target: _pd.Series,
    trials: list[OutlierFilterTrial] | tuple[OutlierFilterTrial, ...],
) -> _pd.DataFrame:
    """Compare filtered candidates with the unchanged accepted ensemble."""

    if not trials:
        raise ValueError("At least one outlier-filter trial is required.")
    baseline_accuracy = float(accepted.metric_summary.loc["accuracy", "mean"])
    baseline_repair = float(
        accepted.metric_summary.loc["recall: functional needs repair", "mean"]
    )
    baseline_quality = probability_quality_summary(
        target,
        accepted.out_of_fold_probabilities.to_numpy(),
    )
    baseline_log_loss = float(baseline_quality["log_loss"])
    baseline_brier = float(baseline_quality["multiclass_brier"])
    rows = [
        {
            "policy": "accepted_baseline",
            "label": "Accepted ensemble without filtering",
            "mean_accuracy": baseline_accuracy,
            "accuracy_change": 0.0,
            "fold_wins": 0,
            "worst_fold_change": 0.0,
            "repair_recall": baseline_repair,
            "repair_recall_change": 0.0,
            "mean_removed_rows": 0.0,
            "mean_removed_share": 0.0,
            "log_loss": baseline_log_loss,
            "log_loss_change": 0.0,
            "brier_score": baseline_brier,
            "brier_score_change": 0.0,
            "passes_gate": False,
        }
    ]
    for trial in trials:
        candidate = trial.blend
        _validate_candidate_alignment(accepted, candidate)
        fold_change = (
            candidate.fold_metrics["accuracy"]
            - accepted.fold_metrics["accuracy"]
        )
        accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        accuracy_change = accuracy - baseline_accuracy
        repair_change = repair - baseline_repair
        quality = probability_quality_summary(
            target,
            candidate.out_of_fold_probabilities.to_numpy(),
        )
        log_loss = float(quality["log_loss"])
        brier = float(quality["multiclass_brier"])
        rows.append(
            {
                "policy": trial.policy.key,
                "label": trial.policy.label,
                "mean_accuracy": accuracy,
                "accuracy_change": accuracy_change,
                "fold_wins": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall": repair,
                "repair_recall_change": repair_change,
                "mean_removed_rows": float(
                    trial.removal_diagnostics["removed_rows"].mean()
                ),
                "mean_removed_share": float(
                    trial.removal_diagnostics["removed_share"].mean()
                ),
                "log_loss": log_loss,
                "log_loss_change": log_loss - baseline_log_loss,
                "brier_score": brier,
                "brier_score_change": brier - baseline_brier,
                "passes_gate": (
                    accuracy_change >= SELECTION_GATE_ACCURACY
                    and int(fold_change.gt(0).sum()) >= 3
                    and float(fold_change.min()) >= SELECTION_GATE_WORST_FOLD
                    and repair_change >= SELECTION_GATE_REPAIR_RECALL
                ),
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("policy")
        .sort_values(["mean_accuracy", "policy"], ascending=[False, True])
    )


def _physical_contradiction_mask(X_training: _pd.DataFrame) -> _pd.Series:
    recorded_year = _pd.to_datetime(
        X_training["date_recorded"],
        errors="raise",
    ).dt.year
    construction_year = _pd.to_numeric(
        X_training["construction_year"],
        errors="coerce",
    )
    flagged = construction_year.gt(recorded_year)
    for column in ("amount_tsh", "population", "num_private"):
        flagged |= _pd.to_numeric(X_training[column], errors="coerce").lt(0)
    return flagged.astype(bool)


def _duplicate_conflict_mask(
    policy: OutlierFilterPolicy,
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
) -> _pd.Series:
    if policy.minimum_duplicate_rows < 3:
        raise ValueError("Duplicate filtering requires groups of at least three.")
    if not 0.5 < policy.minimum_duplicate_purity <= 1.0:
        raise ValueError("Duplicate purity must be above one half and at most one.")
    engineered = engineer_initial_features(X_training)
    row_hash = _pd.util.hash_pandas_object(
        engineered.astype("string").fillna("__missing_hash_value__"),
        index=False,
    )
    pairs = _pd.DataFrame(
        {"row_hash": row_hash.to_numpy(), "target": y_training.to_numpy()},
        index=y_training.index,
    )
    class_counts = (
        pairs.groupby(["row_hash", "target"], sort=False)
        .size()
        .rename("class_rows")
        .reset_index()
    )
    group_rows = class_counts.groupby("row_hash", sort=False)[
        "class_rows"
    ].sum().rename("group_rows")
    modal = (
        class_counts.sort_values(
            ["row_hash", "class_rows", "target"],
            ascending=[True, False, True],
            kind="stable",
        )
        .drop_duplicates("row_hash")
        .set_index("row_hash")
        .join(group_rows)
    )
    modal["purity"] = modal["class_rows"] / modal["group_rows"]
    annotated = pairs.join(
        modal[["target", "group_rows", "purity"]].rename(
            columns={"target": "modal_target"}
        ),
        on="row_hash",
    )
    return (
        annotated["group_rows"].ge(policy.minimum_duplicate_rows)
        & annotated["purity"].ge(policy.minimum_duplicate_purity)
        & annotated["target"].ne(annotated["modal_target"])
    )


def _isolation_forest_mask(
    policy: OutlierFilterPolicy,
    X_training: _pd.DataFrame,
) -> _pd.Series:
    if not 0.0 < policy.contamination <= MAXIMUM_REMOVAL_SHARE:
        raise ValueError(
            "Isolation contamination must be positive and within the removal cap."
        )
    support = _numeric_support_matrix(X_training)
    detector = _IsolationForest(
        n_estimators=200,
        max_samples=min(4096, len(support)),
        contamination=policy.contamination,
        random_state=OUTLIER_RANDOM_SEED,
        n_jobs=-1,
    )
    return _pd.Series(
        detector.fit_predict(support) == -1,
        index=X_training.index,
        name="outlier",
    )


def _numeric_support_matrix(X_training: _pd.DataFrame) -> _pd.DataFrame:
    engineered = engineer_initial_features(X_training)
    support = engineered.loc[:, list(_CONTINUOUS_SUPPORT_FEATURES)].astype(
        "float64"
    )
    for column in _LOG_SUPPORT_FEATURES:
        support[column] = _np.log1p(support[column].clip(lower=0))
    medians = support.median()
    support = support.fillna(medians)
    interquartile_range = support.quantile(0.75) - support.quantile(0.25)
    scale = interquartile_range.mask(interquartile_range.eq(0), 1.0)
    support = (support - medians).div(scale)
    if not _np.isfinite(support.to_numpy()).all():
        raise ValueError("Numeric outlier support contains non-finite values.")
    return support


def _empty_probability_matrix(partitioned_data: PartitionedData) -> _np.ndarray:
    return _np.full(
        (len(partitioned_data.y_development), len(CLASS_LABELS)),
        fill_value=_np.nan,
    )


def _validate_training_inputs(
    X_training: _pd.DataFrame,
    y_training: _pd.Series,
) -> None:
    if not X_training.index.equals(y_training.index):
        raise ValueError("Training predictors and target indices must align.")
    if X_training.empty:
        raise ValueError("Training outlier filtering requires at least one row.")
    if y_training.isna().any():
        raise ValueError("Training targets must be complete.")


def _validate_retained_rows(retained: _pd.Series, y_training: _pd.Series) -> None:
    if retained.dtype != bool or not retained.index.equals(y_training.index):
        raise ValueError("The retained-row mask must be aligned and boolean.")
    removed_share = float((~retained).mean())
    if removed_share > MAXIMUM_REMOVAL_SHARE + 1e-12:
        raise ValueError(
            f"Outlier filtering removed {removed_share:.3%}, above the "
            f"{MAXIMUM_REMOVAL_SHARE:.1%} cap."
        )
    before = set(y_training)
    after = set(y_training.loc[retained])
    if after != before:
        raise ValueError("Outlier filtering removed all rows from a class.")


def _validate_accepted_xgboost(
    partitioned_data: PartitionedData,
    accepted_xgboost: CandidateEvaluation,
) -> None:
    if (
        accepted_xgboost.cross_validation_fingerprint
        != partitioned_data.cross_validation_fingerprint
    ):
        raise ValueError("Accepted XGBoost evidence uses different frozen folds.")
    required = _pd.Index(range(1, CROSS_VALIDATION_FOLDS + 1))
    if not required.equals(accepted_xgboost.diagnostics.index):
        raise ValueError("Accepted XGBoost diagnostics do not cover five folds.")
    if "selected_iterations" not in accepted_xgboost.diagnostics:
        raise ValueError("Accepted XGBoost tree counts are unavailable.")


def _validate_candidate_alignment(
    accepted: CandidateEvaluation,
    candidate: CandidateEvaluation,
) -> None:
    if (
        candidate.cross_validation_fingerprint
        != accepted.cross_validation_fingerprint
    ):
        raise ValueError("Outlier candidate uses different frozen folds.")
    if not candidate.out_of_fold_probabilities.index.equals(
        accepted.out_of_fold_probabilities.index
    ):
        raise ValueError("Outlier candidate rows do not align with the baseline.")

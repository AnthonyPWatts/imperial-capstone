"""Evaluate bounded neural-network candidates on the frozen development folds."""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
import time as _time

import numpy as _np
import pandas as _pd
from sklearn.neural_network import MLPClassifier as _MLPClassifier
from sklearn.pipeline import Pipeline as _Pipeline

from data_partitioning import CROSS_VALIDATION_FOLDS, PartitionedData
from high_cardinality_features import HIGH_CARDINALITY_POLICIES
from high_cardinality_features import make_high_cardinality_preprocessor
from model_evaluation import CandidateEvaluation
from model_evaluation import build_candidate_evaluation
from model_evaluation import compare_candidate_diversity
from model_evaluation import evaluate_weighted_soft_vote
from model_preprocessing import make_initial_preprocessor


ANN_SEED = 20260822
ANN_MAX_ITERATIONS = 120
ANN_EARLY_STOPPING_PATIENCE = 10
ANN_VALIDATION_FRACTION = 0.10

ACCEPTED_BASELINE_NAME = (
    "55% XGBoost depth 8 child 1 [current one-hot] + 45% Random Forest"
)
ANN_BLEND_WEIGHTS = (
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
)

SELECTION_GATE_ACCURACY = 0.001
SELECTION_GATE_FOLD_WINS = 3
SELECTION_GATE_WORST_FOLD = -0.0025
SELECTION_GATE_REPAIR_RECALL = -0.02

_CLASS_LABELS = (
    "functional",
    "functional needs repair",
    "non functional",
)


@_dataclass(frozen=True)
class AnnCandidateSpec:
    """One explicit multilayer-perceptron candidate."""

    key: str
    label: str
    hidden_layer_sizes: tuple[int, ...]
    alpha: float
    feature_policy: str = "baseline"
    learning_rate_init: float = 0.001
    batch_size: int = 256
    seed: int = ANN_SEED

    @property
    def model_name(self) -> str:
        feature_label = (
            "current one-hot"
            if self.feature_policy == "baseline"
            else self.feature_policy.replace("_", " ")
        )
        return f"MLP {self.label} [scaled {feature_label}]"


ANN_CANDIDATES = {
    spec.key: spec
    for spec in (
        AnnCandidateSpec(
            key="relu_64",
            label="ReLU 64",
            hidden_layer_sizes=(64,),
            alpha=0.0001,
        ),
        AnnCandidateSpec(
            key="relu_128",
            label="ReLU 128",
            hidden_layer_sizes=(128,),
            alpha=0.0001,
        ),
        AnnCandidateSpec(
            key="relu_128_64",
            label="ReLU 128-64",
            hidden_layer_sizes=(128, 64),
            alpha=0.0001,
        ),
        AnnCandidateSpec(
            key="relu_128_64_regularised",
            label="ReLU 128-64 regularised",
            hidden_layer_sizes=(128, 64),
            alpha=0.01,
        ),
        AnnCandidateSpec(
            key="relu_128_64_funder_frequency",
            label="ReLU 128-64",
            hidden_layer_sizes=(128, 64),
            alpha=0.0001,
            feature_policy="funder_frequency",
        ),
        AnnCandidateSpec(
            key="relu_128_64_funder_identity_frequency",
            label="ReLU 128-64",
            hidden_layer_sizes=(128, 64),
            alpha=0.0001,
            feature_policy="funder_rare20_frequency",
        ),
        AnnCandidateSpec(
            key="relu_128_64_both_identity_frequency",
            label="ReLU 128-64",
            hidden_layer_sizes=(128, 64),
            alpha=0.0001,
            feature_policy="both_rare20_frequency",
        ),
    )
}


def make_ann_pipeline(spec: AnnCandidateSpec) -> _Pipeline:
    """Return scaled fold-fitted preprocessing and one fixed MLP."""

    classifier = _MLPClassifier(
        hidden_layer_sizes=spec.hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=spec.alpha,
        batch_size=spec.batch_size,
        learning_rate="constant",
        learning_rate_init=spec.learning_rate_init,
        max_iter=ANN_MAX_ITERATIONS,
        shuffle=True,
        random_state=spec.seed,
        tol=0.0001,
        early_stopping=True,
        validation_fraction=ANN_VALIDATION_FRACTION,
        n_iter_no_change=ANN_EARLY_STOPPING_PATIENCE,
    )
    if spec.feature_policy == "baseline":
        preprocessor = make_initial_preprocessor(
            sparse_output=True,
            scale_numeric=True,
        )
    else:
        if spec.feature_policy not in HIGH_CARDINALITY_POLICIES:
            raise ValueError(
                f"Unknown ANN feature policy: {spec.feature_policy!r}."
            )
        preprocessor = make_high_cardinality_preprocessor(
            HIGH_CARDINALITY_POLICIES[spec.feature_policy],
            sparse_output=True,
            scale_numeric=True,
        )
    return _Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_ann_candidate(
    spec: AnnCandidateSpec,
    partitioned_data: PartitionedData,
    cross_validation: object,
) -> CandidateEvaluation:
    """Fit one MLP from scratch inside every frozen development fold."""

    probability_values = _np.full(
        (len(partitioned_data.y_development), len(_CLASS_LABELS)),
        fill_value=_np.nan,
    )
    diagnostic_rows = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        X_training = partitioned_data.X_development.iloc[training_positions]
        y_training = partitioned_data.y_development.iloc[training_positions]
        X_validation = partitioned_data.X_development.iloc[validation_positions]

        started = _time.perf_counter()
        pipeline = make_ann_pipeline(spec)
        pipeline.fit(X_training, y_training)
        fit_seconds = _time.perf_counter() - started

        predict_started = _time.perf_counter()
        classifier = pipeline.named_steps["classifier"]
        probabilities = _pd.DataFrame(
            pipeline.predict_proba(X_validation),
            columns=classifier.classes_,
        ).loc[:, list(_CLASS_LABELS)]
        predict_seconds = _time.perf_counter() - predict_started
        probability_values[validation_positions] = probabilities.to_numpy()

        parameter_count = sum(
            values.size
            for values in (*classifier.coefs_, *classifier.intercepts_)
        )
        diagnostic_rows.append(
            {
                "validation_fold": fold_number,
                "iterations": classifier.n_iter_,
                "final_loss": classifier.loss_,
                "best_internal_validation_score": (
                    classifier.best_validation_score_
                ),
                "parameters": parameter_count,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "total_seconds": _time.perf_counter() - started,
            }
        )
        print(
            f"Completed {spec.model_name} fold {fold_number}/"
            f"{CROSS_VALIDATION_FOLDS}: {classifier.n_iter_} epochs in "
            f"{fit_seconds:.1f} seconds."
        )

    return build_candidate_evaluation(
        model_name=spec.model_name,
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
        probability_values=probability_values,
        diagnostic_rows=diagnostic_rows,
    )


def summarise_ann_candidates(
    baseline: CandidateEvaluation,
    candidates: dict[str, CandidateEvaluation],
) -> _pd.DataFrame:
    """Compare standalone MLP candidates with the accepted ensemble."""

    baseline_accuracy = float(baseline.metric_summary.loc["accuracy", "mean"])
    baseline_repair = float(
        baseline.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    rows = []
    for key, candidate in candidates.items():
        fold_change = (
            candidate.fold_metrics["accuracy"]
            - baseline.fold_metrics["accuracy"]
        )
        mean_accuracy = float(candidate.metric_summary.loc["accuracy", "mean"])
        repair_recall = float(
            candidate.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        rows.append(
            {
                "candidate": key,
                "model_name": candidate.model_name,
                "mean_accuracy": mean_accuracy,
                "accuracy_change": mean_accuracy - baseline_accuracy,
                "fold_wins": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall": repair_recall,
                "repair_recall_change": repair_recall - baseline_repair,
                "non_functional_recall": float(
                    candidate.metric_summary.loc[
                        "recall: non functional",
                        "mean",
                    ]
                ),
                "mean_iterations": float(candidate.diagnostics["iterations"].mean()),
                "mean_fit_seconds": float(
                    candidate.diagnostics["fit_seconds"].mean()
                ),
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index("candidate")
        .sort_values(["mean_accuracy", "candidate"], ascending=[False, True])
    )


def evaluate_ann_blends(
    partitioned_data: PartitionedData,
    cross_validation: object,
    baseline: CandidateEvaluation,
    candidates: dict[str, CandidateEvaluation],
    *,
    ann_weights: tuple[float, ...] = ANN_BLEND_WEIGHTS,
) -> dict[tuple[str, float], CandidateEvaluation]:
    """Blend each ANN into the accepted ensemble without model refitting."""

    if not ann_weights or any(weight <= 0 or weight >= 1 for weight in ann_weights):
        raise ValueError("ANN blend weights must be strictly between zero and one.")
    blends = {}
    for key, candidate in candidates.items():
        for ann_weight in ann_weights:
            baseline_weight = 1.0 - ann_weight
            blend = evaluate_weighted_soft_vote(
                partitioned_data,
                cross_validation,
                baseline,
                candidate,
                weights=[baseline_weight, ann_weight],
                model_name=(
                    f"{baseline_weight:.0%} accepted ensemble + "
                    f"{ann_weight:.0%} {candidate.model_name}"
                ),
            )
            blends[(key, ann_weight)] = blend
    return blends


def summarise_ann_blends(
    baseline: CandidateEvaluation,
    blends: dict[tuple[str, float], CandidateEvaluation],
) -> _pd.DataFrame:
    """Compare ANN soft votes with the pre-declared acceptance gate."""

    baseline_accuracy = float(baseline.metric_summary.loc["accuracy", "mean"])
    baseline_repair = float(
        baseline.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
    )
    rows = []
    for (key, ann_weight), blend in blends.items():
        fold_change = blend.fold_metrics["accuracy"] - baseline.fold_metrics[
            "accuracy"
        ]
        mean_accuracy = float(blend.metric_summary.loc["accuracy", "mean"])
        repair_recall = float(
            blend.metric_summary.loc[
                "recall: functional needs repair",
                "mean",
            ]
        )
        accuracy_change = mean_accuracy - baseline_accuracy
        repair_change = repair_recall - baseline_repair
        fold_wins = int(fold_change.gt(0).sum())
        worst_fold_change = float(fold_change.min())
        rows.append(
            {
                "candidate": key,
                "ann_weight": ann_weight,
                "model_name": blend.model_name,
                "mean_accuracy": mean_accuracy,
                "accuracy_change": accuracy_change,
                "fold_wins": fold_wins,
                "worst_fold_change": worst_fold_change,
                "repair_recall": repair_recall,
                "repair_recall_change": repair_change,
                "non_functional_recall": float(
                    blend.metric_summary.loc[
                        "recall: non functional",
                        "mean",
                    ]
                ),
                "passes_gate": (
                    accuracy_change >= SELECTION_GATE_ACCURACY
                    and fold_wins >= SELECTION_GATE_FOLD_WINS
                    and worst_fold_change >= SELECTION_GATE_WORST_FOLD
                    and repair_change >= SELECTION_GATE_REPAIR_RECALL
                ),
            }
        )
    return (
        _pd.DataFrame(rows)
        .set_index(["candidate", "ann_weight"])
        .sort_values(
            ["mean_accuracy", "candidate", "ann_weight"],
            ascending=[False, True, True],
        )
    )


def compare_ann_diversity(
    partitioned_data: PartitionedData,
    baseline: CandidateEvaluation,
    candidates: dict[str, CandidateEvaluation],
) -> _pd.DataFrame:
    """Return pairwise out-of-fold error overlap for the ANN screen."""

    return compare_candidate_diversity(
        partitioned_data,
        baseline,
        *candidates.values(),
    )

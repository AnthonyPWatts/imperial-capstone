"""Evaluate and prepare conservative repair-residual submission candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "repair-residual-screen"
SUBMISSION_DIR = STAGE_DIR / "submissions" / "2026-09-04-final-push"
INCUMBENT_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-08-23-deep-archive-seed-20260824"
    / "01-deep-archive-seed-20260824.csv"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from data_partitioning import partition_modelling_data
from deep_archive_confirmation import blend_deep_archive
from final_model import CLASS_LABELS
from final_model import build_competition_prediction
from final_model import write_validated_submission
from modelling_data import prepare_modelling_data
from repair_residual_specialist import binary_repair_target
from repair_residual_specialist import build_repair_meta_features
from repair_residual_specialist import make_repair_meta_classifier
from repair_residual_specialist import operational_training_mask
from repair_residual_specialist import overlay_repair_probabilities
from repair_residual_specialist import repair_history_mask


CANDIDATE_FILES = {
    "repair_meta_plus_history": "01-repair-meta-070-plus-subvillage-history.csv",
    "repair_meta": "02-repair-meta-070.csv",
    "subvillage_history": "03-subvillage-history.csv",
}


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = partition_modelling_data(modelling_data)
    cross_validation = make_cross_validation(partitioned)

    oof_components = _load_oof_components(partitioned)
    oof_base = blend_deep_archive(oof_components)
    oof_features = build_repair_meta_features(oof_components, oof_base)
    local_components, local_base = _load_local_components(partitioned)
    local_features = build_repair_meta_features(local_components, local_base)
    competition_components = _load_competition_components(modelling_data)
    competition_base = blend_deep_archive(competition_components)
    competition_features = build_repair_meta_features(
        competition_components,
        competition_base,
    )
    _validate_incumbent_reconstruction(competition_base)

    oof_repair = _cross_fitted_repair_probabilities(
        oof_features,
        partitioned.y_development,
        cross_validation,
    )
    development_model = _fit_meta_model(
        oof_features,
        partitioned.y_development,
    )
    local_repair = development_model.predict_proba(local_features)[:, 1]
    competition_repair = _fit_final_meta_model(
        oof_features,
        partitioned.y_development,
        local_features,
        partitioned.y_local_test,
    ).predict_proba(competition_features)[:, 1]

    oof_history = _cross_fitted_history_mask(partitioned, cross_validation)
    local_history = repair_history_mask(
        partitioned.X_development,
        partitioned.y_development,
        partitioned.X_local_test,
    )
    competition_history = repair_history_mask(
        modelling_data.X_original,
        modelling_data.y_original,
        modelling_data.X_competition,
    )

    oof_candidates = _build_candidates(oof_base, oof_repair, oof_history)
    local_candidates = _build_candidates(
        local_base,
        local_repair,
        local_history,
    )
    competition_candidates = _build_candidates(
        competition_base,
        competition_repair,
        competition_history,
    )
    summary = _summarise_evidence(
        partitioned,
        cross_validation,
        oof_base,
        oof_candidates,
        local_base,
        local_candidates,
    )
    summary.to_csv(RUNTIME_DIR / "candidate-summary.csv")

    prediction_records = _write_competition_candidates(
        modelling_data,
        competition_candidates,
    )
    payload = {
        "development_ids": partitioned.development_ids.copy(),
        "local_test_ids": partitioned.local_test_ids.copy(),
        "competition_ids": modelling_data.competition_ids.copy(),
        "oof_repair_probabilities": oof_repair,
        "local_repair_probabilities": local_repair,
        "competition_repair_probabilities": competition_repair,
        "oof_history_mask": oof_history,
        "local_history_mask": local_history,
        "competition_history_mask": competition_history,
    }
    joblib.dump(payload, RUNTIME_DIR / "repair-residual-evidence.joblib")
    result = _write_manifests(summary, prediction_records)
    print(_format_summary(summary), flush=True)
    print(json.dumps(result, indent=2), flush=True)


def _load_oof_components(partitioned) -> dict[str, np.ndarray]:
    baseline = joblib.load(
        PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
    )
    deep = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "archived-deep-xgboost-screen"
        / "depth-17-seed-20260824.joblib"
    )
    frequency = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "categorical-frequency-forest-screen"
        / "all-categorical-occurrence-counts.joblib"
    ).random_forest
    spatial = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "spatial-height-imputation-screen"
        / "ten-neighbour-height.joblib"
    )
    identity = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "catboost-identity-screen"
        / "complete-deferred-identities.joblib"
    ).evaluation
    evaluations = {
        "deep_xgboost": deep,
        "spatial_xgboost": spatial.xgboost,
        "random_forest": baseline.random_forest,
        "frequency_random_forest": frequency,
        "spatial_random_forest": spatial.random_forest,
        "identity_catboost": identity,
    }
    for name, evaluation in evaluations.items():
        if (
            evaluation.cross_validation_fingerprint
            != partitioned.cross_validation_fingerprint
        ):
            raise ValueError(f"OOF component {name!r} uses different folds.")
    return {
        name: evaluation.out_of_fold_probabilities.to_numpy(dtype="float64")
        for name, evaluation in evaluations.items()
    }


def _load_local_components(partitioned) -> tuple[dict[str, np.ndarray], np.ndarray]:
    archive_payload = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "deep-archive-confirmation"
        / "local-test-confirmation.joblib"
    )
    seed_payload = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "deep-seed-20260824-confirmation"
        / "local-test-confirmation.joblib"
    )
    for name, payload in (("archive", archive_payload), ("seed", seed_payload)):
        if payload["local_test_fingerprint"] != partitioned.local_test_fingerprint:
            raise ValueError(f"Local {name} payload uses a different test set.")
        if not payload["local_test_ids"].reset_index(drop=True).equals(
            partitioned.local_test_ids.reset_index(drop=True)
        ):
            raise ValueError(f"Local {name} payload changed ID order.")
    archived = archive_payload["confirmation"].probabilities
    components = {
        "deep_xgboost": seed_payload["deep_xgboost"],
        "spatial_xgboost": archived["spatial_xgboost"],
        "random_forest": archived["random_forest"],
        "frequency_random_forest": archived["frequency_random_forest"],
        "spatial_random_forest": archived["spatial_random_forest"],
        "identity_catboost": archived["identity_catboost"],
    }
    base = blend_deep_archive(components)
    if not np.allclose(base, seed_payload["candidate"]):
        raise ValueError("Local deep-archive reconstruction changed.")
    return components, base


def _load_competition_components(modelling_data) -> dict[str, np.ndarray]:
    accepted = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "class-membership-analysis"
        / "accepted-competition-probabilities.joblib"
    )
    identity = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "catboost-identity-competition"
        / "complete-identity-catboost.joblib"
    )
    archive = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "archive-synthesis-competition"
        / "archive-components.joblib"
    )
    deep = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "deep-archive-seed-20260824-competition"
        / "deep-xgboost.joblib"
    )
    for name, payload in (
        ("accepted", accepted),
        ("identity", identity),
        ("archive", archive),
        ("deep", deep),
    ):
        if not payload["competition_ids"].reset_index(drop=True).equals(
            modelling_data.competition_ids.reset_index(drop=True)
        ):
            raise ValueError(f"Competition {name} payload changed ID order.")
    return {
        "deep_xgboost": deep["probabilities"],
        "spatial_xgboost": archive["spatial_xgboost"],
        "random_forest": accepted["random_forest_probabilities"],
        "frequency_random_forest": archive["frequency_random_forest"],
        "spatial_random_forest": archive["spatial_random_forest"],
        "identity_catboost": identity["probabilities"],
    }


def _validate_incumbent_reconstruction(probabilities: np.ndarray) -> None:
    incumbent = pd.read_csv(INCUMBENT_PATH)
    labels = np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]
    if not np.array_equal(labels, incumbent["status_group"].to_numpy()):
        raise ValueError("Cached probabilities do not recreate the 0.8298 CSV.")


def _cross_fitted_repair_probabilities(
    features: np.ndarray,
    target: pd.Series,
    cross_validation,
) -> np.ndarray:
    probabilities = np.full(len(target), np.nan, dtype="float64")
    values = target.to_numpy()
    for training_positions, validation_positions in cross_validation.split():
        training_target = values[training_positions]
        operational = operational_training_mask(training_target)
        model = make_repair_meta_classifier()
        model.fit(
            features[training_positions][operational],
            binary_repair_target(training_target[operational]),
        )
        probabilities[validation_positions] = model.predict_proba(
            features[validation_positions]
        )[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("Cross-fitted repair probabilities are incomplete.")
    return probabilities


def _fit_meta_model(features: np.ndarray, target: pd.Series | np.ndarray):
    values = np.asarray(target, dtype=object)
    operational = operational_training_mask(values)
    model = make_repair_meta_classifier()
    model.fit(
        features[operational],
        binary_repair_target(values[operational]),
    )
    return model


def _fit_final_meta_model(
    oof_features: np.ndarray,
    oof_target: pd.Series,
    local_features: np.ndarray,
    local_target: pd.Series,
):
    features = np.vstack([oof_features, local_features])
    target = np.concatenate([oof_target.to_numpy(), local_target.to_numpy()])
    return _fit_meta_model(features, target)


def _cross_fitted_history_mask(partitioned, cross_validation) -> np.ndarray:
    selected = np.zeros(len(partitioned.y_development), dtype=bool)
    for training_positions, validation_positions in cross_validation.split():
        selected[validation_positions] = repair_history_mask(
            partitioned.X_development.iloc[training_positions],
            partitioned.y_development.iloc[training_positions],
            partitioned.X_development.iloc[validation_positions],
        )
    return selected


def _build_candidates(
    base: np.ndarray,
    repair_probabilities: np.ndarray,
    history_mask: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    zero_probabilities = np.zeros(len(base), dtype="float64")
    meta, meta_mask = overlay_repair_probabilities(base, repair_probabilities)
    history, selected_history = overlay_repair_probabilities(
        base,
        zero_probabilities,
        threshold=1.0,
        additional_mask=history_mask,
    )
    combined, combined_mask = overlay_repair_probabilities(
        base,
        repair_probabilities,
        additional_mask=history_mask,
    )
    return {
        "repair_meta_plus_history": (combined, combined_mask),
        "repair_meta": (meta, meta_mask),
        "subvillage_history": (history, selected_history),
    }


def _summarise_evidence(
    partitioned,
    cross_validation,
    oof_base: np.ndarray,
    oof_candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    local_base: np.ndarray,
    local_candidates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows = []
    labels = np.asarray(CLASS_LABELS)
    oof_target = partitioned.y_development.to_numpy()
    local_target = partitioned.y_local_test.to_numpy()
    base_oof_labels = labels[oof_base.argmax(axis=1)]
    base_local_labels = labels[local_base.argmax(axis=1)]
    base_oof_accuracy = accuracy_score(oof_target, base_oof_labels)
    base_local_accuracy = accuracy_score(local_target, base_local_labels)
    folds = list(cross_validation.split())
    for name in CANDIDATE_FILES:
        oof, oof_mask = oof_candidates[name]
        local, local_mask = local_candidates[name]
        oof_labels = labels[oof.argmax(axis=1)]
        local_labels = labels[local.argmax(axis=1)]
        fold_changes = [
            accuracy_score(oof_target[validation], oof_labels[validation])
            - accuracy_score(
                oof_target[validation],
                base_oof_labels[validation],
            )
            for _, validation in folds
        ]
        row = {
            "candidate": name,
            "development_accuracy": accuracy_score(oof_target, oof_labels),
            "development_change": (
                accuracy_score(oof_target, oof_labels) - base_oof_accuracy
            ),
            "development_net_correct": int(
                (oof_labels == oof_target).sum()
                - (base_oof_labels == oof_target).sum()
            ),
            "development_repair_recall": recall_score(
                oof_target,
                oof_labels,
                labels=["functional needs repair"],
                average=None,
                zero_division=0,
            )[0],
            "development_repair_precision": precision_score(
                oof_target,
                oof_labels,
                labels=["functional needs repair"],
                average=None,
                zero_division=0,
            )[0],
            "development_flips": int(oof_mask.sum()),
            "fold_wins": int(sum(change > 0 for change in fold_changes)),
            "worst_fold_change": float(min(fold_changes)),
            "local_accuracy": accuracy_score(local_target, local_labels),
            "local_change": (
                accuracy_score(local_target, local_labels) - base_local_accuracy
            ),
            "local_net_correct": int(
                (local_labels == local_target).sum()
                - (base_local_labels == local_target).sum()
            ),
            "local_repair_recall": recall_score(
                local_target,
                local_labels,
                labels=["functional needs repair"],
                average=None,
                zero_division=0,
            )[0],
            "local_repair_precision": precision_score(
                local_target,
                local_labels,
                labels=["functional needs repair"],
                average=None,
                zero_division=0,
            )[0],
            "local_flips": int(local_mask.sum()),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("candidate")


def _write_competition_candidates(
    modelling_data,
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, dict[str, object]]:
    template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")
    incumbent = pd.read_csv(INCUMBENT_PATH)
    records = {}
    for name, filename in CANDIDATE_FILES.items():
        probabilities, selected = candidates[name]
        prediction = build_competition_prediction(
            modelling_data,
            template,
            probabilities,
            pd.Series(dtype="float64", name="fit and predict seconds"),
        )
        destination = write_validated_submission(
            prediction,
            SUBMISSION_DIR / filename,
        )
        submission = pd.read_csv(destination)
        changed = submission["status_group"].ne(incumbent["status_group"])
        if int(changed.sum()) != int(selected.sum()):
            raise ValueError(f"Candidate {name!r} changed an unexpected row count.")
        transitions = (
            pd.crosstab(
                incumbent.loc[changed, "status_group"],
                submission.loc[changed, "status_group"],
            )
            .stack(future_stack=True)
            .loc[lambda values: values.gt(0)]
        )
        records[name] = {
            "csv": filename,
            "rows": len(submission),
            "unique_ids": int(submission["id"].nunique()),
            "changed_vs_incumbent": int(changed.sum()),
            "transitions": {
                f"{source} -> {destination_label}": int(value)
                for (source, destination_label), value in transitions.items()
            },
            "class_counts": {
                label: int(value)
                for label, value in submission["status_group"]
                .value_counts()
                .reindex(CLASS_LABELS, fill_value=0)
                .items()
            },
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    return records


def _write_manifests(
    summary: pd.DataFrame,
    prediction_records: dict[str, dict[str, object]],
) -> dict[str, object]:
    entries = []
    for order, name in enumerate(CANDIDATE_FILES, start=1):
        evidence = {
            key: int(value) if key.endswith(("_correct", "_flips")) else float(value)
            for key, value in summary.loc[name].items()
        }
        entries.append(
            {
                "order": order,
                "key": name,
                **prediction_records[name],
                "evidence": evidence,
                "public_score": None,
                "submission_id": None,
            }
        )
    result = {
        "date": "2026-09-04",
        "status": "prepared_not_uploaded",
        "incumbent_public_score": 0.8298,
        "repair_meta": {
            "model": "standardised logistic regression",
            "C": 0.10,
            "threshold": 0.70,
            "training_classes": ["functional", "functional needs repair"],
            "input_features": 26,
        },
        "repair_history": {
            "column": "normalised subvillage",
            "minimum_functional_plus_repair_support": 20,
            "laplace_smoothed_minimum_repair_rate": 0.40,
        },
        "entries": entries,
    }
    (RUNTIME_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (SUBMISSION_DIR / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in display.columns:
        if any(
            token in column
            for token in ("accuracy", "change", "recall", "precision")
        ):
            display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


if __name__ == "__main__":
    main()

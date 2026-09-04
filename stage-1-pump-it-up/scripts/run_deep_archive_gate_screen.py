"""Screen a repair-preserving gate between archive and deep decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "deep-archive-gate-screen"
SUBMISSION_DIR = STAGE_DIR / "submissions" / "2026-09-04-deep-archive-gate"
CANDIDATE_NAME = "01-repair-preserving-archive-gate.csv"
UNION_CANDIDATE_NAME = "02-gate-plus-strict-repair-core.csv"
INCUMBENT_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-08-23-deep-archive-seed-20260824"
    / "01-deep-archive-seed-20260824.csv"
)
ARCHIVE_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-08-22-archive-synthesis"
    / (
        "01-33-xgboost-11-spatial-xgboost-18-random-forest-"
        "09-frequency-forest-09-spatial-forest-20-identity-catboost.csv"
    )
)
STRICT_REPAIR_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-09-04-repair-rule-ensemble"
    / "03-strict-history-core-plus-residual-history.csv"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import partition_modelling_data
from deep_archive_gate import ARCHIVE_GATE_THRESHOLD
from deep_archive_gate import apply_repair_preserving_archive_gate
from deep_archive_gate import archive_choice_target
from deep_archive_gate import build_archive_and_deep_probabilities
from deep_archive_gate import build_archive_gate_features
from deep_archive_gate import make_archive_gate_classifier
from final_model import CLASS_LABELS
from final_model import build_competition_prediction
from final_model import validate_probabilities
from final_model import write_validated_submission
from modelling_data import prepare_modelling_data
from repair_residual_specialist import overlay_repair_probabilities


def main() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = partition_modelling_data(modelling_data)

    oof_components = _load_oof_components(partitioned)
    local_components = _load_local_components(partitioned)
    competition_components = _load_competition_components(modelling_data)
    strict_repair_masks = _load_strict_repair_masks(
        partitioned,
        modelling_data,
    )
    oof_archive, oof_deep = build_archive_and_deep_probabilities(
        oof_components
    )
    local_archive, local_deep = build_archive_and_deep_probabilities(
        local_components
    )
    competition_archive, competition_deep = (
        build_archive_and_deep_probabilities(competition_components)
    )
    _validate_reconstructions(competition_archive, competition_deep)

    oof_features = build_archive_gate_features(
        oof_components,
        oof_archive,
        oof_deep,
    )
    local_features = build_archive_gate_features(
        local_components,
        local_archive,
        local_deep,
    )
    competition_features = build_archive_gate_features(
        competition_components,
        competition_archive,
        competition_deep,
    )

    oof_choice = _cross_fitted_choice_probabilities(
        oof_features,
        partitioned.y_development,
        partitioned.validation_folds,
    )
    development_model = _fit_choice_model(
        [(oof_features, partitioned.y_development)]
    )
    local_choice = development_model.predict_proba(local_features.values)[:, 1]
    final_model = _fit_choice_model(
        [
            (oof_features, partitioned.y_development),
            (local_features, partitioned.y_local_test),
        ]
    )
    competition_choice = final_model.predict_proba(
        competition_features.values
    )[:, 1]

    oof_candidate, oof_selected = apply_repair_preserving_archive_gate(
        oof_archive,
        oof_deep,
        oof_features,
        oof_choice,
    )
    local_candidate, local_selected = apply_repair_preserving_archive_gate(
        local_archive,
        local_deep,
        local_features,
        local_choice,
    )
    competition_candidate, competition_selected = (
        apply_repair_preserving_archive_gate(
            competition_archive,
            competition_deep,
            competition_features,
            competition_choice,
        )
    )
    oof_union = _apply_strict_repair_after_gate(
        oof_deep,
        oof_candidate,
        strict_repair_masks["development"],
    )
    local_union = _apply_strict_repair_after_gate(
        local_deep,
        local_candidate,
        strict_repair_masks["local_test"],
    )
    competition_union = _apply_strict_repair_after_gate(
        competition_deep,
        competition_candidate,
        strict_repair_masks["competition"],
    )

    summary = _summarise_evidence(
        partitioned,
        oof_archive,
        oof_deep,
        oof_candidate,
        oof_selected,
        local_archive,
        local_deep,
        local_candidate,
        local_selected,
        oof_union,
        strict_repair_masks["development"],
        local_union,
        strict_repair_masks["local_test"],
    )
    summary.to_csv(RUNTIME_DIR / "evidence-summary.csv")
    oof_audit = _build_reversion_audit(
        partitioned.development_ids,
        oof_features,
        oof_choice,
        oof_selected,
        target=partitioned.y_development,
        folds=partitioned.validation_folds,
    )
    local_audit = _build_reversion_audit(
        partitioned.local_test_ids,
        local_features,
        local_choice,
        local_selected,
        target=partitioned.y_local_test,
    )
    competition_audit = _build_reversion_audit(
        modelling_data.competition_ids,
        competition_features,
        competition_choice,
        competition_selected,
    )
    oof_audit.to_csv(RUNTIME_DIR / "oof-reversions.csv", index=False)
    local_audit.to_csv(RUNTIME_DIR / "local-reversions.csv", index=False)
    competition_audit.to_csv(
        RUNTIME_DIR / "competition-reversions.csv",
        index=False,
    )

    template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")
    prediction = build_competition_prediction(
        modelling_data,
        template,
        competition_candidate,
        pd.Series({"archive-choice gate": 0.0}, name="seconds"),
    )
    destination = write_validated_submission(
        prediction,
        SUBMISSION_DIR / CANDIDATE_NAME,
    )
    reloaded = pd.read_csv(destination)
    incumbent = pd.read_csv(INCUMBENT_PATH)
    if not reloaded["id"].equals(incumbent["id"]):
        raise ValueError("Archive-gate candidate changed incumbent ID order.")
    hard_changes = reloaded["status_group"].ne(incumbent["status_group"])
    if int(hard_changes.sum()) != int(competition_selected.sum()):
        raise ValueError("Archive-gate selection and CSV changes disagree.")
    union_prediction = build_competition_prediction(
        modelling_data,
        template,
        competition_union,
        pd.Series(
            {"archive-choice gate plus strict repair": 0.0},
            name="seconds",
        ),
    )
    union_destination = write_validated_submission(
        union_prediction,
        SUBMISSION_DIR / UNION_CANDIDATE_NAME,
    )
    union_reloaded = pd.read_csv(union_destination)
    union_changes = union_reloaded["status_group"].ne(
        incumbent["status_group"]
    )
    expected_union_changes = competition_selected | strict_repair_masks[
        "competition"
    ]
    if not np.array_equal(union_changes.to_numpy(), expected_union_changes):
        raise ValueError("Union CSV changes do not match its two fixed masks.")
    _validate_strict_repair_composition(
        incumbent,
        reloaded,
        union_reloaded,
        strict_repair_masks["competition"],
    )

    evidence = {
        "development": _evidence_record(
            partitioned.y_development,
            oof_deep,
            oof_candidate,
            oof_selected,
            oof_features,
        ),
        "local_test": _evidence_record(
            partitioned.y_local_test,
            local_deep,
            local_candidate,
            local_selected,
            local_features,
        ),
    }
    union_evidence = {
        "development": _paired_candidate_evidence(
            partitioned.y_development,
            oof_deep,
            oof_union,
        ),
        "local_test": _paired_candidate_evidence(
            partitioned.y_local_test,
            local_deep,
            local_union,
        ),
    }
    overlap_counts = {
        "development": int(
            np.sum(oof_selected & strict_repair_masks["development"])
        ),
        "local_test": int(
            np.sum(local_selected & strict_repair_masks["local_test"])
        ),
        "competition": int(
            np.sum(
                competition_selected & strict_repair_masks["competition"]
            )
        ),
    }
    manifest = {
        "date": "2026-09-04",
        "status": "prepared_not_uploaded",
        "selection": {
            "gate": "cross-fitted logistic archive choice",
            "logistic_C": 1.0,
            "threshold": ARCHIVE_GATE_THRESHOLD,
            "constraint": "never replace a deep repair prediction",
            **evidence,
        },
        "competition": {
            "archive_deep_disagreements": len(
                competition_features.row_positions
            ),
            "selected_reversions": int(competition_selected.sum()),
            "prediction_share_changed": float(hard_changes.mean()),
            "repair_prediction_change": int(
                reloaded["status_group"]
                .eq("functional needs repair")
                .sum()
                - incumbent["status_group"]
                .eq("functional needs repair")
                .sum()
            ),
        },
        "csv": destination.name,
        "rows": len(reloaded),
        "unique_ids": int(reloaded["id"].nunique()),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "union_candidate": {
            "composition": (
                "repair-preserving archive gate plus strict history core "
                "and residual history"
            ),
            "strict_repair_precedence": True,
            "mask_overlap_counts": overlap_counts,
            "selection": union_evidence,
            "competition": {
                "selected_gate_reversions": int(competition_selected.sum()),
                "selected_strict_repairs": int(
                    strict_repair_masks["competition"].sum()
                ),
                "changed_vs_incumbent": int(union_changes.sum()),
                "prediction_share_changed": float(union_changes.mean()),
                "repair_prediction_change": int(
                    union_reloaded["status_group"]
                    .eq("functional needs repair")
                    .sum()
                    - incumbent["status_group"]
                    .eq("functional needs repair")
                    .sum()
                ),
                "class_counts": {
                    label: int(union_reloaded["status_group"].eq(label).sum())
                    for label in CLASS_LABELS
                },
            },
            "csv": union_destination.name,
            "rows": len(union_reloaded),
            "unique_ids": int(union_reloaded["id"].nunique()),
            "sha256": hashlib.sha256(
                union_destination.read_bytes()
            ).hexdigest(),
        },
    }
    (SUBMISSION_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    result = {
        **manifest,
        "runtime_audits": {
            "oof": "oof-reversions.csv",
            "local_test": "local-reversions.csv",
            "competition": "competition-reversions.csv",
        },
    }
    (RUNTIME_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "development_ids": partitioned.development_ids.copy(),
            "local_test_ids": partitioned.local_test_ids.copy(),
            "competition_ids": modelling_data.competition_ids.copy(),
            "oof_choice_probabilities": oof_choice,
            "local_choice_probabilities": local_choice,
            "competition_choice_probabilities": competition_choice,
            "oof_selected": oof_selected,
            "local_selected": local_selected,
            "competition_selected": competition_selected,
            "strict_repair_masks": strict_repair_masks,
        },
        RUNTIME_DIR / "gate-evidence.joblib",
    )
    print(_format_summary(summary), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


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
        "accepted_xgboost": baseline.xgboost,
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


def _load_local_components(partitioned) -> dict[str, np.ndarray]:
    archive_payload = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "archive-synthesis-confirmation"
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
    archive = archive_payload["confirmation"].probabilities
    components = {
        "accepted_xgboost": archive["xgboost"],
        "deep_xgboost": seed_payload["deep_xgboost"],
        "spatial_xgboost": archive["spatial_xgboost"],
        "random_forest": archive["random_forest"],
        "frequency_random_forest": archive["frequency_random_forest"],
        "spatial_random_forest": archive["spatial_random_forest"],
        "identity_catboost": archive["identity_catboost"],
    }
    _, deep = build_archive_and_deep_probabilities(components)
    if not np.allclose(deep, seed_payload["candidate"]):
        raise ValueError("Local deep-archive reconstruction changed.")
    return components


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
        "accepted_xgboost": accepted["xgboost_probabilities"],
        "deep_xgboost": deep["probabilities"],
        "spatial_xgboost": archive["spatial_xgboost"],
        "random_forest": accepted["random_forest_probabilities"],
        "frequency_random_forest": archive["frequency_random_forest"],
        "spatial_random_forest": archive["spatial_random_forest"],
        "identity_catboost": identity["probabilities"],
    }


def _load_strict_repair_masks(partitioned, modelling_data) -> dict[str, np.ndarray]:
    payload = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "repair-rule-ensemble"
        / "repair-rule-evidence.joblib"
    )
    expected_ids = {
        "development_ids": partitioned.development_ids,
        "local_test_ids": partitioned.local_test_ids,
        "competition_ids": modelling_data.competition_ids,
    }
    for key, expected in expected_ids.items():
        if not payload[key].reset_index(drop=True).equals(
            expected.reset_index(drop=True)
        ):
            raise ValueError(f"Strict repair payload changed {key!r} order.")
    candidate_key = "strict_history_core_plus_residual_history"
    masks = {
        "development": np.asarray(
            payload["development_candidates"][candidate_key],
            dtype=bool,
        ),
        "local_test": np.asarray(
            payload["local_candidates"][candidate_key],
            dtype=bool,
        ),
        "competition": np.asarray(
            payload["competition_candidates"][candidate_key],
            dtype=bool,
        ),
    }
    expected_lengths = {
        "development": len(partitioned.y_development),
        "local_test": len(partitioned.y_local_test),
        "competition": len(modelling_data.X_competition),
    }
    for key, mask in masks.items():
        if mask.shape != (expected_lengths[key],):
            raise ValueError(f"Strict repair {key} mask has the wrong shape.")
    return masks


def _apply_strict_repair_after_gate(
    deep_probabilities: np.ndarray,
    gated_probabilities: np.ndarray,
    strict_repair_mask: np.ndarray,
) -> np.ndarray:
    base = np.asarray(deep_probabilities, dtype="float64")
    gated = np.asarray(gated_probabilities, dtype="float64")
    mask = np.asarray(strict_repair_mask, dtype=bool)
    validate_probabilities(base, len(base))
    validate_probabilities(gated, len(base))
    if mask.shape != (len(base),):
        raise ValueError("Strict repair mask has the wrong shape.")
    repaired, selected = overlay_repair_probabilities(
        base,
        np.zeros(len(base), dtype="float64"),
        threshold=1.0,
        additional_mask=mask,
    )
    if not np.array_equal(selected, mask):
        raise ValueError("Strict repair mask contains a non-functional base row.")
    combined = gated.copy()
    combined[mask] = repaired[mask]
    validate_probabilities(combined, len(base))
    if not np.all(combined[mask].argmax(axis=1) == 1):
        raise ValueError("Strict repair did not take precedence in the union.")
    return combined


def _validate_reconstructions(archive: np.ndarray, deep: np.ndarray) -> None:
    labels = np.asarray(CLASS_LABELS)
    expected_archive = pd.read_csv(ARCHIVE_PATH)
    expected_deep = pd.read_csv(INCUMBENT_PATH)
    if not np.array_equal(
        labels[archive.argmax(axis=1)],
        expected_archive["status_group"].to_numpy(),
    ):
        raise ValueError("Cached probabilities do not recreate archive CSV.")
    if not np.array_equal(
        labels[deep.argmax(axis=1)],
        expected_deep["status_group"].to_numpy(),
    ):
        raise ValueError("Cached probabilities do not recreate 0.8298 CSV.")


def _validate_strict_repair_composition(
    incumbent: pd.DataFrame,
    gate: pd.DataFrame,
    union: pd.DataFrame,
    strict_repair_mask: np.ndarray,
) -> None:
    strict = pd.read_csv(STRICT_REPAIR_PATH)
    for name, candidate in (("gate", gate), ("union", union), ("strict", strict)):
        if not candidate["id"].equals(incumbent["id"]):
            raise ValueError(f"{name.title()} candidate changed incumbent IDs.")
    strict_changes = strict["status_group"].ne(incumbent["status_group"])
    if not np.array_equal(strict_changes.to_numpy(), strict_repair_mask):
        raise ValueError("Committed strict repair CSV and cached mask disagree.")
    if not strict.loc[
        strict_repair_mask,
        "status_group",
    ].eq("functional needs repair").all():
        raise ValueError("Strict repair CSV contains a non-repair override.")
    expected = gate["status_group"].copy()
    expected.loc[strict_repair_mask] = strict.loc[
        strict_repair_mask,
        "status_group",
    ]
    if not expected.equals(union["status_group"]):
        raise ValueError("Union does not apply strict repair after the gate.")


def _cross_fitted_choice_probabilities(features, target, folds) -> np.ndarray:
    relevant, archive_correct = archive_choice_target(target, features)
    disagreement_folds = folds.to_numpy()[features.row_positions]
    probabilities = np.full(len(features.row_positions), np.nan, dtype="float64")
    for fold_number in sorted(np.unique(disagreement_folds)):
        training = (disagreement_folds != fold_number) & relevant
        validation = disagreement_folds == fold_number
        model = make_archive_gate_classifier()
        model.fit(features.values.loc[training], archive_correct[training])
        probabilities[validation] = model.predict_proba(
            features.values.loc[validation]
        )[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("Cross-fitted archive-choice probabilities are incomplete.")
    return probabilities


def _fit_choice_model(evidence):
    training_frames = []
    training_targets = []
    for features, target in evidence:
        relevant, archive_correct = archive_choice_target(target, features)
        training_frames.append(features.values.loc[relevant])
        training_targets.append(archive_correct[relevant])
    model = make_archive_gate_classifier()
    model.fit(
        pd.concat(training_frames, ignore_index=True),
        np.concatenate(training_targets),
    )
    return model


def _summarise_evidence(
    partitioned,
    oof_archive,
    oof_deep,
    oof_candidate,
    oof_selected,
    local_archive,
    local_deep,
    local_candidate,
    local_selected,
    oof_union,
    oof_strict_repair,
    local_union,
    local_strict_repair,
) -> pd.DataFrame:
    rows = []
    folds = partitioned.validation_folds.to_numpy()
    for name, target, archive, deep, candidate, selected, row_folds in (
        (
            "development",
            partitioned.y_development,
            oof_archive,
            oof_deep,
            oof_candidate,
            oof_selected,
            folds,
        ),
        (
            "local_test",
            partitioned.y_local_test,
            local_archive,
            local_deep,
            local_candidate,
            local_selected,
            None,
        ),
        (
            "development_union",
            partitioned.y_development,
            oof_archive,
            oof_deep,
            oof_union,
            oof_selected | oof_strict_repair,
            folds,
        ),
        (
            "local_test_union",
            partitioned.y_local_test,
            local_archive,
            local_deep,
            local_union,
            local_selected | local_strict_repair,
            None,
        ),
    ):
        values = target.to_numpy()
        labels = np.asarray(CLASS_LABELS)
        deep_labels = labels[deep.argmax(axis=1)]
        candidate_labels = labels[candidate.argmax(axis=1)]
        recalls_deep = recall_score(
            values,
            deep_labels,
            labels=list(CLASS_LABELS),
            average=None,
            zero_division=0,
        )
        recalls_candidate = recall_score(
            values,
            candidate_labels,
            labels=list(CLASS_LABELS),
            average=None,
            zero_division=0,
        )
        row = {
            "evidence": name,
            "rows": len(values),
            "archive_deep_disagreements": int(
                np.sum(archive.argmax(axis=1) != deep.argmax(axis=1))
            ),
            "selected_reversions": int(selected.sum()),
            "deep_accuracy": accuracy_score(values, deep_labels),
            "gated_accuracy": accuracy_score(values, candidate_labels),
            "net_additional_correct": int(
                np.sum(candidate_labels == values) - np.sum(deep_labels == values)
            ),
            "functional_recall_change": float(
                recalls_candidate[0] - recalls_deep[0]
            ),
            "repair_recall_change": float(
                recalls_candidate[1] - recalls_deep[1]
            ),
            "non_functional_recall_change": float(
                recalls_candidate[2] - recalls_deep[2]
            ),
            "candidate_only_correct_selected": int(
                np.sum(selected & (candidate_labels == values))
            ),
            "deep_only_correct_selected": int(
                np.sum(selected & (deep_labels == values))
            ),
        }
        row["third_class_selected"] = (
            row["selected_reversions"]
            - row["candidate_only_correct_selected"]
            - row["deep_only_correct_selected"]
        )
        if row_folds is not None:
            for fold_number in range(1, 6):
                fold = row_folds == fold_number
                row[f"fold_{fold_number}_net"] = int(
                    np.sum(fold & (candidate_labels == values))
                    - np.sum(fold & (deep_labels == values))
                )
        rows.append(row)
    return pd.DataFrame(rows).set_index("evidence")


def _evidence_record(target, deep, candidate, selected, features) -> dict:
    labels = np.asarray(CLASS_LABELS)
    values = target.to_numpy()
    deep_labels = labels[deep.argmax(axis=1)]
    candidate_labels = labels[candidate.argmax(axis=1)]
    archive_labels = deep_labels.copy()
    archive_labels[features.row_positions] = labels[features.archive_positions]
    archive_only = int(np.sum(selected & (archive_labels == values)))
    deep_only = int(np.sum(selected & (deep_labels == values)))
    return {
        "archive_deep_disagreements": len(features.row_positions),
        "selected_reversions": int(selected.sum()),
        "net_additional_correct": int(
            np.sum(candidate_labels == values) - np.sum(deep_labels == values)
        ),
        "archive_only_correct_selected": archive_only,
        "deep_only_correct_selected": deep_only,
        "third_class_selected": int(selected.sum()) - archive_only - deep_only,
        "deep_accuracy": float(accuracy_score(values, deep_labels)),
        "gated_accuracy": float(accuracy_score(values, candidate_labels)),
    }


def _paired_candidate_evidence(target, deep, candidate) -> dict:
    labels = np.asarray(CLASS_LABELS)
    values = target.to_numpy()
    deep_labels = labels[deep.argmax(axis=1)]
    candidate_labels = labels[candidate.argmax(axis=1)]
    changed = candidate_labels != deep_labels
    candidate_only = int(np.sum(changed & (candidate_labels == values)))
    deep_only = int(np.sum(changed & (deep_labels == values)))
    recalls_deep = recall_score(
        values,
        deep_labels,
        labels=list(CLASS_LABELS),
        average=None,
        zero_division=0,
    )
    recalls_candidate = recall_score(
        values,
        candidate_labels,
        labels=list(CLASS_LABELS),
        average=None,
        zero_division=0,
    )
    return {
        "changed_vs_deep": int(changed.sum()),
        "net_additional_correct": candidate_only - deep_only,
        "candidate_only_correct": candidate_only,
        "deep_only_correct": deep_only,
        "third_class": int(changed.sum()) - candidate_only - deep_only,
        "deep_accuracy": float(accuracy_score(values, deep_labels)),
        "candidate_accuracy": float(accuracy_score(values, candidate_labels)),
        "functional_recall_change": float(
            recalls_candidate[0] - recalls_deep[0]
        ),
        "repair_recall_change": float(recalls_candidate[1] - recalls_deep[1]),
        "non_functional_recall_change": float(
            recalls_candidate[2] - recalls_deep[2]
        ),
    }


def _build_reversion_audit(
    ids,
    features,
    choice_probabilities,
    selected,
    *,
    target=None,
    folds=None,
) -> pd.DataFrame:
    labels = np.asarray(CLASS_LABELS)
    selected_disagreements = selected[features.row_positions]
    row_positions = features.row_positions[selected_disagreements]
    audit = pd.DataFrame(
        {
            "id": ids.to_numpy()[row_positions],
            "row_position": row_positions,
            "archive_prediction": labels[
                features.archive_positions[selected_disagreements]
            ],
            "deep_prediction": labels[features.deep_positions[selected_disagreements]],
            "transition": features.values.loc[
                selected_disagreements,
                "transition",
            ].to_numpy(),
            "archive_choice_probability": choice_probabilities[
                selected_disagreements
            ],
        }
    )
    if target is not None:
        actual = target.to_numpy()[row_positions]
        audit["actual"] = actual
        audit["outcome"] = np.where(
            actual == audit["archive_prediction"],
            "archive_only_correct",
            np.where(
                actual == audit["deep_prediction"],
                "deep_only_correct",
                "third_class",
            ),
        )
    if folds is not None:
        audit["validation_fold"] = folds.to_numpy()[row_positions]
    return audit


def _format_summary(summary: pd.DataFrame) -> str:
    display = summary.copy()
    for column in (
        "deep_accuracy",
        "gated_accuracy",
        "functional_recall_change",
        "repair_recall_change",
        "non_functional_recall_change",
    ):
        display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


if __name__ == "__main__":
    main()

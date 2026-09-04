"""Evaluate and generate the fixed six-rule repair override candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = STAGE_DIR.parent
DATA_DIR = STAGE_DIR / "data"
SRC_DIR = STAGE_DIR / "src"
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "repair-rule-ensemble"
SUBMISSION_DIR = STAGE_DIR / "submissions" / "2026-09-04-repair-rule-ensemble"
REPORT_PATH = STAGE_DIR / "reports" / "repair-rule-ensemble-screen.md"
INCUMBENT_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-08-23-deep-archive-seed-20260824"
    / "01-deep-archive-seed-20260824.csv"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import partition_modelling_data
from deep_archive_confirmation import blend_deep_archive
from final_model import CLASS_LABELS
from final_model import build_competition_prediction
from final_model import write_validated_submission
from modelling_data import prepare_modelling_data
from repair_rule_ensemble import FULL_RULE_NAMES
from repair_rule_ensemble import FUNCTIONAL_LABEL
from repair_rule_ensemble import HISTORY_RULE_NAMES
from repair_rule_ensemble import IDENTITY_CONFLICT_RULE
from repair_rule_ensemble import IDENTITY_REPAIR_TO_FUNCTIONAL_RATIO
from repair_rule_ensemble import NON_FUNCTIONAL_LABEL
from repair_rule_ensemble import REPAIR_HISTORY_RULES
from repair_rule_ensemble import REPAIR_LABEL
from repair_rule_ensemble import STRICT_HISTORY_RULE_NAMES
from repair_rule_ensemble import build_repair_rule_masks
from repair_rule_ensemble import cross_fit_repair_rule_masks
from repair_rule_ensemble import overlay_repair_rule_union
from repair_rule_ensemble import rule_overlap_counts
from repair_rule_ensemble import selected_rule_names
from repair_rule_ensemble import summarise_rule_contributions
from repair_residual_specialist import REPAIR_HISTORY_COLUMN
from repair_residual_specialist import REPAIR_HISTORY_MINIMUM_RATE
from repair_residual_specialist import REPAIR_HISTORY_MINIMUM_SUPPORT
from repair_residual_specialist import repair_history_mask


RESIDUAL_HISTORY_RULE_NAME = "residual_supported_subvillage_history"
CANDIDATES = {
    "full_union": {
        "filename": "01-repair-rule-full-union.csv",
        "rules": FULL_RULE_NAMES,
        "repair_rule_names": FULL_RULE_NAMES,
        "include_residual_history": False,
        "public_score": 0.8296,
        "submission_id": 321013,
    },
    "history_only": {
        "filename": "02-repair-rule-history-only.csv",
        "rules": HISTORY_RULE_NAMES,
        "repair_rule_names": HISTORY_RULE_NAMES,
        "include_residual_history": False,
        "public_score": None,
        "submission_id": None,
    },
    "strict_history_core_plus_residual_history": {
        "filename": "03-strict-history-core-plus-residual-history.csv",
        "rules": (*STRICT_HISTORY_RULE_NAMES, RESIDUAL_HISTORY_RULE_NAME),
        "repair_rule_names": STRICT_HISTORY_RULE_NAMES,
        "include_residual_history": True,
        "public_score": None,
        "submission_id": None,
    },
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

    oof_components = _load_oof_components(partitioned)
    oof_base = blend_deep_archive(oof_components)
    local_components, local_base = _load_local_components(partitioned)
    competition_components = _load_competition_components(modelling_data)
    competition_base = blend_deep_archive(competition_components)
    _validate_incumbent_reconstruction(competition_base)

    oof_masks = cross_fit_repair_rule_masks(
        partitioned.X_development,
        partitioned.y_development,
        partitioned.validation_folds,
        oof_base,
        oof_components["identity_catboost"],
    )
    local_masks = build_repair_rule_masks(
        partitioned.X_development,
        partitioned.y_development,
        partitioned.X_local_test,
        local_base,
        local_components["identity_catboost"],
    )
    competition_masks = build_repair_rule_masks(
        modelling_data.X_original,
        modelling_data.y_original,
        modelling_data.X_competition,
        competition_base,
        competition_components["identity_catboost"],
    )
    residual_history_masks = _build_residual_history_masks(
        modelling_data,
        partitioned,
    )

    oof_candidates = _build_candidates(
        oof_base,
        oof_masks,
        residual_history_masks["development"],
    )
    local_candidates = _build_candidates(
        local_base,
        local_masks,
        residual_history_masks["local"],
    )
    competition_candidates = _build_candidates(
        competition_base,
        competition_masks,
        residual_history_masks["competition"],
    )
    candidate_summary = _summarise_candidates(
        partitioned,
        oof_base,
        oof_candidates,
        local_base,
        local_candidates,
    )
    contingency = _build_contingency(
        partitioned,
        oof_masks,
        local_masks,
        competition_masks,
    )
    contributions = _build_contributions(
        partitioned,
        oof_masks,
        local_masks,
        competition_masks,
    )
    overlaps = {
        "development": rule_overlap_counts(oof_masks),
        "local": rule_overlap_counts(local_masks),
        "competition": rule_overlap_counts(competition_masks),
    }

    candidate_summary.to_csv(RUNTIME_DIR / "candidate-summary.csv")
    contingency.to_csv(RUNTIME_DIR / "rule-contingency.csv", index=False)
    contributions.to_csv(RUNTIME_DIR / "rule-contributions.csv", index=False)
    for dataset, overlap in overlaps.items():
        overlap.to_csv(RUNTIME_DIR / f"{dataset}-rule-overlap.csv")

    prediction_records, competition_flips = _write_competition_candidates(
        modelling_data,
        competition_base,
        competition_components["identity_catboost"],
        competition_masks,
        competition_candidates,
    )
    competition_flips.to_csv(RUNTIME_DIR / "competition-flips.csv", index=False)

    evidence_payload = {
        "development_ids": partitioned.development_ids.copy(),
        "local_test_ids": partitioned.local_test_ids.copy(),
        "competition_ids": modelling_data.competition_ids.copy(),
        "development_rule_masks": oof_masks,
        "local_rule_masks": local_masks,
        "competition_rule_masks": competition_masks,
        "residual_history_masks": {
            name: values.copy()
            for name, values in residual_history_masks.items()
        },
        "development_candidates": {
            name: selected.copy()
            for name, (_, selected) in oof_candidates.items()
        },
        "local_candidates": {
            name: selected.copy()
            for name, (_, selected) in local_candidates.items()
        },
        "competition_candidates": {
            name: selected.copy()
            for name, (_, selected) in competition_candidates.items()
        },
    }
    joblib.dump(evidence_payload, RUNTIME_DIR / "repair-rule-evidence.joblib")

    result = _build_result(
        candidate_summary,
        prediction_records,
        competition_masks,
    )
    (RUNTIME_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    (SUBMISSION_DIR / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(
        _format_report(
            candidate_summary,
            contingency,
            contributions,
            overlaps,
            prediction_records,
        ),
        encoding="utf-8",
    )

    print(_format_console_summary(candidate_summary), flush=True)
    print("\nPer-rule fold/local contingency (R/F/N, net):", flush=True)
    print(
        _format_contingency_for_console(contingency),
        flush=True,
    )
    print("\nCompetition rule contribution and overlap:", flush=True)
    print(
        contributions.loc[contributions["dataset"].eq("competition")]
        .set_index("rule")
        .to_string(),
        flush=True,
    )
    for name, record in prediction_records.items():
        print(
            f"\n{name}: {record['changed_vs_incumbent']} competition flips; "
            f"SHA-256 {record['sha256']}",
            flush=True,
        )
        print(f"IDs: {record['changed_ids']}", flush=True)


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


def _build_residual_history_masks(modelling_data, partitioned) -> dict[str, np.ndarray]:
    development = np.zeros(len(partitioned.X_development), dtype=bool)
    folds = partitioned.validation_folds.to_numpy()
    for fold in sorted(np.unique(folds)):
        training = folds != fold
        validation = folds == fold
        development[validation] = repair_history_mask(
            partitioned.X_development.iloc[training],
            partitioned.y_development.iloc[training],
            partitioned.X_development.iloc[validation],
        )
    result = {
        "development": development,
        "local": repair_history_mask(
            partitioned.X_development,
            partitioned.y_development,
            partitioned.X_local_test,
        ),
        "competition": repair_history_mask(
            modelling_data.X_original,
            modelling_data.y_original,
            modelling_data.X_competition,
        ),
    }
    expected_rows = {
        "development": len(partitioned.X_development),
        "local": len(partitioned.X_local_test),
        "competition": len(modelling_data.X_competition),
    }
    for name, values in result.items():
        if np.asarray(values).shape != (expected_rows[name],):
            raise ValueError(f"Residual-history {name} mask is misaligned.")
    return {name: np.asarray(values, dtype=bool) for name, values in result.items()}


def _build_candidates(
    base: np.ndarray,
    masks: pd.DataFrame,
    residual_history: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        name: overlay_repair_rule_union(
            base,
            masks,
            rule_names=details["repair_rule_names"],
            additional_mask=(
                residual_history
                if details["include_residual_history"]
                else None
            ),
        )
        for name, details in CANDIDATES.items()
    }


def _summarise_candidates(
    partitioned,
    oof_base: np.ndarray,
    oof_candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    local_base: np.ndarray,
    local_candidates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    labels = np.asarray(CLASS_LABELS)
    oof_target = partitioned.y_development.to_numpy()
    local_target = partitioned.y_local_test.to_numpy()
    oof_base_labels = labels[oof_base.argmax(axis=1)]
    local_base_labels = labels[local_base.argmax(axis=1)]
    rows = []
    for name in CANDIDATES:
        oof_probabilities, oof_selected = oof_candidates[name]
        local_probabilities, local_selected = local_candidates[name]
        oof_labels = labels[oof_probabilities.argmax(axis=1)]
        local_labels = labels[local_probabilities.argmax(axis=1)]
        row = {"candidate": name}
        row.update(
            _candidate_metrics(
                "development",
                oof_target,
                oof_base_labels,
                oof_labels,
                oof_selected,
            )
        )
        row.update(
            _candidate_metrics(
                "local",
                local_target,
                local_base_labels,
                local_labels,
                local_selected,
            )
        )
        fold_nets = []
        for fold in sorted(partitioned.validation_folds.unique()):
            fold_mask = partitioned.validation_folds.eq(fold).to_numpy()
            fold_metrics = _candidate_metrics(
                f"fold_{int(fold)}",
                oof_target[fold_mask],
                oof_base_labels[fold_mask],
                oof_labels[fold_mask],
                oof_selected[fold_mask],
            )
            row.update(fold_metrics)
            fold_nets.append(fold_metrics[f"fold_{int(fold)}_net_correct"])
        row["development_fold_wins"] = sum(net > 0 for net in fold_nets)
        row["development_worst_fold_net_correct"] = min(fold_nets)
        rows.append(row)
    return pd.DataFrame(rows).set_index("candidate")


def _candidate_metrics(
    prefix: str,
    target: np.ndarray,
    base_labels: np.ndarray,
    candidate_labels: np.ndarray,
    selected: np.ndarray,
) -> dict[str, float | int]:
    base_correct = base_labels == target
    candidate_correct = candidate_labels == target
    repair = int((target[selected] == REPAIR_LABEL).sum())
    functional = int((target[selected] == FUNCTIONAL_LABEL).sum())
    non_functional = int((target[selected] == NON_FUNCTIONAL_LABEL).sum())
    discordant = repair + functional
    p_value = (
        float(binomtest(repair, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        f"{prefix}_base_accuracy": accuracy_score(target, base_labels),
        f"{prefix}_accuracy": accuracy_score(target, candidate_labels),
        f"{prefix}_change": (
            accuracy_score(target, candidate_labels)
            - accuracy_score(target, base_labels)
        ),
        f"{prefix}_net_correct": int(candidate_correct.sum() - base_correct.sum()),
        f"{prefix}_flips": int(selected.sum()),
        f"{prefix}_flipped_actual_repair": repair,
        f"{prefix}_flipped_actual_functional": functional,
        f"{prefix}_flipped_actual_non_functional": non_functional,
        f"{prefix}_mcnemar_exact_p_value": p_value,
        f"{prefix}_base_repair_recall": recall_score(
            target,
            base_labels,
            labels=[REPAIR_LABEL],
            average=None,
            zero_division=0,
        )[0],
        f"{prefix}_repair_recall": recall_score(
            target,
            candidate_labels,
            labels=[REPAIR_LABEL],
            average=None,
            zero_division=0,
        )[0],
        f"{prefix}_base_repair_precision": precision_score(
            target,
            base_labels,
            labels=[REPAIR_LABEL],
            average=None,
            zero_division=0,
        )[0],
        f"{prefix}_repair_precision": precision_score(
            target,
            candidate_labels,
            labels=[REPAIR_LABEL],
            average=None,
            zero_division=0,
        )[0],
    }


def _build_contingency(
    partitioned,
    oof_masks: pd.DataFrame,
    local_masks: pd.DataFrame,
    competition_masks: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    fold_values = partitioned.validation_folds.to_numpy()
    oof_target = partitioned.y_development.to_numpy()
    for fold in sorted(np.unique(fold_values)):
        selected_rows = fold_values == fold
        rows.extend(
            _contingency_rows(
                "development",
                f"fold_{int(fold)}",
                oof_masks.loc[selected_rows].reset_index(drop=True),
                oof_target[selected_rows],
            )
        )
    rows.extend(
        _contingency_rows(
            "development",
            "all",
            oof_masks,
            oof_target,
        )
    )
    rows.extend(
        _contingency_rows(
            "local",
            "all",
            local_masks,
            partitioned.y_local_test.to_numpy(),
        )
    )
    rows.extend(
        _contingency_rows(
            "competition",
            "all",
            competition_masks,
            None,
        )
    )
    return pd.DataFrame(rows)


def _contingency_rows(
    dataset: str,
    segment: str,
    masks: pd.DataFrame,
    target: np.ndarray | None,
) -> list[dict[str, object]]:
    selections = {
        **{name: masks[name].to_numpy(dtype=bool) for name in FULL_RULE_NAMES},
        "history_only_union": masks.loc[:, list(HISTORY_RULE_NAMES)]
        .any(axis=1)
        .to_numpy(),
        "full_union": masks.loc[:, list(FULL_RULE_NAMES)].any(axis=1).to_numpy(),
    }
    rows = []
    for name, selected in selections.items():
        row: dict[str, object] = {
            "dataset": dataset,
            "segment": segment,
            "rule": name,
            "triggers": int(selected.sum()),
        }
        if target is not None:
            row.update(
                {
                    "actual_repair": int((target[selected] == REPAIR_LABEL).sum()),
                    "actual_functional": int(
                        (target[selected] == FUNCTIONAL_LABEL).sum()
                    ),
                    "actual_non_functional": int(
                        (target[selected] == NON_FUNCTIONAL_LABEL).sum()
                    ),
                }
            )
            row["net_correct"] = row["actual_repair"] - row["actual_functional"]
        else:
            row.update(
                {
                    "actual_repair": None,
                    "actual_functional": None,
                    "actual_non_functional": None,
                    "net_correct": None,
                }
            )
        rows.append(row)
    return rows


def _build_contributions(
    partitioned,
    oof_masks: pd.DataFrame,
    local_masks: pd.DataFrame,
    competition_masks: pd.DataFrame,
) -> pd.DataFrame:
    tables = []
    for dataset, masks, target in (
        ("development", oof_masks, partitioned.y_development),
        ("local", local_masks, partitioned.y_local_test),
        ("competition", competition_masks, None),
    ):
        table = summarise_rule_contributions(masks, target=target).reset_index()
        table.insert(0, "dataset", dataset)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _write_competition_candidates(
    modelling_data,
    base_probabilities: np.ndarray,
    identity_probabilities: np.ndarray,
    masks: pd.DataFrame,
    candidates: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")
    incumbent = pd.read_csv(INCUMBENT_PATH)
    records = {}
    for name, details in CANDIDATES.items():
        probabilities, selected = candidates[name]
        prediction = build_competition_prediction(
            modelling_data,
            template,
            probabilities,
            pd.Series(dtype="float64", name="fit and predict seconds"),
        )
        destination = write_validated_submission(
            prediction,
            SUBMISSION_DIR / details["filename"],
        )
        submission = pd.read_csv(destination)
        changed = submission["status_group"].ne(incumbent["status_group"])
        _validate_candidate_submission(
            name,
            submission,
            incumbent,
            modelling_data.competition_ids,
            selected,
        )
        changed_ids = submission.loc[changed, "id"].astype(int).tolist()
        records[name] = {
            "csv": details["filename"],
            "rules": list(details["rules"]),
            "rows": len(submission),
            "unique_ids": int(submission["id"].nunique()),
            "changed_vs_incumbent": int(changed.sum()),
            "changed_ids": changed_ids,
            "transitions": {f"{FUNCTIONAL_LABEL} -> {REPAIR_LABEL}": int(changed.sum())},
            "class_counts": {
                label: int(value)
                for label, value in submission["status_group"]
                .value_counts()
                .reindex(CLASS_LABELS, fill_value=0)
                .items()
            },
            "schema_validation": {
                "columns_exact": True,
                "row_count": True,
                "id_order": True,
                "ids_unique": True,
                "no_missing_values": True,
                "labels_valid": True,
                "selected_rows_only": True,
                "functional_to_repair_only": True,
            },
            "sha256": _sha256(destination),
        }

    full_selected = candidates["full_union"][1]
    base_ratio = _safe_ratio(
        base_probabilities[:, CLASS_LABELS.index(REPAIR_LABEL)],
        base_probabilities[:, CLASS_LABELS.index(FUNCTIONAL_LABEL)],
    )
    identity_ratio = _safe_ratio(
        identity_probabilities[:, CLASS_LABELS.index(REPAIR_LABEL)],
        identity_probabilities[:, CLASS_LABELS.index(FUNCTIONAL_LABEL)],
    )
    competition_flips = pd.DataFrame(
        {
            "id": modelling_data.competition_ids.loc[full_selected]
            .reset_index(drop=True)
            .astype(int),
            "base_label": incumbent.loc[full_selected, "status_group"].reset_index(
                drop=True
            ),
            "full_union_label": REPAIR_LABEL,
            "in_history_only": candidates["history_only"][1][full_selected],
            "base_repair_to_functional_ratio": base_ratio[full_selected],
            "identity_repair_to_functional_ratio": identity_ratio[full_selected],
            "triggered_rules": masks.loc[full_selected]
            .apply(selected_rule_names, axis=1)
            .reset_index(drop=True),
        }
    )
    if competition_flips["id"].tolist() != records["full_union"]["changed_ids"]:
        raise ValueError("Competition flip audit changed ID order.")
    return records, competition_flips


def _validate_candidate_submission(
    name: str,
    submission: pd.DataFrame,
    incumbent: pd.DataFrame,
    expected_ids: pd.Series,
    selected: np.ndarray,
) -> None:
    if list(submission.columns) != ["id", "status_group"]:
        raise ValueError(f"Candidate {name!r} has the wrong schema.")
    if len(submission) != len(expected_ids):
        raise ValueError(f"Candidate {name!r} has the wrong row count.")
    if not submission["id"].reset_index(drop=True).equals(
        expected_ids.reset_index(drop=True)
    ):
        raise ValueError(f"Candidate {name!r} changed competition ID order.")
    if submission["id"].duplicated().any():
        raise ValueError(f"Candidate {name!r} contains duplicate IDs.")
    if submission.isna().any().any():
        raise ValueError(f"Candidate {name!r} contains missing values.")
    if not set(submission["status_group"]).issubset(CLASS_LABELS):
        raise ValueError(f"Candidate {name!r} contains invalid labels.")
    changed = submission["status_group"].ne(incumbent["status_group"]).to_numpy()
    if not np.array_equal(changed, selected):
        raise ValueError(f"Candidate {name!r} changed unexpected rows.")
    if not incumbent.loc[changed, "status_group"].eq(FUNCTIONAL_LABEL).all():
        raise ValueError(f"Candidate {name!r} changed a non-functional decision.")
    if not submission.loc[changed, "status_group"].eq(REPAIR_LABEL).all():
        raise ValueError(f"Candidate {name!r} made an unexpected transition.")


def _build_result(
    summary: pd.DataFrame,
    prediction_records: dict[str, dict[str, object]],
    competition_masks: pd.DataFrame,
) -> dict[str, object]:
    entries = []
    for order, name in enumerate(CANDIDATES, start=1):
        evidence = {
            key: _json_number(key, value)
            for key, value in summary.loc[name].items()
        }
        entries.append(
            {
                "order": order,
                "key": name,
                **prediction_records[name],
                "evidence": evidence,
                "public_score": CANDIDATES[name]["public_score"],
                "submission_id": CANDIDATES[name]["submission_id"],
            }
        )
    full_ids = set(prediction_records["full_union"]["changed_ids"])
    history_ids = set(prediction_records["history_only"]["changed_ids"])
    strict_ids = set(
        prediction_records[
            "strict_history_core_plus_residual_history"
        ]["changed_ids"]
    )
    union_ids = pd.Series(
        prediction_records["full_union"]["changed_ids"],
        dtype="int64",
    )
    union_mask = competition_masks.loc[:, list(FULL_RULE_NAMES)].any(axis=1)
    union_rule_masks = competition_masks.loc[union_mask].reset_index(drop=True)
    return {
        "date": "2026-09-04",
        "status": "one_submitted_public_scored",
        "submission_date": "2026-09-04",
        "daily_submission_allowance": "one of three used; two remaining",
        "incumbent": {
            "csv": INCUMBENT_PATH.relative_to(STAGE_DIR).as_posix(),
            "public_score": 0.8298,
            "sha256": _sha256(INCUMBENT_PATH),
        },
        "method": {
            "scope": "functional decisions only",
            "transition": f"{FUNCTIONAL_LABEL} -> {REPAIR_LABEL}",
            "history_denominator": "functional + functional needs repair",
            "non_functional_excluded_from_history_support": True,
            "history_cross_fit": "each development fold used mappings fitted on the other four folds",
            "history_rules": [
                {
                    "name": rule.name,
                    "key": rule.key,
                    "minimum_conditional_repair_rate": rule.minimum_repair_rate,
                    "minimum_functional_plus_repair_support": rule.minimum_support,
                    "minimum_final_repair_to_functional_ratio": (
                        rule.minimum_repair_to_functional_ratio
                    ),
                }
                for rule in REPAIR_HISTORY_RULES
            ],
            "identity_rule": {
                "name": IDENTITY_CONFLICT_RULE,
                "minimum_identity_catboost_repair_to_functional_ratio": (
                    IDENTITY_REPAIR_TO_FUNCTIONAL_RATIO
                ),
            },
            "residual_history_rule": {
                "name": RESIDUAL_HISTORY_RULE_NAME,
                "column": REPAIR_HISTORY_COLUMN,
                "minimum_functional_plus_repair_support": (
                    REPAIR_HISTORY_MINIMUM_SUPPORT
                ),
                "laplace_smoothed_minimum_repair_rate": (
                    REPAIR_HISTORY_MINIMUM_RATE
                ),
                "history_cross_fit": (
                    "each development fold used mappings fitted on the other "
                    "four folds"
                ),
            },
        },
        "nested_ladder": {
            "history_only_is_strict_subset_of_full_union": (
                history_ids < full_ids
            ),
            "shared_competition_flips": len(history_ids & full_ids),
            "full_only_competition_flips": len(full_ids - history_ids),
            "strict_candidate_overlap_with_submitted_full_union": len(
                strict_ids & full_ids
            ),
            "strict_candidate_flips_outside_submitted_full_union": len(
                strict_ids - full_ids
            ),
        },
        "competition_per_rule_trigger_ids": {
            rule: union_ids.loc[union_rule_masks[rule]].astype(int).tolist()
            for rule in FULL_RULE_NAMES
        },
        "input_sha256": {
            path.name: _sha256(path)
            for path in (
                DATA_DIR / "TrainingSetValues.csv",
                DATA_DIR / "TrainingSetLabels.csv",
                DATA_DIR / "TestSetValues.csv",
                DATA_DIR / "SubmissionFormat.csv",
            )
        },
        "entries": entries,
        "runtime_artifacts": [
            "candidate-summary.csv",
            "rule-contingency.csv",
            "rule-contributions.csv",
            "development-rule-overlap.csv",
            "local-rule-overlap.csv",
            "competition-rule-overlap.csv",
            "competition-flips.csv",
            "repair-rule-evidence.joblib",
            "result.json",
        ],
    }


def _format_report(
    summary: pd.DataFrame,
    contingency: pd.DataFrame,
    contributions: pd.DataFrame,
    overlaps: dict[str, pd.DataFrame],
    prediction_records: dict[str, dict[str, object]],
) -> str:
    candidate_rows = []
    for name in CANDIDATES:
        row = summary.loc[name]
        candidate_rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(int(row["development_flips"])),
                    _rfnet(row, "development"),
                    f"{row['development_accuracy']:.6f}",
                    f"{row['development_change']:+.6f}",
                    str(int(row["local_flips"])),
                    _rfnet(row, "local"),
                    f"{row['local_accuracy']:.6f}",
                    f"{row['local_change']:+.6f}",
                    str(prediction_records[name]["changed_vs_incumbent"]),
                ]
            )
            + " |"
        )

    dev_folds = contingency.loc[
        contingency["dataset"].eq("development")
        & contingency["segment"].str.startswith("fold_")
    ]
    local = contingency.loc[
        contingency["dataset"].eq("local")
        & contingency["segment"].eq("all")
    ]
    contingency_rows = []
    for rule in (*FULL_RULE_NAMES, "history_only_union", "full_union"):
        cells = []
        for fold in range(1, 6):
            record = dev_folds.loc[
                dev_folds["rule"].eq(rule)
                & dev_folds["segment"].eq(f"fold_{fold}")
            ].iloc[0]
            cells.append(_contingency_cell(record))
        cells.append(_contingency_cell(local.loc[local["rule"].eq(rule)].iloc[0]))
        contingency_rows.append(
            "| " + " | ".join([rule, *cells]) + " |"
        )

    comp_contribution = contributions.loc[
        contributions["dataset"].eq("competition")
    ].set_index("rule")
    contribution_rows = []
    for rule in FULL_RULE_NAMES:
        record = comp_contribution.loc[rule]
        contribution_rows.append(
            "| "
            + " | ".join(
                [
                    rule,
                    str(int(record["standalone_triggers"])),
                    str(int(record["unique_triggers"])),
                    str(int(record["incremental_triggers"])),
                    str(int(record["overlaps_prior_rules"])),
                    ", ".join(
                        str(value)
                        for value in _competition_rule_ids(
                            rule,
                            prediction_records,
                            overlaps,
                        )
                    )
                    or "none",
                ]
            )
            + " |"
        )

    full = prediction_records["full_union"]
    history = prediction_records["history_only"]
    strict = prediction_records["strict_history_core_plus_residual_history"]
    strict_evidence = summary.loc[
        "strict_history_core_plus_residual_history"
    ]
    strict_fold_nets = [
        int(strict_evidence[f"fold_{fold}_net_correct"])
        for fold in range(1, 6)
    ]
    return f"""# Repair-rule ensemble screen

## Outcome

The fixed six-rule union recovered **23 net OOF rows** and **8 net local-test rows** by changing 65 and 25 incumbent `functional` decisions respectively to `functional needs repair`. Every development fold improved. After that union scored 0.8296 publicly, a stricter candidate retained only the four non-scheme history rules and added the independent supported-subvillage history rule.

| Candidate | OOF flips | OOF R/F/N (net) | OOF accuracy | OOF delta | Local flips | Local R/F/N (net) | Local accuracy | Local delta | Competition flips |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
{chr(10).join(candidate_rows)}

The full-union OOF accuracy changes from {summary.iloc[0]['development_base_accuracy']:.6f} to {summary.loc['full_union', 'development_accuracy']:.6f}; local accuracy changes from {summary.iloc[0]['local_base_accuracy']:.6f} to {summary.loc['full_union', 'local_accuracy']:.6f}. Its repair recall moves from {summary.iloc[0]['development_base_repair_recall']:.2%} to {summary.loc['full_union', 'development_repair_recall']:.2%} OOF and from {summary.iloc[0]['local_base_repair_recall']:.2%} to {summary.loc['full_union', 'local_repair_recall']:.2%} locally. The exact McNemar p-values are {summary.loc['full_union', 'development_mcnemar_exact_p_value']:.4f} and {summary.loc['full_union', 'local_mcnemar_exact_p_value']:.4f}; this is promising but still a small, adaptively selected override.

## Fixed rule contract

All rules can act only when the incumbent predicts `functional`. History support counts only `functional` and `functional needs repair`; `non functional` is excluded. OOF history is leakage-safe: each validation fold uses a mapping fitted on the other four folds.

| Rule | Key | Minimum repair rate | Minimum support | Probability guard |
| --- | --- | ---: | ---: | --- |
{chr(10).join(_rule_contract_rows())}

## Fold and local contingency

Cells are actual repair / functional / non-functional among flipped rows, followed by net accuracy contribution. The union rows include overlap de-duplication.

| Rule | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Local |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(contingency_rows)}

## Competition contribution and overlap

Incremental counts follow the fixed rule order above. Pairwise overlap matrices and the row-level trigger audit are saved under `.runtime/repair-rule-ensemble/`.

| Rule | Standalone | Globally unique | Incremental | Overlaps prior | Exact IDs |
| --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(contribution_rows)}

The history-only candidate changes these {history['changed_vs_incumbent']} IDs:

`{history['changed_ids']}`

The full union changes these {full['changed_vs_incumbent']} IDs:

`{full['changed_ids']}`

History-only is a strict subset of full union; the identity conflict adds eight unique competition rows. The grid rule triggers no competition row, and the strict subvillage rule adds no row beyond the LGA-subvillage rule on this test set, although both have independent OOF evidence.

## Post-result strict candidate

The strict history core excludes the weaker `scheme_name_history` and identity-conflict blocks. Its union with the independently defined supported-subvillage history rule changes {strict['changed_vs_incumbent']} competition rows, of which eight overlap the submitted full union and four are new. It recovers {int(strict_evidence['development_net_correct'])} net OOF rows with fold nets `{strict_fold_nets}`, and {int(strict_evidence['local_net_correct'])} net local rows.

Exact IDs:

`{strict['changed_ids']}`

## Generated candidates

- `{history['csv']}` — SHA-256 `{history['sha256']}`.
- `{full['csv']}` — SHA-256 `{full['sha256']}`.
- `{strict['csv']}` — SHA-256 `{strict['sha256']}`.

All files contain 14,850 unique IDs in template order, no missing or invalid labels, and only the intended `functional` to `functional needs repair` transitions.

## Public result

The full union was submitted unchanged on 4 September 2026 as submission `321013` and scored **0.8296**, below the 0.8298 incumbent. If all 14,850 competition rows are scored, four-decimal rounding makes this exactly a loss of two to four correct rows. The evaluator denominator is not disclosed locally, so this does not identify any row label. The history-only and strict candidates remain unsubmitted at this checkpoint.

## Interpretation

This is a targeted correction rather than a replacement model. The original positive direction replicated internally but failed to transfer for the complete 22-row union. The 12-row candidate is explicitly post-result and therefore adaptive: its rationale is to retain the strongest internal history block and add four rows from a separately defined rule outside the failed union, not to claim knowledge of hidden row labels.
"""


def _rule_contract_rows() -> list[str]:
    rows = [
        "| "
        + " | ".join(
            [
                rule.name,
                rule.key,
                f"{rule.minimum_repair_rate:.2f}",
                str(rule.minimum_support),
                (
                    "final ensemble repair/functional "
                    f">= {rule.minimum_repair_to_functional_ratio:.2f}"
                ),
            ]
        )
        + " |"
        for rule in REPAIR_HISTORY_RULES
    ]
    rows.append(
        "| "
        + " | ".join(
            [
                IDENTITY_CONFLICT_RULE,
                "none",
                "n/a",
                "n/a",
                (
                    "identity CatBoost repair/functional "
                    f">= {IDENTITY_REPAIR_TO_FUNCTIONAL_RATIO:.2f}"
                ),
            ]
        )
        + " |"
    )
    return rows


def _competition_rule_ids(
    rule: str,
    prediction_records: dict[str, dict[str, object]],
    overlaps: dict[str, pd.DataFrame],
) -> list[int]:
    del overlaps
    flips = pd.read_csv(RUNTIME_DIR / "competition-flips.csv")
    return flips.loc[
        flips["triggered_rules"].str.split(";").map(lambda names: rule in names),
        "id",
    ].astype(int).tolist()


def _format_console_summary(summary: pd.DataFrame) -> str:
    columns = [
        "development_accuracy",
        "development_change",
        "development_net_correct",
        "development_flips",
        "development_flipped_actual_repair",
        "development_flipped_actual_functional",
        "development_flipped_actual_non_functional",
        "development_fold_wins",
        "local_accuracy",
        "local_change",
        "local_net_correct",
        "local_flips",
        "local_flipped_actual_repair",
        "local_flipped_actual_functional",
        "local_flipped_actual_non_functional",
    ]
    return summary.loc[:, columns].to_string()


def _format_contingency_for_console(contingency: pd.DataFrame) -> str:
    selected = contingency.loc[
        ~contingency["dataset"].eq("competition")
        & ~(
            contingency["dataset"].eq("development")
            & contingency["segment"].eq("all")
        )
    ].copy()
    selected["cell"] = selected.apply(_contingency_cell, axis=1)
    return selected.pivot(index="rule", columns="segment", values="cell").to_string()


def _contingency_cell(record: pd.Series) -> str:
    return (
        f"{int(record['actual_repair'])}/"
        f"{int(record['actual_functional'])}/"
        f"{int(record['actual_non_functional'])} "
        f"({int(record['net_correct']):+d})"
    )


def _rfnet(row: pd.Series, prefix: str) -> str:
    return (
        f"{int(row[f'{prefix}_flipped_actual_repair'])}/"
        f"{int(row[f'{prefix}_flipped_actual_functional'])}/"
        f"{int(row[f'{prefix}_flipped_actual_non_functional'])} "
        f"({int(row[f'{prefix}_net_correct']):+d})"
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.asarray(numerator, dtype="float64") / np.maximum(
        np.asarray(denominator, dtype="float64"),
        1e-12,
    )


def _json_number(name: str, value) -> int | float:
    if name.endswith(
        (
            "_net_correct",
            "_flips",
            "_actual_repair",
            "_actual_functional",
            "_actual_non_functional",
            "_fold_wins",
        )
    ):
        return int(value)
    return float(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

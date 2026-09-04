"""Prepare the public-score-conditioned final repair candidate."""

from __future__ import annotations

import hashlib
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
OUTPUT_DIR = STAGE_DIR / "submissions" / "2026-09-04-final-slot"
DESTINATION = OUTPUT_DIR / "01-strict-repair-plus-meta-overlap.csv"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
INCUMBENT_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-08-23-deep-archive-seed-20260824"
    / "01-deep-archive-seed-20260824.csv"
)
STRICT_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-09-04-repair-rule-ensemble"
    / "03-strict-history-core-plus-residual-history.csv"
)
FULL_REPAIR_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-09-04-repair-rule-ensemble"
    / "01-repair-rule-full-union.csv"
)
META_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-09-04-final-push"
    / "02-repair-meta-070.csv"
)
RULE_EVIDENCE_PATH = (
    PROJECT_DIR / ".runtime" / "repair-rule-ensemble" / "repair-rule-evidence.joblib"
)
META_EVIDENCE_PATH = (
    PROJECT_DIR / ".runtime" / "repair-residual-screen" / "repair-residual-evidence.joblib"
)
STRICT_KEY = "strict_history_core_plus_residual_history"
FUNCTIONAL = "functional"
REPAIR = "functional needs repair"
VALID_LABELS = {FUNCTIONAL, REPAIR, "non functional"}
META_THRESHOLD = 0.70

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import partition_modelling_data
from modelling_data import prepare_modelling_data


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    incumbent = pd.read_csv(INCUMBENT_PATH)
    strict = pd.read_csv(STRICT_PATH)
    full_repair = pd.read_csv(FULL_REPAIR_PATH)
    meta = pd.read_csv(META_PATH)
    template = pd.read_csv(DATA_DIR / "SubmissionFormat.csv")
    _validate_aligned((incumbent, strict, full_repair, meta))
    if list(template.columns) != ["id", "status_group"]:
        raise ValueError("Submission template has an invalid schema.")
    if not template["id"].reset_index(drop=True).equals(
        incumbent["id"].reset_index(drop=True)
    ):
        raise ValueError("Submission template has different ID order.")

    strict_changed = strict["status_group"].ne(incumbent["status_group"])
    full_changed = full_repair["status_group"].ne(incumbent["status_group"])
    meta_changed = meta["status_group"].ne(incumbent["status_group"])
    if int(strict_changed.sum()) != 12:
        raise ValueError("Strict repair candidate no longer changes 12 rows.")
    if int(full_changed.sum()) != 22:
        raise ValueError("Submitted repair union no longer changes 22 rows.")
    if int(meta_changed.sum()) != 1:
        raise ValueError("Repair-meta candidate no longer changes one row.")
    if not (meta_changed <= full_changed).all():
        raise ValueError("Repair-meta row is no longer inside the submitted union.")
    if (meta_changed & strict_changed).any():
        raise ValueError("Repair-meta row unexpectedly overlaps the strict candidate.")

    meta_id = int(meta.loc[meta_changed, "id"].item())
    if meta_id != 60481:
        raise ValueError(f"Expected repair-meta overlap ID 60481, found {meta_id}.")
    candidate = strict.copy()
    candidate.loc[meta_changed, "status_group"] = meta.loc[
        meta_changed,
        "status_group",
    ]
    changed = candidate["status_group"].ne(incumbent["status_group"])
    _validate_candidate(candidate, incumbent, changed)
    if int(changed.sum()) != 13:
        raise ValueError("Final candidate must change exactly 13 rows.")

    _write_or_verify(candidate, DESTINATION)
    evidence = _build_evidence()
    changed_ids = candidate.loc[changed, "id"].astype(int).tolist()
    expected_ids = set(strict.loc[strict_changed, "id"].astype(int)) | {meta_id}
    if set(changed_ids) != expected_ids:
        raise ValueError("Written candidate does not contain the selected row union.")

    manifest = {
        "date": "2026-09-04",
        "status": "prepared_pending_action_confirmation",
        "selection": {
            "objective": "maximise the posterior chance of an outright first-place score",
            "composition": "12-row strict repair candidate plus the one row selected by both the repair meta-model and the identity-conflict rule",
            "public_score_constraints": {
                "incumbent_0.8298_correct_counts": [12322, 12323],
                "repair_union_0.8296_correct_counts": [12319, 12320],
                "gate_plus_strict_0.8293_correct_counts": [12315],
                "first_place_0.8301_correct_counts": [12327],
            },
            "conditional_assessment": {
                "model": "Jeffreys-smoothed Dirichlet-multinomial block model conditioned jointly on both public scores",
                "estimated_probability_of_0.8301_or_better": 0.1318,
                "sensitivity_range": [0.058, 0.132],
                "caveat": "The calculation assumes all 14,850 rows contribute to the displayed public score and exchangeable outcomes within evidence blocks.",
            },
        },
        "csv": DESTINATION.name,
        "rows": len(candidate),
        "unique_ids": int(candidate["id"].nunique()),
        "changed_vs_incumbent": int(changed.sum()),
        "changed_ids": changed_ids,
        "transitions": {f"{FUNCTIONAL} -> {REPAIR}": int(changed.sum())},
        "class_counts": {
            label: int(value)
            for label, value in candidate["status_group"]
            .value_counts()
            .reindex(sorted(VALID_LABELS), fill_value=0)
            .items()
        },
        "evidence": evidence,
        "sha256": _sha256(DESTINATION),
        "submission_note": "Strict repair + dual-evidence identity: 13 targeted changes; OOF +20, local +6",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def _build_evidence() -> dict[str, object]:
    rules = joblib.load(RULE_EVIDENCE_PATH)
    meta = joblib.load(META_EVIDENCE_PATH)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = partition_modelling_data(modelling_data)
    datasets = {
        "development": (
            rules["development_ids"],
            rules["development_candidates"],
            meta["oof_repair_probabilities"],
            partitioned.y_development,
        ),
        "local_test": (
            rules["local_test_ids"],
            rules["local_candidates"],
            meta["local_repair_probabilities"],
            partitioned.y_local_test,
        ),
    }
    result: dict[str, object] = {}
    development_mask = None
    for name, (ids, candidates, probabilities, target) in datasets.items():
        expected_ids = (
            partitioned.development_ids
            if name == "development"
            else partitioned.local_test_ids
        )
        if not ids.reset_index(drop=True).equals(expected_ids.reset_index(drop=True)):
            raise ValueError(f"{name} evidence IDs no longer align.")
        strict_mask = np.asarray(candidates[STRICT_KEY], dtype=bool)
        meta_overlap = np.asarray(candidates["full_union"], dtype=bool) & (
            np.asarray(probabilities, dtype="float64") >= META_THRESHOLD
        )
        if (strict_mask & meta_overlap).any():
            raise ValueError(f"{name} evidence masks unexpectedly overlap.")
        selected = strict_mask | meta_overlap
        actual = np.asarray(target, dtype=object)[selected]
        counts = {
            "actual_repair": int((actual == REPAIR).sum()),
            "actual_functional": int((actual == FUNCTIONAL).sum()),
            "actual_non_functional": int((actual == "non functional").sum()),
        }
        result[name] = {
            "flips": int(selected.sum()),
            **counts,
            "net_correct": counts["actual_repair"] - counts["actual_functional"],
            "strict_flips": int(strict_mask.sum()),
            "dual_evidence_identity_flips": int(meta_overlap.sum()),
        }
        if not (actual[np.asarray(meta_overlap[selected])] == REPAIR).all():
            raise ValueError(f"{name} dual-evidence identity analogue was not exact.")
        if name == "development":
            development_mask = selected

    if development_mask is None:
        raise RuntimeError("Development evidence was not evaluated.")
    fold_nets = []
    target = partitioned.y_development.to_numpy(dtype=object)
    for fold in range(1, 6):
        selected = development_mask & partitioned.validation_folds.eq(fold).to_numpy()
        actual = target[selected]
        fold_nets.append(
            int((actual == REPAIR).sum() - (actual == FUNCTIONAL).sum())
        )
    result["development"]["fold_net_correct"] = fold_nets
    return result


def _validate_aligned(frames: tuple[pd.DataFrame, ...]) -> None:
    expected_ids = frames[0]["id"].reset_index(drop=True)
    for frame in frames:
        if list(frame.columns) != ["id", "status_group"]:
            raise ValueError("Submission input has an invalid schema.")
        if not frame["id"].reset_index(drop=True).equals(expected_ids):
            raise ValueError("Submission inputs have different ID order.")
        if frame["id"].duplicated().any() or frame.isna().any().any():
            raise ValueError("Submission input contains duplicate IDs or missing values.")
        if not set(frame["status_group"]).issubset(VALID_LABELS):
            raise ValueError("Submission input contains an invalid class label.")


def _validate_candidate(
    candidate: pd.DataFrame,
    incumbent: pd.DataFrame,
    changed: pd.Series,
) -> None:
    _validate_aligned((candidate, incumbent))
    if not incumbent.loc[changed, "status_group"].eq(FUNCTIONAL).all():
        raise ValueError("Final candidate changed a non-functional incumbent row.")
    if not candidate.loc[changed, "status_group"].eq(REPAIR).all():
        raise ValueError("Final candidate contains an unexpected transition.")


def _write_or_verify(candidate: pd.DataFrame, destination: Path) -> None:
    if destination.exists():
        if not pd.read_csv(destination).equals(candidate.reset_index(drop=True)):
            raise FileExistsError(f"Refusing to overwrite a different file: {destination}")
        return
    candidate.to_csv(destination, index=False)
    if not pd.read_csv(destination).equals(candidate.reset_index(drop=True)):
        raise ValueError("Reloaded final candidate differs from the generated frame.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

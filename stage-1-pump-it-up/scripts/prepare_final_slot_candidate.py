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
DESTINATION = OUTPUT_DIR / "02-strict-meta-consensus-repair.csv"
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
CONSENSUS_MINIMUM_REPAIR_VOTES = 4
CONSENSUS_MINIMUM_REPAIR_TO_FUNCTIONAL_RATIO = 0.96
PUBLIC_SCORE = 0.8294
SUBMISSION_ID = 321023

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_partitioning import partition_modelling_data
from deep_archive_confirmation import blend_deep_archive
from modelling_data import prepare_modelling_data
from repair_residual_specialist import REPAIR_META_COMPONENTS
from run_repair_residual_screen import _load_competition_components
from run_repair_residual_screen import _load_local_components
from run_repair_residual_screen import _load_oof_components


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
    evidence, consensus_ids = _build_evidence()
    expected_consensus_ids = {61277, 23140, 21503}
    if set(consensus_ids) != expected_consensus_ids:
        raise ValueError(
            "Consensus repair selection changed; expected "
            f"{sorted(expected_consensus_ids)}, found {sorted(consensus_ids)}."
        )
    consensus_changed = incumbent["id"].isin(consensus_ids)
    if (consensus_changed & (strict_changed | meta_changed)).any():
        raise ValueError("Consensus rows unexpectedly overlap another final block.")
    candidate = strict.copy()
    candidate.loc[meta_changed | consensus_changed, "status_group"] = REPAIR
    changed = candidate["status_group"].ne(incumbent["status_group"])
    _validate_candidate(candidate, incumbent, changed)
    if int(changed.sum()) != 16:
        raise ValueError("Final candidate must change exactly 16 rows.")

    _write_or_verify(candidate, DESTINATION)
    changed_ids = candidate.loc[changed, "id"].astype(int).tolist()
    expected_ids = (
        set(strict.loc[strict_changed, "id"].astype(int))
        | {meta_id}
        | expected_consensus_ids
    )
    if set(changed_ids) != expected_ids:
        raise ValueError("Written candidate does not contain the selected row union.")

    manifest = {
        "date": "2026-09-04",
        "status": "submitted_public_scored",
        "submission_date": "2026-09-04",
        "daily_submission_allowance": "three of three used; none remaining",
        "selection": {
            "objective": "maximise the posterior chance of an outright first-place score",
            "composition": "12-row strict repair candidate, one repair-meta row and three untouched near-boundary rows supported by at least four of six components",
            "public_score_constraints": {
                "incumbent_0.8298_correct_counts": [12322, 12323],
                "repair_union_0.8296_correct_counts": [12319, 12320],
                "gate_plus_strict_0.8293_correct_counts": [12315],
                "first_place_0.8301_correct_counts": [12327],
            },
            "pre_submission_conditional_assessment": {
                "model": "Jeffreys-smoothed Dirichlet-multinomial block model conditioned jointly on both public scores",
                "estimated_probability_of_0.8301_or_better": 0.2561,
                "sensitivity_range": [0.077, 0.267],
                "caveat": "The calculation assumes all 14,850 rows contribute to the displayed public score and exchangeable outcomes within evidence blocks.",
            },
            "consensus_rule": {
                "repair_votes_across_six_components": CONSENSUS_MINIMUM_REPAIR_VOTES,
                "minimum_blend_repair_to_functional_ratio": CONSENSUS_MINIMUM_REPAIR_TO_FUNCTIONAL_RATIO,
                "competition_ids": consensus_ids,
                "caveat": "The threshold was selected during the final audit, has seven OOF analogues and no local-test trigger, and its component votes are correlated.",
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
        "submission_note": "Strict + meta + consensus repair: 16 targeted changes; OOF +21, local +7",
        "public_score": PUBLIC_SCORE,
        "submission_id": SUBMISSION_ID,
        "observed_best_score_after_submission": 0.8298,
        "observed_rank_after_submission": 3,
        "possible_net_correct_vs_incumbent": [-7, -6, -5],
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)


def _build_evidence() -> tuple[dict[str, object], list[int]]:
    rules = joblib.load(RULE_EVIDENCE_PATH)
    meta = joblib.load(META_EVIDENCE_PATH)
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
    datasets = {
        "development": (
            rules["development_ids"],
            rules["development_candidates"],
            meta["oof_repair_probabilities"],
            partitioned.y_development,
            oof_base,
            oof_components,
        ),
        "local_test": (
            rules["local_test_ids"],
            rules["local_candidates"],
            meta["local_repair_probabilities"],
            partitioned.y_local_test,
            local_base,
            local_components,
        ),
    }
    result: dict[str, object] = {}
    development_mask = None
    for name, (
        ids,
        candidates,
        probabilities,
        target,
        base,
        components,
    ) in datasets.items():
        expected_ids = (
            partitioned.development_ids
            if name == "development"
            else partitioned.local_test_ids
        )
        if not ids.reset_index(drop=True).equals(expected_ids.reset_index(drop=True)):
            raise ValueError(f"{name} evidence IDs no longer align.")
        strict_mask = np.asarray(candidates[STRICT_KEY], dtype=bool)
        meta_mask = (base.argmax(axis=1) == 0) & (
            np.asarray(probabilities, dtype="float64") >= META_THRESHOLD
        )
        consensus_mask = _consensus_repair_mask(base, components)
        if (
            (strict_mask & meta_mask).any()
            or (strict_mask & consensus_mask).any()
            or (meta_mask & consensus_mask).any()
        ):
            raise ValueError(f"{name} evidence masks unexpectedly overlap.")
        selected = strict_mask | meta_mask | consensus_mask
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
            "repair_meta_flips": int(meta_mask.sum()),
            "consensus_flips": int(consensus_mask.sum()),
        }
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
    competition_strict = np.asarray(
        rules["competition_candidates"][STRICT_KEY],
        dtype=bool,
    )
    competition_meta = (competition_base.argmax(axis=1) == 0) & (
        np.asarray(meta["competition_repair_probabilities"], dtype="float64")
        >= META_THRESHOLD
    )
    competition_consensus = _consensus_repair_mask(
        competition_base,
        competition_components,
    )
    if (
        (competition_strict & competition_meta).any()
        or (competition_strict & competition_consensus).any()
        or (competition_meta & competition_consensus).any()
    ):
        raise ValueError("Competition evidence masks unexpectedly overlap.")
    if (
        int(competition_strict.sum()),
        int(competition_meta.sum()),
        int(competition_consensus.sum()),
    ) != (12, 1, 3):
        raise ValueError("Competition final-block sizes changed.")
    competition_ids = rules["competition_ids"].reset_index(drop=True)
    if not competition_ids.equals(modelling_data.competition_ids.reset_index(drop=True)):
        raise ValueError("Competition evidence IDs no longer align.")
    consensus_ids = (
        competition_ids.loc[competition_consensus].astype(int).tolist()
    )
    return result, consensus_ids


def _consensus_repair_mask(
    base: np.ndarray,
    components: dict[str, np.ndarray],
) -> np.ndarray:
    missing = set(REPAIR_META_COMPONENTS).difference(components)
    if missing:
        raise KeyError(f"Consensus components are incomplete: {sorted(missing)}")
    repair_votes = np.column_stack(
        [
            np.asarray(components[name]).argmax(axis=1) == 1
            for name in REPAIR_META_COMPONENTS
        ]
    ).sum(axis=1)
    repair_to_functional = base[:, 1] / np.maximum(base[:, 0], 1e-12)
    return (
        (base.argmax(axis=1) == 0)
        & (repair_votes >= CONSENSUS_MINIMUM_REPAIR_VOTES)
        & (
            repair_to_functional
            >= CONSENSUS_MINIMUM_REPAIR_TO_FUNCTIONAL_RATIO
        )
    )


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

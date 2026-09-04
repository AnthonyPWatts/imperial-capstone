"""Run one checkpointed RealMLP-TD screen on the frozen partitions."""

from __future__ import annotations

import argparse
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
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "realmlp-screen"
SUBMISSION_DIR = STAGE_DIR / "submissions" / "2026-09-04-realmlp"
INCUMBENT_PATH = (
    STAGE_DIR
    / "submissions"
    / "2026-08-23-deep-archive-seed-20260824"
    / "01-deep-archive-seed-20260824.csv"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import CROSS_VALIDATION_FOLDS
from data_partitioning import make_cross_validation
from data_partitioning import partition_modelling_data
from deep_archive_confirmation import blend_deep_archive
from final_model import build_competition_prediction
from final_model import write_validated_submission
from modelling_data import prepare_modelling_data
from realmlp_evaluation import CLASS_LABELS
from realmlp_evaluation import REALMLP_BLEND_WEIGHTS
from realmlp_evaluation import REALMLP_MODEL_NAME
from realmlp_evaluation import REALMLP_SEED
from realmlp_evaluation import blend_probabilities
from realmlp_evaluation import fit_predict_realmlp
from realmlp_evaluation import hard_predictions
from realmlp_evaluation import validate_probabilities


def main() -> None:
    args = _parse_arguments()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    _validate_device(args.device)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = partition_modelling_data(modelling_data)
    cross_validation = make_cross_validation(partitioned)
    incumbent_oof = _load_oof_incumbent(partitioned)
    incumbent_local = _load_local_incumbent(partitioned)

    realmlp_oof, fold_diagnostics = _fit_oof(
        partitioned,
        cross_validation,
        device=args.device,
        force=args.force,
    )
    realmlp_local, local_diagnostics = _fit_local(
        partitioned,
        device=args.device,
        force=args.force,
    )
    summary, transitions = _summarise_screen(
        partitioned,
        cross_validation,
        incumbent_oof,
        realmlp_oof,
        incumbent_local,
        realmlp_local,
    )
    summary.to_csv(RUNTIME_DIR / "candidate-summary.csv")
    transitions.to_csv(RUNTIME_DIR / "prediction-transitions.csv", index=False)

    selected_key = _select_strictly_positive_blend(summary)
    competition_record = None
    if selected_key is not None:
        selected_weight = float(summary.loc[selected_key, "realmlp_weight"])
        competition_record = _prepare_competition_candidate(
            modelling_data,
            selected_weight,
            device=args.device,
            force=args.force,
        )

    result = {
        "model": REALMLP_MODEL_NAME,
        "pytabkit_version": "1.7.3",
        "seed": REALMLP_SEED,
        "device": args.device,
        "folds": CROSS_VALIDATION_FOLDS,
        "oversampling": False,
        "class_weights": None,
        "features": "accepted 29 plus six normalised deferred identities",
        "numeric_missing_values": "training-fold median",
        "blend_weights": list(REALMLP_BLEND_WEIGHTS),
        "selection_rule": (
            "highest OOF accuracy change among fixed blends with strictly "
            "positive OOF and local accuracy changes; local change then lower "
            "weight break ties"
        ),
        "selected_candidate": selected_key,
        "competition_candidate": competition_record,
        "fold_diagnostics": fold_diagnostics.to_dict(orient="index"),
        "local_diagnostics": local_diagnostics,
        "summary": {
            key: {
                column: _json_scalar(value)
                for column, value in row.items()
            }
            for key, row in summary.iterrows()
        },
    }
    (RUNTIME_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "cross_validation_fingerprint": (
                partitioned.cross_validation_fingerprint
            ),
            "local_test_fingerprint": partitioned.local_test_fingerprint,
            "development_ids": partitioned.development_ids.copy(),
            "local_test_ids": partitioned.local_test_ids.copy(),
            "realmlp_oof_probabilities": realmlp_oof,
            "realmlp_local_probabilities": realmlp_local,
            "incumbent_oof_probabilities": incumbent_oof,
            "incumbent_local_probabilities": incumbent_local,
        },
        RUNTIME_DIR / "realmlp-evidence.joblib",
    )
    print("\nRealMLP screen", flush=True)
    print(_format_summary(summary), flush=True)
    print("\n" + json.dumps(result, indent=2), flush=True)


def _fit_oof(partitioned, cross_validation, *, device: str, force: bool):
    probabilities = np.full(
        (len(partitioned.y_development), len(CLASS_LABELS)),
        np.nan,
        dtype="float64",
    )
    diagnostics = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        checkpoint_path = RUNTIME_DIR / f"fold-{fold_number}.joblib"
        expected_ids = partitioned.development_ids.iloc[
            validation_positions
        ].reset_index(drop=True)
        if checkpoint_path.exists() and not force:
            checkpoint = joblib.load(checkpoint_path)
            _validate_checkpoint(
                checkpoint,
                fingerprint=partitioned.cross_validation_fingerprint,
                expected_ids=expected_ids,
                expected_rows=len(validation_positions),
            )
            fold_probabilities = checkpoint["probabilities"]
            fit_seconds = float(checkpoint["fit_seconds"])
            predict_seconds = float(checkpoint["predict_seconds"])
            source = "checkpoint"
        else:
            fitted = fit_predict_realmlp(
                partitioned.X_development.iloc[training_positions],
                partitioned.y_development.iloc[training_positions],
                partitioned.X_development.iloc[validation_positions],
                RUNTIME_DIR / "tmp" / f"fold-{fold_number}",
                device=device,
            )
            fold_probabilities = fitted.probabilities
            fit_seconds = fitted.fit_seconds
            predict_seconds = fitted.predict_seconds
            checkpoint = {
                "fingerprint": partitioned.cross_validation_fingerprint,
                "prediction_ids": expected_ids,
                "seed": REALMLP_SEED,
                "probabilities": fold_probabilities,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
            joblib.dump(checkpoint, checkpoint_path)
            source = "fitted"
        probabilities[validation_positions] = fold_probabilities
        diagnostics.append(
            {
                "validation_fold": fold_number,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
                "source": source,
            }
        )
        fold_accuracy = accuracy_score(
            partitioned.y_development.iloc[validation_positions],
            hard_predictions(fold_probabilities),
        )
        print(
            f"Completed RealMLP fold {fold_number}/{CROSS_VALIDATION_FOLDS}: "
            f"accuracy={fold_accuracy:.4%}, fit={fit_seconds:.1f}s "
            f"({source}).",
            flush=True,
        )
    validate_probabilities(probabilities, len(partitioned.y_development))
    return probabilities, pd.DataFrame(diagnostics).set_index("validation_fold")


def _fit_local(partitioned, *, device: str, force: bool):
    checkpoint_path = RUNTIME_DIR / "local-test.joblib"
    if checkpoint_path.exists() and not force:
        checkpoint = joblib.load(checkpoint_path)
        _validate_checkpoint(
            checkpoint,
            fingerprint=partitioned.local_test_fingerprint,
            expected_ids=partitioned.local_test_ids,
            expected_rows=len(partitioned.y_local_test),
        )
        probabilities = checkpoint["probabilities"]
        source = "checkpoint"
    else:
        fitted = fit_predict_realmlp(
            partitioned.X_development,
            partitioned.y_development,
            partitioned.X_local_test,
            RUNTIME_DIR / "tmp" / "local-test",
            device=device,
        )
        probabilities = fitted.probabilities
        checkpoint = {
            "fingerprint": partitioned.local_test_fingerprint,
            "prediction_ids": partitioned.local_test_ids.reset_index(drop=True),
            "seed": REALMLP_SEED,
            "probabilities": probabilities,
            "fit_seconds": fitted.fit_seconds,
            "predict_seconds": fitted.predict_seconds,
        }
        joblib.dump(checkpoint, checkpoint_path)
        source = "fitted"
    return probabilities, {
        "fit_seconds": float(checkpoint["fit_seconds"]),
        "predict_seconds": float(checkpoint["predict_seconds"]),
        "source": source,
    }


def _summarise_screen(
    partitioned,
    cross_validation,
    incumbent_oof,
    realmlp_oof,
    incumbent_local,
    realmlp_local,
):
    candidates = {"standalone": (1.0, realmlp_oof, realmlp_local)}
    for weight in REALMLP_BLEND_WEIGHTS:
        candidates[f"blend_{int(weight * 100):02d}"] = (
            weight,
            blend_probabilities(incumbent_oof, realmlp_oof, weight),
            blend_probabilities(incumbent_local, realmlp_local, weight),
        )

    base_oof_metrics = _metrics(partitioned.y_development, incumbent_oof)
    base_local_metrics = _metrics(partitioned.y_local_test, incumbent_local)
    rows = []
    transition_frames = []
    for key, (weight, oof, local) in candidates.items():
        oof_metrics = _metrics(partitioned.y_development, oof)
        local_metrics = _metrics(partitioned.y_local_test, local)
        fold_changes = []
        for _, validation_positions in cross_validation.split():
            fold_changes.append(
                accuracy_score(
                    partitioned.y_development.iloc[validation_positions],
                    hard_predictions(oof[validation_positions]),
                )
                - accuracy_score(
                    partitioned.y_development.iloc[validation_positions],
                    hard_predictions(incumbent_oof[validation_positions]),
                )
            )
        oof_changes = _change_metrics(
            partitioned.y_development,
            incumbent_oof,
            oof,
        )
        local_changes = _change_metrics(
            partitioned.y_local_test,
            incumbent_local,
            local,
        )
        rows.append(
            {
                "candidate": key,
                "realmlp_weight": weight,
                "oof_accuracy": oof_metrics["accuracy"],
                "oof_accuracy_change": (
                    oof_metrics["accuracy"] - base_oof_metrics["accuracy"]
                ),
                "oof_net_correct": oof_changes["net_correct"],
                "oof_fold_wins": sum(change > 0 for change in fold_changes),
                "oof_worst_fold_change": min(fold_changes),
                "oof_repair_recall": oof_metrics["repair_recall"],
                "oof_repair_recall_change": (
                    oof_metrics["repair_recall"]
                    - base_oof_metrics["repair_recall"]
                ),
                "oof_repair_precision": oof_metrics["repair_precision"],
                "oof_changed_rows": oof_changes["changed_rows"],
                "oof_functional_to_repair": oof_changes[
                    "functional_to_repair"
                ],
                "oof_functional_to_repair_net_correct": oof_changes[
                    "functional_to_repair_net_correct"
                ],
                "oof_repair_to_functional": oof_changes[
                    "repair_to_functional"
                ],
                "oof_repair_to_functional_net_correct": oof_changes[
                    "repair_to_functional_net_correct"
                ],
                "local_accuracy": local_metrics["accuracy"],
                "local_accuracy_change": (
                    local_metrics["accuracy"] - base_local_metrics["accuracy"]
                ),
                "local_net_correct": local_changes["net_correct"],
                "local_repair_recall": local_metrics["repair_recall"],
                "local_repair_recall_change": (
                    local_metrics["repair_recall"]
                    - base_local_metrics["repair_recall"]
                ),
                "local_repair_precision": local_metrics["repair_precision"],
                "local_changed_rows": local_changes["changed_rows"],
                "local_functional_to_repair": local_changes[
                    "functional_to_repair"
                ],
                "local_functional_to_repair_net_correct": local_changes[
                    "functional_to_repair_net_correct"
                ],
                "local_repair_to_functional": local_changes[
                    "repair_to_functional"
                ],
                "local_repair_to_functional_net_correct": local_changes[
                    "repair_to_functional_net_correct"
                ],
            }
        )
        transition_frames.extend(
            [
                _transition_frame(
                    "development OOF",
                    key,
                    partitioned.y_development,
                    incumbent_oof,
                    oof,
                ),
                _transition_frame(
                    "local test",
                    key,
                    partitioned.y_local_test,
                    incumbent_local,
                    local,
                ),
            ]
        )
    summary = pd.DataFrame(rows).set_index("candidate")
    transitions = pd.concat(transition_frames, ignore_index=True)
    return summary, transitions


def _metrics(target, probabilities):
    predictions = hard_predictions(probabilities)
    values = np.asarray(target)
    return {
        "accuracy": float(accuracy_score(values, predictions)),
        "repair_recall": float(
            recall_score(
                values,
                predictions,
                labels=["functional needs repair"],
                average=None,
                zero_division=0,
            )[0]
        ),
        "repair_precision": float(
            precision_score(
                values,
                predictions,
                labels=["functional needs repair"],
                average=None,
                zero_division=0,
            )[0]
        ),
    }


def _change_metrics(target, incumbent, candidate):
    values = np.asarray(target)
    base = hard_predictions(incumbent)
    changed = hard_predictions(candidate)
    base_correct = base == values
    changed_correct = changed == values
    functional_to_repair = (base == "functional") & (
        changed == "functional needs repair"
    )
    repair_to_functional = (base == "functional needs repair") & (
        changed == "functional"
    )
    return {
        "changed_rows": int((base != changed).sum()),
        "net_correct": int(changed_correct.sum() - base_correct.sum()),
        "functional_to_repair": int(functional_to_repair.sum()),
        "functional_to_repair_net_correct": int(
            (changed_correct[functional_to_repair]).sum()
            - (base_correct[functional_to_repair]).sum()
        ),
        "repair_to_functional": int(repair_to_functional.sum()),
        "repair_to_functional_net_correct": int(
            (changed_correct[repair_to_functional]).sum()
            - (base_correct[repair_to_functional]).sum()
        ),
    }


def _transition_frame(
    partition_name,
    candidate_name,
    target,
    incumbent,
    candidate,
):
    frame = pd.DataFrame(
        {
            "actual": np.asarray(target),
            "incumbent_prediction": hard_predictions(incumbent),
            "candidate_prediction": hard_predictions(candidate),
        }
    )
    frame = frame.loc[
        frame["incumbent_prediction"].ne(frame["candidate_prediction"])
    ]
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "partition",
                "candidate",
                "incumbent_prediction",
                "candidate_prediction",
                "actual",
                "rows",
            ]
        )
    grouped = (
        frame.groupby(
            ["incumbent_prediction", "candidate_prediction", "actual"],
            observed=True,
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    grouped.insert(0, "candidate", candidate_name)
    grouped.insert(0, "partition", partition_name)
    return grouped


def _select_strictly_positive_blend(summary):
    eligible = summary.loc[
        summary.index.str.startswith("blend_")
        & summary["oof_accuracy_change"].gt(0.0)
        & summary["local_accuracy_change"].gt(0.0)
    ]
    if eligible.empty:
        return None
    ranked = eligible.sort_values(
        ["oof_accuracy_change", "local_accuracy_change", "realmlp_weight"],
        ascending=[False, False, True],
    )
    return str(ranked.index[0])


def _prepare_competition_candidate(
    modelling_data,
    selected_weight,
    *,
    device,
    force,
):
    incumbent = _load_competition_incumbent(modelling_data)
    _validate_incumbent_reconstruction(incumbent)
    checkpoint_path = RUNTIME_DIR / "competition.joblib"
    if checkpoint_path.exists() and not force:
        checkpoint = joblib.load(checkpoint_path)
        _validate_checkpoint(
            checkpoint,
            fingerprint="competition",
            expected_ids=modelling_data.competition_ids,
            expected_rows=len(modelling_data.X_competition),
        )
        realmlp = checkpoint["probabilities"]
        source = "checkpoint"
    else:
        fitted = fit_predict_realmlp(
            modelling_data.X_original,
            modelling_data.y_original,
            modelling_data.X_competition,
            RUNTIME_DIR / "tmp" / "competition",
            device=device,
        )
        realmlp = fitted.probabilities
        checkpoint = {
            "fingerprint": "competition",
            "prediction_ids": modelling_data.competition_ids.reset_index(drop=True),
            "seed": REALMLP_SEED,
            "probabilities": realmlp,
            "fit_seconds": fitted.fit_seconds,
            "predict_seconds": fitted.predict_seconds,
        }
        joblib.dump(checkpoint, checkpoint_path)
        source = "fitted"

    candidate = blend_probabilities(incumbent, realmlp, selected_weight)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"01-realmlp-td-{int(selected_weight * 100):02d}-blend.csv"
    prediction = build_competition_prediction(
        modelling_data,
        pd.read_csv(DATA_DIR / "SubmissionFormat.csv"),
        candidate,
        pd.Series(
            {"RealMLP-TD": float(checkpoint["fit_seconds"])},
            name="fit and predict seconds",
        ),
    )
    destination = write_validated_submission(
        prediction,
        SUBMISSION_DIR / filename,
    )
    incumbent_labels = hard_predictions(incumbent)
    candidate_labels = hard_predictions(candidate)
    changed = incumbent_labels != candidate_labels
    record = {
        "status": "prepared_not_uploaded",
        "filename": filename,
        "realmlp_weight": selected_weight,
        "rows": len(candidate_labels),
        "unique_ids": int(modelling_data.competition_ids.nunique()),
        "changed_vs_incumbent": int(changed.sum()),
        "functional_to_repair": int(
            (
                (incumbent_labels == "functional")
                & (candidate_labels == "functional needs repair")
            ).sum()
        ),
        "repair_to_functional": int(
            (
                (incumbent_labels == "functional needs repair")
                & (candidate_labels == "functional")
            ).sum()
        ),
        "class_counts": {
            label: int((candidate_labels == label).sum()) for label in CLASS_LABELS
        },
        "fit_seconds": float(checkpoint["fit_seconds"]),
        "source": source,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    manifest = {
        "date": "2026-09-04",
        "incumbent_public_score": 0.8298,
        "model": REALMLP_MODEL_NAME,
        "seed": REALMLP_SEED,
        "candidate": record,
    }
    (SUBMISSION_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def _load_oof_incumbent(partitioned):
    baseline = joblib.load(
        PROJECT_DIR / ".runtime" / "geography-screen" / "frozen-baseline.joblib"
    )
    evaluations = {
        "deep_xgboost": joblib.load(
            PROJECT_DIR
            / ".runtime"
            / "archived-deep-xgboost-screen"
            / "depth-17-seed-20260824.joblib"
        ),
        "spatial_xgboost": joblib.load(
            PROJECT_DIR
            / ".runtime"
            / "spatial-height-imputation-screen"
            / "ten-neighbour-height.joblib"
        ).xgboost,
        "random_forest": baseline.random_forest,
        "frequency_random_forest": joblib.load(
            PROJECT_DIR
            / ".runtime"
            / "categorical-frequency-forest-screen"
            / "all-categorical-occurrence-counts.joblib"
        ).random_forest,
        "spatial_random_forest": joblib.load(
            PROJECT_DIR
            / ".runtime"
            / "spatial-height-imputation-screen"
            / "ten-neighbour-height.joblib"
        ).random_forest,
        "identity_catboost": joblib.load(
            PROJECT_DIR
            / ".runtime"
            / "catboost-identity-screen"
            / "complete-deferred-identities.joblib"
        ).evaluation,
    }
    for name, evaluation in evaluations.items():
        if evaluation.cross_validation_fingerprint != (
            partitioned.cross_validation_fingerprint
        ):
            raise ValueError(f"OOF component {name!r} uses different folds.")
    probabilities = blend_deep_archive(
        {
            name: evaluation.out_of_fold_probabilities.to_numpy(dtype="float64")
            for name, evaluation in evaluations.items()
        }
    )
    validate_probabilities(probabilities, len(partitioned.y_development))
    return probabilities


def _load_local_incumbent(partitioned):
    archive = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "deep-archive-confirmation"
        / "local-test-confirmation.joblib"
    )
    seed = joblib.load(
        PROJECT_DIR
        / ".runtime"
        / "deep-seed-20260824-confirmation"
        / "local-test-confirmation.joblib"
    )
    for name, payload in (("archive", archive), ("seed", seed)):
        if payload["local_test_fingerprint"] != partitioned.local_test_fingerprint:
            raise ValueError(f"Local {name} payload uses a different test set.")
        if not payload["local_test_ids"].reset_index(drop=True).equals(
            partitioned.local_test_ids.reset_index(drop=True)
        ):
            raise ValueError(f"Local {name} payload changed ID order.")
    archived = archive["confirmation"].probabilities
    probabilities = blend_deep_archive(
        {
            "deep_xgboost": seed["deep_xgboost"],
            "spatial_xgboost": archived["spatial_xgboost"],
            "random_forest": archived["random_forest"],
            "frequency_random_forest": archived["frequency_random_forest"],
            "spatial_random_forest": archived["spatial_random_forest"],
            "identity_catboost": archived["identity_catboost"],
        }
    )
    if not np.allclose(probabilities, seed["candidate"]):
        raise ValueError("Local incumbent reconstruction changed.")
    validate_probabilities(probabilities, len(partitioned.y_local_test))
    return probabilities


def _load_competition_incumbent(modelling_data):
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
    probabilities = blend_deep_archive(
        {
            "deep_xgboost": deep["probabilities"],
            "spatial_xgboost": archive["spatial_xgboost"],
            "random_forest": accepted["random_forest_probabilities"],
            "frequency_random_forest": archive["frequency_random_forest"],
            "spatial_random_forest": archive["spatial_random_forest"],
            "identity_catboost": identity["probabilities"],
        }
    )
    validate_probabilities(probabilities, len(modelling_data.X_competition))
    return probabilities


def _validate_incumbent_reconstruction(probabilities):
    incumbent = pd.read_csv(INCUMBENT_PATH)
    if not np.array_equal(
        hard_predictions(probabilities),
        incumbent["status_group"].to_numpy(),
    ):
        raise ValueError("Cached probabilities do not recreate the 0.8298 CSV.")


def _validate_checkpoint(
    checkpoint,
    *,
    fingerprint,
    expected_ids,
    expected_rows,
):
    if checkpoint.get("fingerprint") != fingerprint:
        raise ValueError("RealMLP checkpoint fingerprint changed.")
    if checkpoint.get("seed") != REALMLP_SEED:
        raise ValueError("RealMLP checkpoint seed changed.")
    if not checkpoint["prediction_ids"].reset_index(drop=True).equals(
        expected_ids.reset_index(drop=True)
    ):
        raise ValueError("RealMLP checkpoint prediction IDs changed.")
    validate_probabilities(checkpoint["probabilities"], expected_rows)


def _validate_device(device):
    if not device.startswith("cuda"):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA RealMLP was requested, but this isolated PyTorch build cannot "
            "see the GPU."
        )
    print(
        f"Using {torch.cuda.get_device_name(0)} with PyTorch {torch.__version__}.",
        flush=True,
    )


def _format_summary(summary):
    display = summary.copy()
    for column in display.columns:
        if any(token in column for token in ("accuracy", "recall", "precision")):
            display[column] = display[column].map(lambda value: f"{value:.4%}")
    return display.to_string()


def _json_scalar(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refit and replace otherwise valid RealMLP checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

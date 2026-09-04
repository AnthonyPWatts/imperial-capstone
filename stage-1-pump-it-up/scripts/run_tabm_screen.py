"""Run a two-fold, checkpointed TabM-dagger feasibility screen."""

from __future__ import annotations

import argparse
from importlib.metadata import version
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
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "tabm-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation
from data_partitioning import partition_modelling_data
from modelling_data import prepare_modelling_data
from realmlp_evaluation import blend_probabilities
from realmlp_evaluation import hard_predictions
from run_realmlp_screen import _load_oof_incumbent
from tabm_evaluation import CLASS_LABELS
from tabm_evaluation import TABM_BLEND_WEIGHTS
from tabm_evaluation import TABM_SEED
from tabm_evaluation import TABM_TOP_IDENTITY_VALUES
from tabm_evaluation import fit_predict_tabm


OFFICIAL_REPOSITORY_COMMIT = "28e47ae301c92ec37787dde1ce923a0793f405b4"


def main() -> None:
    args = _parse_arguments()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned = partition_modelling_data(modelling_data)
    cross_validation = make_cross_validation(partitioned)
    incumbent_oof = _load_oof_incumbent(partitioned)

    fold_probabilities = []
    fold_positions = []
    diagnostics = []
    for fold_number, (training_positions, validation_positions) in enumerate(
        cross_validation.split(),
        start=1,
    ):
        if fold_number > args.max_folds:
            break
        expected_ids = partitioned.development_ids.iloc[
            validation_positions
        ].reset_index(drop=True)
        checkpoint_path = RUNTIME_DIR / f"fold-{fold_number}.joblib"
        if checkpoint_path.exists() and not args.force:
            checkpoint = joblib.load(checkpoint_path)
            _validate_checkpoint(
                checkpoint,
                fingerprint=partitioned.cross_validation_fingerprint,
                expected_ids=expected_ids,
                expected_rows=len(validation_positions),
            )
            source = "checkpoint"
        else:
            fitted = fit_predict_tabm(
                partitioned.X_development.iloc[training_positions],
                partitioned.y_development.iloc[training_positions],
                partitioned.X_development.iloc[validation_positions],
                device=args.device,
                seed=TABM_SEED + fold_number - 1,
            )
            checkpoint = {
                "fingerprint": partitioned.cross_validation_fingerprint,
                "prediction_ids": expected_ids,
                "seed": TABM_SEED + fold_number - 1,
                "probabilities": fitted.probabilities,
                "fit_seconds": fitted.fit_seconds,
                "best_epoch": fitted.best_epoch,
                "best_inner_accuracy": fitted.best_inner_accuracy,
                "parameter_count": fitted.parameter_count,
                "category_cardinalities": fitted.category_cardinalities,
            }
            joblib.dump(checkpoint, checkpoint_path)
            source = "fitted"
        fold_probabilities.append(checkpoint["probabilities"])
        fold_positions.append(validation_positions)
        diagnostics.append(
            {
                "fold": fold_number,
                "source": source,
                "fit_seconds": float(checkpoint["fit_seconds"]),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_inner_accuracy": float(
                    checkpoint["best_inner_accuracy"]
                ),
                "standalone_outer_accuracy": float(
                    accuracy_score(
                        partitioned.y_development.iloc[validation_positions],
                        hard_predictions(checkpoint["probabilities"]),
                    )
                ),
                "parameter_count": int(checkpoint["parameter_count"]),
                "categorical_width": int(
                    sum(checkpoint["category_cardinalities"])
                ),
            }
        )
        print(
            f"Completed TabM fold {fold_number}/{args.max_folds}: "
            f"accuracy={diagnostics[-1]['standalone_outer_accuracy']:.4%}, "
            f"best_epoch={diagnostics[-1]['best_epoch']}, "
            f"fit={diagnostics[-1]['fit_seconds']:.1f}s ({source}).",
            flush=True,
        )

    positions = np.concatenate(fold_positions)
    tabm_probabilities = np.concatenate(fold_probabilities)
    incumbent_probabilities = incumbent_oof[positions]
    target = partitioned.y_development.iloc[positions].reset_index(drop=True)
    candidates = {"standalone": (1.0, tabm_probabilities)}
    for weight in TABM_BLEND_WEIGHTS:
        candidates[f"blend_{int(weight * 100):02d}"] = (
            weight,
            blend_probabilities(
                incumbent_probabilities,
                tabm_probabilities,
                weight,
            ),
        )
    summary = _summarise(
        target,
        positions,
        partitioned,
        incumbent_probabilities,
        candidates,
    )
    summary.to_csv(RUNTIME_DIR / "candidate-summary.csv")
    selected = _select_promising_candidate(summary)
    result = {
        "decision": (
            "promising_for_full_confirmation"
            if selected is not None
            else "reject_after_bounded_screen"
        ),
        "selected_candidate": selected,
        "completed_folds": args.max_folds,
        "tabm_version": version("tabm"),
        "rtdl_num_embeddings_version": version("rtdl_num_embeddings"),
        "official_repository_commit": OFFICIAL_REPOSITORY_COMMIT,
        "seed_first_fold": TABM_SEED,
        "model": "TabM-dagger official defaults",
        "representation": (
            "accepted numeric/categorical features plus fold-fitted top-50 "
            "values of each of six normalised identity fields"
        ),
        "blend_weights": list(TABM_BLEND_WEIGHTS),
        "local_holdout_opened": False,
        "competition_predictions_created": False,
        "diagnostics": diagnostics,
        "summary": {
            key: {column: _json_scalar(value) for column, value in row.items()}
            for key, row in summary.iterrows()
        },
    }
    (RUNTIME_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n" + summary.to_string(), flush=True)
    print("\n" + json.dumps(result, indent=2), flush=True)


def _summarise(
    target,
    positions,
    partitioned,
    incumbent,
    candidates,
):
    incumbent_labels = hard_predictions(incumbent)
    incumbent_correct = incumbent_labels == target.to_numpy()
    incumbent_accuracy = float(incumbent_correct.mean())
    incumbent_repair_recall = recall_score(
        target,
        incumbent_labels,
        labels=list(CLASS_LABELS),
        average=None,
        zero_division=0,
    )[1]
    fold_numbers = partitioned.validation_folds.iloc[positions].to_numpy()
    rows = []
    for name, (weight, probabilities) in candidates.items():
        labels = hard_predictions(probabilities)
        correct = labels == target.to_numpy()
        per_class_recall = recall_score(
            target,
            labels,
            labels=list(CLASS_LABELS),
            average=None,
            zero_division=0,
        )
        per_class_precision = precision_score(
            target,
            labels,
            labels=list(CLASS_LABELS),
            average=None,
            zero_division=0,
        )
        fold_nets = [
            int(correct[fold_numbers == fold].sum())
            - int(incumbent_correct[fold_numbers == fold].sum())
            for fold in sorted(set(fold_numbers))
        ]
        changed = labels != incumbent_labels
        functional_to_repair = (
            (incumbent_labels == "functional")
            & (labels == "functional needs repair")
        )
        repair_to_functional = (
            (incumbent_labels == "functional needs repair")
            & (labels == "functional")
        )
        rows.append(
            {
                "candidate": name,
                "tabm_weight": weight,
                "accuracy": float(correct.mean()),
                "accuracy_change": float(correct.mean() - incumbent_accuracy),
                "net_correct_rows": int(correct.sum() - incumbent_correct.sum()),
                "fold_nets": ",".join(map(str, fold_nets)),
                "fold_wins": int(sum(value > 0 for value in fold_nets)),
                "worst_fold_net": int(min(fold_nets)),
                "repair_recall": float(per_class_recall[1]),
                "repair_recall_change": float(
                    per_class_recall[1] - incumbent_repair_recall
                ),
                "repair_precision": float(per_class_precision[1]),
                "changed_predictions": int(changed.sum()),
                "changed_net_correct": int(
                    correct[changed].sum() - incumbent_correct[changed].sum()
                ),
                "functional_to_repair": int(functional_to_repair.sum()),
                "functional_to_repair_net": int(
                    correct[functional_to_repair].sum()
                    - incumbent_correct[functional_to_repair].sum()
                ),
                "repair_to_functional": int(repair_to_functional.sum()),
                "repair_to_functional_net": int(
                    correct[repair_to_functional].sum()
                    - incumbent_correct[repair_to_functional].sum()
                ),
            }
        )
    return pd.DataFrame(rows).set_index("candidate")


def _select_promising_candidate(summary):
    blends = summary.drop(index="standalone")
    eligible = blends.loc[
        (blends["net_correct_rows"] > 0)
        & (blends["worst_fold_net"] >= 0)
    ]
    if eligible.empty:
        return None
    ranked = eligible.sort_values(
        ["net_correct_rows", "repair_recall_change", "tabm_weight"],
        ascending=[False, False, True],
    )
    return str(ranked.index[0])


def _validate_checkpoint(
    checkpoint,
    *,
    fingerprint,
    expected_ids,
    expected_rows,
):
    if checkpoint.get("fingerprint") != fingerprint:
        raise ValueError("TabM checkpoint uses a different fold plan.")
    if checkpoint.get("seed") not in (TABM_SEED, TABM_SEED + 1):
        raise ValueError("TabM checkpoint seed changed.")
    if not checkpoint["prediction_ids"].reset_index(drop=True).equals(expected_ids):
        raise ValueError("TabM checkpoint prediction IDs changed.")
    probabilities = checkpoint["probabilities"]
    if probabilities.shape != (expected_rows, len(CLASS_LABELS)):
        raise ValueError("TabM checkpoint probability shape changed.")
    if not np.isfinite(probabilities).all():
        raise ValueError("TabM checkpoint probabilities are not finite.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("TabM checkpoint probabilities do not sum to one.")


def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-folds", type=int, default=2, choices=(1, 2))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _json_scalar(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


if __name__ == "__main__":
    main()

"""Build the bounded probability-combination screen from persisted OOF trials."""

from __future__ import annotations

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
RUNTIME_DIR = PROJECT_DIR / ".runtime" / "gpu-model-screen"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import make_cross_validation, partition_modelling_data
from gpu_model_evaluation import CATBOOST_FAMILY, LIGHTGBM_FAMILY
from gpu_model_evaluation import INNER_STOP_SEED
from gpu_model_evaluation import SKLEARN_TREE_FAMILY
from gpu_model_evaluation import XGBOOST_FAMILY
from model_evaluation import CandidateEvaluation
from model_evaluation import evaluate_equal_weight_soft_vote
from model_evaluation import evaluate_weighted_soft_vote
from modelling_data import prepare_modelling_data


MEAN_GAIN_GATE = 0.001
FOLD_WIN_GATE = 3
WORST_FOLD_GATE = -0.0025
REPAIR_RECALL_LOSS_GATE = -0.02
SECOND_WAVE_MEAN_GAIN_GATE = 0.0005
CURRENT_LEADER_NAME = (
    "60% XGBoost depth 8 [current one-hot] + 40% Random Forest"
)


def main() -> None:
    (
        incumbent,
        incumbent_components,
        catboost_trials,
        xgboost_trials,
        lightgbm_trials,
        sklearn_tree_trials,
    ) = _load_trials()
    partitioned_data, cross_validation = _load_partition()
    evaluations: dict[str, CandidateEvaluation] = {
        incumbent.model_name: incumbent,
        **{
            evaluation.model_name: evaluation
            for evaluation in incumbent_components.values()
        },
        **{evaluation.model_name: evaluation for _, evaluation in catboost_trials},
        **{evaluation.model_name: evaluation for _, evaluation in xgboost_trials},
        **{evaluation.model_name: evaluation for _, evaluation in lightgbm_trials},
        **{
            evaluation.model_name: evaluation
            for _, evaluation in sklearn_tree_trials
        },
    }
    recipes = {
        name: ((name, 1.0),)
        for name in evaluations
    }

    best_cat = _best(catboost_trials)
    base_xgboost_trials = [
        trial for trial in xgboost_trials if trial[0].seed == INNER_STOP_SEED
    ]
    best_xgb = _best(base_xgboost_trials)
    best_lgb = _best(lightgbm_trials)
    new_trials = (
        catboost_trials
        + xgboost_trials
        + lightgbm_trials
        + sklearn_tree_trials
    )
    for _, evaluation in new_trials:
        for component_label, component in incumbent_components.items():
            weights = (0.40, 0.50, 0.60)
            if (
                component_label == "Random Forest"
                and any(
                    spec.family == XGBOOST_FAMILY
                    and candidate.model_name == evaluation.model_name
                    for spec, candidate in xgboost_trials
                )
            ):
                weights = (0.40, 0.50, 0.55, 0.60, 0.65, 0.70)
            _add_pair_grid(
                evaluations,
                recipes,
                partitioned_data,
                cross_validation,
                evaluation,
                component,
                left_label=evaluation.model_name,
                right_label=component_label,
                left_weights=weights,
            )
    selected_depth8 = next(
        (
            evaluation
            for spec, evaluation in base_xgboost_trials
            if spec.variant == "depth 8"
        ),
        None,
    )
    for _, tree_evaluation in sklearn_tree_trials:
        for _, xgb_evaluation in base_xgboost_trials:
            _add_pair_grid(
                evaluations,
                recipes,
                partitioned_data,
                cross_validation,
                tree_evaluation,
                xgb_evaluation,
                left_label=tree_evaluation.model_name,
                right_label=xgb_evaluation.model_name,
                left_weights=(0.20, 0.30, 0.40, 0.50),
            )
        if selected_depth8 is not None:
            random_forest = incumbent_components["Random Forest"]
            for tree_weight, forest_weight in (
                (0.10, 0.35),
                (0.15, 0.35),
                (0.20, 0.30),
                (0.20, 0.35),
            ):
                xgb_weight = 1.0 - tree_weight - forest_weight
                _add_weighted_recipe(
                    evaluations,
                    recipes,
                    partitioned_data,
                    cross_validation,
                    components=(
                        selected_depth8,
                        random_forest,
                        tree_evaluation,
                    ),
                    weights=(xgb_weight, forest_weight, tree_weight),
                    model_name=(
                        f"{xgb_weight:.0%} XGBoost depth 8 + "
                        f"{forest_weight:.0%} Random Forest + "
                        f"{tree_weight:.0%} {tree_evaluation.model_name}"
                    ),
                )
    for _, lgb_evaluation in lightgbm_trials:
        for _, xgb_evaluation in base_xgboost_trials:
            _add_pair_grid(
                evaluations,
                recipes,
                partitioned_data,
                cross_validation,
                lgb_evaluation,
                xgb_evaluation,
                left_label=lgb_evaluation.model_name,
                right_label=xgb_evaluation.model_name,
                left_weights=(0.20, 0.30, 0.40),
            )
        if selected_depth8 is not None:
            random_forest = incumbent_components["Random Forest"]
            for lgb_weight, forest_weight in (
                (0.10, 0.35),
                (0.15, 0.35),
                (0.20, 0.30),
                (0.20, 0.35),
            ):
                xgb_weight = 1.0 - lgb_weight - forest_weight
                _add_weighted_recipe(
                    evaluations,
                    recipes,
                    partitioned_data,
                    cross_validation,
                    components=(
                        selected_depth8,
                        random_forest,
                        lgb_evaluation,
                    ),
                    weights=(xgb_weight, forest_weight, lgb_weight),
                    model_name=(
                        f"{xgb_weight:.0%} XGBoost depth 8 + "
                        f"{forest_weight:.0%} Random Forest + "
                        f"{lgb_weight:.0%} {lgb_evaluation.model_name}"
                    ),
                )
    if best_cat is not None:
        _add_pair_grid(
            evaluations,
            recipes,
            partitioned_data,
            cross_validation,
            best_cat,
            incumbent,
            left_label="CatBoost",
            right_label="incumbent",
            left_weights=(0.25, 0.40, 0.50, 0.60, 0.75),
        )
    if best_xgb is not None:
        _add_pair_grid(
            evaluations,
            recipes,
            partitioned_data,
            cross_validation,
            best_xgb,
            incumbent,
            left_label="XGBoost",
            right_label="incumbent",
            left_weights=(0.40, 0.50, 0.60),
        )
    if best_lgb is not None:
        _add_pair_grid(
            evaluations,
            recipes,
            partitioned_data,
            cross_validation,
            best_lgb,
            incumbent,
            left_label="LightGBM",
            right_label="incumbent",
            left_weights=(0.25, 0.40, 0.50),
        )
    if best_cat is not None and best_xgb is not None:
        _add_pair_grid(
            evaluations,
            recipes,
            partitioned_data,
            cross_validation,
            best_cat,
            best_xgb,
            left_label="CatBoost",
            right_label="XGBoost",
            left_weights=(0.40, 0.50, 0.60),
        )
        three_family = evaluate_equal_weight_soft_vote(
            partitioned_data,
            cross_validation,
            best_cat,
            best_xgb,
            incumbent,
            model_name="33% CatBoost + 33% XGBoost + 33% incumbent",
        )
        evaluations[three_family.model_name] = three_family
        recipes[three_family.model_name] = (
            (best_cat.model_name, 1 / 3),
            (best_xgb.model_name, 1 / 3),
            (incumbent.model_name, 1 / 3),
        )
    if best_lgb is not None and best_xgb is not None:
        lgb_xgb = evaluate_equal_weight_soft_vote(
            partitioned_data,
            cross_validation,
            best_lgb,
            best_xgb,
            model_name="50% LightGBM + 50% XGBoost",
        )
        evaluations[lgb_xgb.model_name] = lgb_xgb
        recipes[lgb_xgb.model_name] = (
            (best_lgb.model_name, 0.5),
            (best_xgb.model_name, 0.5),
        )
    if best_cat is not None and best_lgb is not None:
        _add_pair_grid(
            evaluations,
            recipes,
            partitioned_data,
            cross_validation,
            best_cat,
            best_lgb,
            left_label="CatBoost",
            right_label="LightGBM",
            left_weights=(0.40, 0.50, 0.60),
        )
    if best_cat is not None and best_xgb is not None and best_lgb is not None:
        boosted_families = evaluate_equal_weight_soft_vote(
            partitioned_data,
            cross_validation,
            best_cat,
            best_xgb,
            best_lgb,
            model_name="equal CatBoost + XGBoost + LightGBM",
        )
        evaluations[boosted_families.model_name] = boosted_families
        recipes[boosted_families.model_name] = (
            (best_cat.model_name, 1 / 3),
            (best_xgb.model_name, 1 / 3),
            (best_lgb.model_name, 1 / 3),
        )
        all_families = evaluate_equal_weight_soft_vote(
            partitioned_data,
            cross_validation,
            best_cat,
            best_xgb,
            best_lgb,
            incumbent,
            model_name="equal CatBoost + XGBoost + LightGBM + incumbent",
        )
        evaluations[all_families.model_name] = all_families
        recipes[all_families.model_name] = (
            (best_cat.model_name, 0.25),
            (best_xgb.model_name, 0.25),
            (best_lgb.model_name, 0.25),
            (incumbent.model_name, 0.25),
        )

    cat_bags = _group_seed_bags(
        catboost_trials,
        family_label="CatBoost",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
    )
    for name, (evaluation, recipe) in cat_bags.items():
        evaluations[name] = evaluation
        recipes[name] = recipe
    xgb_seed_bags = _group_seed_bags(
        xgboost_trials,
        family_label="XGBoost",
        partitioned_data=partitioned_data,
        cross_validation=cross_validation,
    )
    for name, (evaluation, recipe) in xgb_seed_bags.items():
        evaluations[name] = evaluation
        recipes[name] = recipe
        for component_label, component in incumbent_components.items():
            _add_pair_grid(
                evaluations,
                recipes,
                partitioned_data,
                cross_validation,
                evaluation,
                component,
                left_label=name,
                right_label=component_label,
                left_weights=(0.40, 0.50, 0.55, 0.60, 0.65, 0.70),
            )
    initial_variants = {
        "depth 8",
        "depth 11 conservative",
        "lossguide 64",
    }
    second_wave_variants = {
        "depth 6 broad",
        "depth 7",
        "depth 9 regularised",
        "depth 8 max bin 512",
        "depth 8 full rows",
        "depth 8 child 1",
        "depth 8 child 5",
        "depth 8 columns 0.75",
        "depth 8 rows 0.80",
    }
    xgb_bag_groups = (
        (
            "equal XGBoost variant bag",
            [
                trial
                for trial in base_xgboost_trials
                if trial[0].variant in initial_variants
            ],
        ),
        (
            "equal second-wave XGBoost variant bag",
            [
                trial
                for trial in base_xgboost_trials
                if trial[0].variant in second_wave_variants
            ],
        ),
        (
            "equal XGBoost depth 6/7/8 local bag",
            [
                trial
                for trial in base_xgboost_trials
                if trial[0].variant in {"depth 6 broad", "depth 7", "depth 8"}
            ],
        ),
    )
    for bag_name, trials in xgb_bag_groups:
        if len(trials) < 2:
            continue
        xgb_bag = evaluate_equal_weight_soft_vote(
            partitioned_data,
            cross_validation,
            *(evaluation for _, evaluation in trials),
            model_name=bag_name,
        )
        evaluations[xgb_bag.model_name] = xgb_bag
        equal_weight = 1 / len(trials)
        recipes[xgb_bag.model_name] = tuple(
            (evaluation.model_name, equal_weight)
            for _, evaluation in trials
        )
        for component_label, component in incumbent_components.items():
            _add_pair_grid(
                evaluations,
                recipes,
                partitioned_data,
                cross_validation,
                xgb_bag,
                component,
                left_label=bag_name.removeprefix("equal "),
                right_label=component_label,
                left_weights=(
                    (0.40, 0.50, 0.55, 0.60, 0.65, 0.70)
                    if component_label == "Random Forest"
                    else (0.40, 0.50, 0.60)
                ),
            )

    if CURRENT_LEADER_NAME not in evaluations:
        raise KeyError(f"Current leader is missing: {CURRENT_LEADER_NAME!r}.")
    current_leader = evaluations[CURRENT_LEADER_NAME]
    summary = _summarise(evaluations, incumbent, current_leader)
    summary.to_csv(RUNTIME_DIR / "combination-summary.csv")
    joblib.dump(
        {
            "evaluations": evaluations,
            "recipes": recipes,
            "summary": summary,
            "incumbent_name": incumbent.model_name,
            "current_leader_name": current_leader.model_name,
        },
        RUNTIME_DIR / "combination-screen.joblib",
    )
    print(summary.head(20).to_string())
    print(
        "\nGate: mean gain >= 0.10 pp, at least 3/5 fold wins, "
        "worst fold >= -0.25 pp, repair recall loss <= 2 pp."
    )
    print(f"Passing trials: {int(summary['passes_gate'].sum())}")
    print(
        "Second-wave leader gate-passers: "
        f"{int(summary['beats_current_leader_gate'].sum())}"
    )


def _load_trials():
    incumbent = None
    incumbent_components = None
    catboost_trials = []
    xgboost_trials = []
    lightgbm_trials = []
    sklearn_tree_trials = []
    for path in sorted(RUNTIME_DIR.glob("*.joblib")):
        if path.name == "combination-screen.joblib":
            continue
        payload = joblib.load(path)
        evaluation = payload["evaluation"]
        spec = payload["spec"]
        if spec is None:
            incumbent = evaluation
            incumbent_components = payload["components"]
        elif spec.family == CATBOOST_FAMILY:
            catboost_trials.append((spec, evaluation))
        elif spec.family == XGBOOST_FAMILY:
            xgboost_trials.append((spec, evaluation))
        elif spec.family == LIGHTGBM_FAMILY:
            lightgbm_trials.append((spec, evaluation))
        elif spec.family == SKLEARN_TREE_FAMILY:
            sklearn_tree_trials.append((spec, evaluation))
    if incumbent is None:
        raise FileNotFoundError("The persisted incumbent OOF trial is missing.")
    if not incumbent_components:
        raise ValueError("The incumbent component evaluations are missing.")
    return (
        incumbent,
        incumbent_components,
        catboost_trials,
        xgboost_trials,
        lightgbm_trials,
        sklearn_tree_trials,
    )


def _load_partition():
    modelling_data = prepare_modelling_data(
        pd.read_csv(DATA_DIR / "TrainingSetValues.csv"),
        pd.read_csv(DATA_DIR / "TrainingSetLabels.csv"),
        pd.read_csv(DATA_DIR / "TestSetValues.csv"),
    )
    partitioned_data = partition_modelling_data(modelling_data)
    return partitioned_data, make_cross_validation(partitioned_data)


def _best(trials):
    if not trials:
        return None
    return max(
        (evaluation for _, evaluation in trials),
        key=lambda evaluation: evaluation.metric_summary.loc["accuracy", "mean"],
    )


def _add_pair_grid(
    evaluations,
    recipes,
    partitioned_data,
    cross_validation,
    left,
    right,
    *,
    left_label,
    right_label,
    left_weights,
):
    for left_weight in left_weights:
        right_weight = 1.0 - left_weight
        name = (
            f"{left_weight:.0%} {left_label} + "
            f"{right_weight:.0%} {right_label}"
        )
        evaluation = evaluate_weighted_soft_vote(
            partitioned_data,
            cross_validation,
            left,
            right,
            weights=[left_weight, right_weight],
            model_name=name,
        )
        evaluations[name] = evaluation
        recipes[name] = (
            (left.model_name, left_weight),
            (right.model_name, right_weight),
        )


def _add_weighted_recipe(
    evaluations,
    recipes,
    partitioned_data,
    cross_validation,
    *,
    components,
    weights,
    model_name,
):
    evaluation = evaluate_weighted_soft_vote(
        partitioned_data,
        cross_validation,
        *components,
        weights=weights,
        model_name=model_name,
    )
    evaluations[model_name] = evaluation
    recipes[model_name] = tuple(
        (component.model_name, weight)
        for component, weight in zip(components, weights, strict=True)
    )


def _group_seed_bags(
    trials,
    *,
    family_label,
    partitioned_data,
    cross_validation,
):
    by_variant = {}
    for spec, evaluation in trials:
        by_variant.setdefault(spec.variant, []).append(evaluation)
    bags = {}
    for variant, evaluations in by_variant.items():
        if len(evaluations) < 2:
            continue
        name = f"equal {family_label} {variant} seed bag"
        bag = evaluate_equal_weight_soft_vote(
            partitioned_data,
            cross_validation,
            *evaluations,
            model_name=name,
        )
        weight = 1 / len(evaluations)
        bags[name] = (
            bag,
            tuple((evaluation.model_name, weight) for evaluation in evaluations),
        )
    return bags


def _summarise(evaluations, incumbent, current_leader):
    incumbent_accuracy = incumbent.fold_metrics["accuracy"]
    incumbent_repair = incumbent.metric_summary.loc[
        "recall: functional needs repair",
        "mean",
    ]
    incumbent_predictions = _predictions(incumbent)
    leader_accuracy = current_leader.fold_metrics["accuracy"]
    leader_repair = current_leader.metric_summary.loc[
        "recall: functional needs repair",
        "mean",
    ]
    leader_predictions = _predictions(current_leader)
    rows = []
    for name, evaluation in evaluations.items():
        fold_accuracy = evaluation.fold_metrics["accuracy"]
        fold_change = fold_accuracy - incumbent_accuracy
        mean_accuracy = fold_accuracy.mean()
        mean_gain = mean_accuracy - incumbent_accuracy.mean()
        repair_recall = evaluation.metric_summary.loc[
            "recall: functional needs repair",
            "mean",
        ]
        repair_change = repair_recall - incumbent_repair
        disagreement = np.mean(_predictions(evaluation) != incumbent_predictions)
        leader_fold_change = fold_accuracy - leader_accuracy
        leader_gain = mean_accuracy - leader_accuracy.mean()
        leader_repair_change = repair_recall - leader_repair
        leader_disagreement = np.mean(
            _predictions(evaluation) != leader_predictions
        )
        is_incumbent = name == incumbent.model_name
        is_current_leader = name == current_leader.model_name
        is_single_seed_diagnostic = (
            " seed " in name and " seed bag" not in name
        )
        passes = (
            not is_incumbent
            and not is_single_seed_diagnostic
            and mean_gain >= MEAN_GAIN_GATE
            and int(fold_change.gt(0).sum()) >= FOLD_WIN_GATE
            and float(fold_change.min()) >= WORST_FOLD_GATE
            and repair_change >= REPAIR_RECALL_LOSS_GATE
        )
        beats_current_leader = (
            not is_current_leader
            and not is_single_seed_diagnostic
            and leader_gain >= SECOND_WAVE_MEAN_GAIN_GATE
            and int(leader_fold_change.gt(0).sum()) >= FOLD_WIN_GATE
            and float(leader_fold_change.min()) >= WORST_FOLD_GATE
            and leader_repair_change >= REPAIR_RECALL_LOSS_GATE
        )
        rows.append(
            {
                "trial": name,
                "mean_accuracy": mean_accuracy,
                "mean_gain": mean_gain,
                "fold_wins": int(fold_change.gt(0).sum()),
                "worst_fold_change": float(fold_change.min()),
                "repair_recall": repair_recall,
                "repair_recall_change": repair_change,
                "non_functional_recall": evaluation.metric_summary.loc[
                    "recall: non functional",
                    "mean",
                ],
                "incumbent_disagreement": disagreement,
                "current_leader_gain": leader_gain,
                "current_leader_fold_wins": int(
                    leader_fold_change.gt(0).sum()
                ),
                "current_leader_worst_fold_change": float(
                    leader_fold_change.min()
                ),
                "current_leader_repair_recall_change": leader_repair_change,
                "current_leader_disagreement": leader_disagreement,
                "selection_eligible": not is_single_seed_diagnostic,
                "passes_gate": passes,
                "beats_current_leader_gate": beats_current_leader,
            }
        )
    return (
        pd.DataFrame(rows)
        .set_index("trial")
        .sort_values("mean_accuracy", ascending=False)
    )


def _predictions(evaluation):
    return np.asarray(evaluation.out_of_fold_probabilities).argmax(axis=1)


if __name__ == "__main__":
    main()

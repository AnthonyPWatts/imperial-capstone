"""Function 1's target scale, shared by exploration and model notebooks.

The default scale matches the Week 4 model study. Choosing a height scale does
not round, clip or replace the measured outputs.
"""

import numpy as np


def signed_log(values, scale=1e-200):
    """Preserve sign and ordering on the fixed height scale used in exploration."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values) / scale) / np.log(10.0)


def make_review_targets(raw_outputs, *, height_scale):
    """Keep the signed objective distinct from the separately refitted activity target."""
    signed = signed_log(raw_outputs, scale=height_scale)
    return {"signed": signed, "magnitude": np.abs(signed)}


def run_target_reviews(models, inputs, raw_outputs, planned_point, *, input_columns,
                        height_scale, gp_names, selection_metric="LOO MAE",
                        candidate_settings=None, weight=1.96, weights=(0., 1., 1.96, 3.)):
    """Fit both existing target hypotheses and score hypothetical alternatives.

    All calculations accept arbitrary input counts. Model scores and uncertainty
    remain in transformed target units; no inverse-transformed raw mean is claimed.
    The saved coverage point is scored separately and is never replaced here.
    """
    from model_helpers import make_query_candidates, score_query_candidates
    from model_review_helpers import fit_model_review, query_prediction, weight_sensitivity
    import pandas as pd

    settings = dict(candidate_settings or {})
    precision = settings.get("precision", 6)
    targets = make_review_targets(raw_outputs, height_scale=height_scale)
    candidates = make_query_candidates(inputs, **settings)
    result = dict(targets=targets, candidates=candidates, runs={}, scored={},
                  proposals={}, planned_predictions={}, sensitivity={}, gp_sensitivity={})
    for name, target in targets.items():
        run = fit_model_review(models, inputs, target, gp_names=gp_names, selection_metric=selection_metric)
        scored = score_query_candidates(run["selected_model"], candidates, input_columns=input_columns,
                                         weight=weight, precision=precision)
        point = scored.iloc[0][list(input_columns)].to_numpy(dtype=float)
        result["runs"][name], result["scored"][name] = run, scored
        result["proposals"][name] = query_prediction(run["selected_model"], point, inputs, target,
                                                      weight=weight, precision=precision)
        result["planned_predictions"][name] = query_prediction(run["selected_model"], planned_point, inputs, target,
                                                                weight=weight, precision=precision)
        result["sensitivity"][name] = weight_sensitivity(scored, inputs, input_columns=input_columns,
            modelled_axes=tuple(range(len(input_columns))), weights=weights, precision=precision)
        alternatives = []
        for gp_name in gp_names:
            gp_scores = (scored if gp_name == run["selected_name"] else score_query_candidates(
                run["fitted"][gp_name], candidates, input_columns=input_columns, weight=weight, precision=precision))
            table = (result["sensitivity"][name].copy() if gp_name == run["selected_name"] else weight_sensitivity(
                gp_scores, inputs, input_columns=input_columns, modelled_axes=tuple(range(len(input_columns))),
                weights=weights, precision=precision))
            table.insert(0, "GP", gp_name)
            alternatives.append(table)
        result["gp_sensitivity"][name] = pd.concat(alternatives, ignore_index=True)
    return result


def summarise_target_reviews(study, *, target_labels):
    """Retain unit-labelled results and keep the actual plan outside hypothetical policies."""
    result = {}
    for name, run in study["runs"].items():
        result[name] = dict(target_units=target_labels[name], selected_model=run["selected_name"],
            selection_metric=run["selection_metric"], selected_mae=run["selected_mae"],
            baseline_mae=run["baseline_mae"], selected_rmse=run["selected_rmse"],
            baseline_rmse=run["baseline_rmse"], mae_baseline_ratio=run["selected_mae"]/run["baseline_mae"],
            model_comparison=run["table"].to_dict(orient="records"),
            hypothetical=study["proposals"][name], saved_coverage=study["planned_predictions"][name],
            candidate_count=len(study["candidates"]),
            weight_sensitivity=study["sensitivity"][name].to_dict(orient="records"),
            all_gp_weight_sensitivity=study["gp_sensitivity"][name].to_dict(orient="records"))
    return result


def validate_saved_target_review(saved, current, *, observation_count, planned_portal_input):
    """Reject stale official evidence before comparing it with a pooled refit."""
    if (saved["observations"] != observation_count
            or saved["planned_portal_input"] != planned_portal_input
            or saved["target_reviews"].keys() != current.keys()):
        raise ValueError("Saved official targets or coverage plan do not match the refitted snapshot")
    for name, review in current.items():
        old = saved["target_reviews"][name]
        if (old["selected_model"] != review["selected_model"]
                or old["target_units"] != review["target_units"]
                or old["selection_metric"] != review["selection_metric"]
                or old["hypothetical"]["portal_input"] != review["hypothetical"]["portal_input"]
                or not np.allclose([old["selected_mae"], old["baseline_mae"], old["selected_rmse"]],
                                   [review["selected_mae"], review["baseline_mae"], review["selected_rmse"]],
                                   rtol=1e-10, atol=1e-12)):
            raise ValueError("Saved official target review does not match the refitted snapshot")


def compare_cohort_targets(own_study, pooled_study, pooled_inputs, *,
                            own_positions, peer_count, planned_point, input_columns,
                            weight=1.96, precision=6):
    """Compare identical transformed targets, symmetric candidate sets and the saved plan."""
    from cohort_review_helpers import paired_target_comparison, common_candidate_comparison, compare_proposal_predictions

    result = {}
    for name, own_run in own_study["runs"].items():
        fits = {"Official fit": own_run, "Pooled fit": pooled_study["runs"][name]}
        proposals = {"Saved coverage query": planned_point,
            "Official hypothetical": np.asarray(own_study["proposals"][name]["point"]),
            "Cohort hypothetical": np.asarray(pooled_study["proposals"][name]["point"])}
        paired = paired_target_comparison(own_run, pooled_study["runs"][name], own_study["targets"][name],
                                          own_positions=own_positions, peer_count=peer_count)
        common = common_candidate_comparison(fits,
            [own_study["candidates"], pooled_study["candidates"], *proposals.values()], pooled_inputs,
            input_columns=input_columns, weight=weight, precision=precision)
        predictions = compare_proposal_predictions(fits, proposals, input_columns=input_columns, weight=weight)
        result[name] = dict(paired_targets=paired, common_candidates=common, proposal_predictions=predictions)
    return result


def save_target_model_views(study, inputs, anchor, planned_point, latest_mask, *,
                             input_columns, gp_names, target_labels, title,
                             figures_dir, grid_side=61, weight=1.96):
    """Export explicit two-input views with transformed units and the retained plan marker."""
    from pathlib import Path
    from model_review_plots import plot_held_out_comparison, plot_coordinate_model_review
    from two_input_model_helpers import prediction_grid, plot_contour_comparison, plot_gp_maps

    figures = []
    for name, run in study["runs"].items():
        target = study["targets"][name]
        label = target_labels[name]
        destination = Path(figures_dir)
        figures.append(plot_held_out_comparison(run["predictions"], target,
            model_names=["Mean baseline", "Quadratic ridge", run["selected_name"]],
            latest_mask=latest_mask, title=f"{title}: held-out {label}", target_label=label,
            save_to=destination/f"{name}-model-validation.png"))
        figures.append(plot_coordinate_model_review({key: run["fitted"][key] for key in gp_names},
            inputs, target, anchor, input_columns=input_columns, target_label=label,
            title=f"{title}: conditional {label}", figsize=(12, 8),
            save_to=destination/f"{name}-gp-coordinate-slices.png"))
        grid = prediction_grid(run["selected_model"], inputs, side=grid_side, uncertainty=True)
        reference = prediction_grid(run["fitted"]["Quadratic ridge"], inputs, side=grid_side)
        figures.append(plot_contour_comparison(inputs,
            [("Quadratic ridge", reference, target), (run["selected_name"], grid, target)],
            title=f"{title}: fitted {label}", target_label=label,
            save_to=destination/f"{name}-fitted-contours.png"))
        figures.append(plot_gp_maps(grid, inputs, np.asarray(planned_point)[None, :],
            weight=weight, target_label=label, suggestion=np.asarray(study["proposals"][name]["point"]),
            title=f"{title}: hypothetical {label} policy; saved coverage retained",
            save_to=destination/f"{name}-model-query-proposal.png"))
    return figures

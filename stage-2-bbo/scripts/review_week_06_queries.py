"""Review saved Week 6 fits and prepare eight inputs without portal submission.

This dated review reuses fitted model exports and saved held-out predictions.
It does not fit more models or adapt the neural architecture/seed budget.
"""

import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import cdist

STAGE = Path(__file__).resolve().parents[1]
ROOT = STAGE.parent
sys.path.insert(0, str(STAGE / "notebooks/Shared"))
from neural_explorer import gp_prediction, network_prediction
from prediction_review_helpers import make_candidate_review

DIMENSIONS = (2, 2, 3, 4, 4, 5, 6, 8)
F5_LOCAL_RADIUS = .15
F5_MIN_DISTANCE = .02


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_review(function):
    """Read saved outputs and require agreement between their original source hashes."""
    week = STAGE / f"notebooks/Function_{function}/Week_06"
    folder = week / "official" if function <= 4 else week
    summary_path = folder / "model-review.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    csv_path = folder / "observations.csv"
    if fingerprint(csv_path) != summary["snapshot_sha256"]:
        raise ValueError("Week 6 official snapshot changed after model fitting")
    data = pd.read_csv(csv_path, float_precision="round_trip")
    explorer_path = folder / "figures/neural-network-explorer.html"
    html = explorer_path.read_text(encoding="utf-8")
    match = re.search(r'<script id="model-data" type="application/json">(.*?)</script>', html, re.S)
    if match is None:
        raise ValueError("Expected one saved explorer model payload")
    fitted = json.loads(match[1])
    if (summary["function"] != function or summary["week"] != 6
            or len(fitted["input_columns"]) != DIMENSIONS[function-1]
            or fitted["proposal"] != summary["point"]):
        raise ValueError("Saved model/explorer identity mismatch")
    source_paths = [csv_path, summary_path, folder / "model_predictions.ipynb", explorer_path]
    known = data[fitted["input_columns"]].to_numpy()
    if function <= 4:
        pooled_path = week / "cohort/observations.csv"
        audit_path = week / "cohort/data-audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (fingerprint(pooled_path) != audit["cohort_csv_sha256"]
                or fingerprint(audit_path) != summary["cohort_audit_sha256"]):
            raise ValueError("Retained peer training evidence changed")
        pooled = pd.read_csv(pooled_path, float_precision="round_trip")
        own = pooled.loc[~pooled.observation_id.str.startswith("peer-")].reset_index(drop=True)
        pd.testing.assert_frame_equal(own, data, check_exact=True)
        known = pooled[fitted["input_columns"]].to_numpy()
        source_paths += [pooled_path, audit_path]
    sources = [{"path": path.relative_to(ROOT).as_posix(), "sha256": fingerprint(path)} for path in source_paths]
    return dict(summary=summary, data=data, fitted=fitted, known=known, sources=sources)


def score_points(review, points):
    """Keep GP latent SD distinct from variation across fitted neural seeds."""
    points = np.atleast_2d(np.asarray(points, float))
    mean, sd = gp_prediction(review["fitted"]["gp"], points)
    architecture = review["summary"]["selected_neural_architecture"]
    specs = review["fitted"]["networks"][architecture]
    neural = np.array([network_prediction(spec, points) for spec in specs])
    return dict(gp_mean=mean, gp_sd=sd, gp_ucb=mean+1.96*sd,
                neural_mean=neural.mean(axis=0), neural_seed_sd=neural.std(axis=0, ddof=1),
                neural_seed_predictions=neural.T)


def point_record(review, point):
    point = np.asarray(point, float)
    scores = score_points(review, point)
    training = (review["known"] if review["summary"]["training_scope"] != "official" else
                review["data"][review["fitted"]["input_columns"]].to_numpy())
    try:
        inside = bool(Delaunay(training).find_simplex(point) >= 0)
    except QhullError:
        inside = None
    return dict(point=point.tolist(), portal_input="-".join(f"{x:.6f}" for x in point),
                gp_mean=float(scores["gp_mean"][0]), gp_latent_sd=float(scores["gp_sd"][0]),
                gp_ucb_1_96=float(scores["gp_ucb"][0]), neural_mean=float(scores["neural_mean"][0]),
                neural_seed_sd=float(scores["neural_seed_sd"][0]),
                neural_seed_predictions=scores["neural_seed_predictions"][0].tolist(),
                inside_selected_gp_training_hull=inside,
                nearest_known_input_distance=float(cdist(point[None], review["known"]).min()))


def prediction_checks(review):
    """Review existing identical held-out targets, with descriptive region sensitivity."""
    summary, data = review["summary"], review["data"]
    y = data.y.to_numpy()
    gp = np.asarray(summary["held_out_predictions"][summary["query_model"]])
    neural = np.mean([summary["held_out_predictions"][f"NN {summary['selected_neural_architecture']} seed {seed}"]
                      for seed in summary["neural_seeds"]], axis=0)
    gp_error, neural_error = gp-y, neural-y
    weekly = data.source_round.gt(0).to_numpy()
    return dict(gp_rmse=float(np.sqrt(np.mean(gp_error**2))), neural_rmse=float(np.sqrt(np.mean(neural_error**2))),
                gp_mae=float(np.abs(gp_error).mean()), neural_mae=float(np.abs(neural_error).mean()),
                gp_weekly_target_rmse=float(np.sqrt(np.mean(gp_error[weekly]**2))),
                neural_weekly_target_rmse=float(np.sqrt(np.mean(neural_error[weekly]**2))),
                weekly_target_count=int(weekly.sum()),
                neural_smaller_absolute_error_cases=int((abs(neural_error)<abs(gp_error)).sum()),
                scoring_cases=len(y), gp_latest_held_out_error=float(gp_error[-1]),
                neural_latest_held_out_error=float(neural_error[-1]),
                note="Subset scoring reuses the same retrospective LOO fits; it is not new temporal validation.")


def review_f1(review):
    """Keep the agreed central hunch separate from a demonstrated source detection."""
    greedy = next(row for row in review["summary"]["weight_sensitivity"] if row["Weight"] == 0)
    point = np.asarray([greedy[column] for column in review["fitted"]["input_columns"]])
    if not np.array_equal(point, [.537942, .522217]):
        raise ValueError("The agreed Function 1 central candidate no longer matches the saved fit")
    details = dict(
        decision="hunch_led_central_signed_mean", effective_ucb_weight=0.,
        rationale="Follow through on the central-region hunch with the saved signed-log GP mean maximum. "
                  "The nearby positive peer report motivates a query of our own; magnitude studies inform "
                  "the hypothesis rather than replace the signed-score objective. This is not an established source detection.",
        signal_assessment="None of the outputs so far is regarded as clearly non-zero in practical or source-detection terms. "
                          "The stored values are numerically non-zero and remain unchanged; no background or significance threshold has been established.",
        outputs_regarded_as_clearly_nonzero=False, numerical_outputs_modified=False,
        limitations="The choice depends substantially on a self-reported positive peer reading. The official-only neural fit "
                    "disagrees on sign. GP uncertainty is conditional on the fitted kernel; no raw expected score or positive return is guaranteed.",
        alternatives=dict(original_gp_ucb=point_record(review, review["summary"]["point"]),
                          agreed_central_signed_mean=point_record(review, point)))
    return point, details


def review_f5(review):
    """Choose one global information query, outside the existing local-search box.

    Maximise fitted GP latent SD first; geometric distance and coordinates break
    exact ties. High expected output is deliberately not this query's objective.
    """
    summary, fitted = review["summary"], review["fitted"]
    columns = fitted["input_columns"]
    anchor = np.asarray(fitted["incumbent"])
    pool, coverage = make_candidate_review(review["known"], anchor, input_columns=columns,
        search_settings=summary["search_settings"], minimum_distance=F5_MIN_DISTANCE)
    scored = score_points(review, pool[columns].to_numpy())
    pool["gp_latent_sd"], pool["gp_mean"] = scored["gp_sd"], scored["gp_mean"]
    pool["neural_mean"] = scored["neural_mean"]
    pool["outside_incumbent_box"] = (np.abs(pool[columns].to_numpy()-anchor).max(axis=1) > F5_LOCAL_RADIUS)
    eligible = pool.loc[pool.outside_incumbent_box].copy()
    ranked = eligible.sort_values(["gp_latent_sd", "nearest_observed_distance", *columns],
        ascending=[False, False, *[True]*len(columns)])
    point = ranked.iloc[0][columns].to_numpy(dtype=float)
    alternatives = {
        "original_gp_ucb": point_record(review, summary["point"]),
        "original_neural_mean": point_record(review, summary["neural_proposal"]["point"]),
        "global_information_query": point_record(review, point),
        "finite_pool_coverage": point_record(review, coverage),
    }
    sensitivity = []
    for seed, power in ((43, 14), (42, 12), (42, 15)):
        settings = {**summary["search_settings"], "broad_seed": seed, "broad_power": power}
        changed, _ = make_candidate_review(review["known"], anchor, input_columns=columns,
            search_settings=settings, minimum_distance=F5_MIN_DISTANCE)
        remote = changed.loc[np.abs(changed[columns].to_numpy()-anchor).max(axis=1) > F5_LOCAL_RADIUS].copy()
        _, sd = gp_prediction(fitted["gp"], remote[columns].to_numpy())
        remote["gp_latent_sd"] = sd
        top = remote.sort_values(["gp_latent_sd", "nearest_observed_distance", *columns],
            ascending=[False, False, *[True]*len(columns)]).iloc[0]
        sensitivity.append(dict(broad_seed=seed, broad_power=power, eligible_candidates=len(remote),
            portal_input="-".join(f"{top[c]:.6f}" for c in columns), gp_latent_sd=float(top.gp_latent_sd)))
    details = dict(decision="global_information_priority", local_exclusion_radius=F5_LOCAL_RADIUS,
                   radius_definition="maximum absolute coordinate change from incumbent; same box as existing local search",
                   minimum_known_input_distance=F5_MIN_DISTANCE, candidate_count=len(pool),
                   eligible_global_candidates=len(eligible), alternatives=alternatives, search_sensitivity=sensitivity,
                   prediction_checks=prediction_checks(review),
                   original_gp_and_neural_proposals_identical=summary["point"] == summary["neural_proposal"]["point"],
                   uncertainty_and_coverage_agree=bool(np.array_equal(point, coverage)),
                   selection_uses_ucb_weight=False,
                   untested_local_direction="There has been no isolated near-incumbent x3 reduction. The original GP/NN candidate would test that direction; local refinement is not exhausted.",
                   learning_tradeoff="The exploration point prioritises GP latent SD and geometric coverage. Both models forecast a low return. "
                                     "It may simply confirm low yield, and no greatest downstream optimisation benefit has been established.",
                   rationale="Three lower near-corner returns support pausing local refinement for one exploration query. "
                             "Test the largest finite-pool model/coverage gap; both fits predict a low score, so this is a learning query. "
                             "An isolated local x3 test remains available. The existing best remains 8662.405001248297; no local or global maximum is established.")
    return point, details, pool


def review_f6(review):
    """Adopt the saved neural mean candidate for one GP-cross-checked focused test."""
    summary = review["summary"]
    checks = prediction_checks(review)
    gp = point_record(review, summary["point"])
    neural = point_record(review, summary["neural_proposal"]["point"])
    greedy = next(row for row in summary["weight_sensitivity"] if row["Weight"] == 0)
    gp_greedy = point_record(review, [greedy[c] for c in review["fitted"]["input_columns"]])
    # The policy choice is explicit and uses the already-declared seed ensemble.
    guardrails = dict(neural_rmse_lower=checks["neural_rmse"] < checks["gp_rmse"],
                      neural_mae_lower=checks["neural_mae"] < checks["gp_mae"],
                      gp_mean_higher_at_neural_candidate=neural["gp_mean"] > gp["gp_mean"],
                      all_selected_neural_seeds_prefer_neural_candidate=all(
                          a>b for a,b in zip(neural["neural_seed_predictions"], gp["neural_seed_predictions"])))
    if not all(guardrails.values()):
        raise ValueError("The stated Function 6 candidate cross-checks no longer hold")
    details = dict(decision="neural_seed_mean_candidate_with_gp_crosscheck", prediction_checks=checks,
                   guardrails=guardrails, alternatives=dict(original_gp_ucb=gp, gp_mean_only=gp_greedy,
                                                           selected_neural_candidate=neural),
                   rationale="Use one focused, non-repeated query near the incumbent after four non-improving returns. The neural ensemble has lower overall RMSE/MAE and both model families prefer this candidate's mean to the GP-UCB candidate's mean. The roughly 4% RMSE gain, worse five-week-target RMSE and seed variability limit confidence; this is not a general replacement of GP acquisition.",
                   tradeoff="Less GP uncertainty/coverage gain than the original UCB point; prioritises a plausible near-incumbent improvement test.",
                   selection_uses_ucb_weight=False,
                   neural_seed_spread_is_posterior_uncertainty=False)
    return np.asarray(neural["point"]), details


def validate_lines(points, reviews, prior_files):
    """Reject all formatting, bounds or known-input problems before any write."""
    if len(points) != 8 or len(reviews) != 8:
        raise ValueError("Require exactly eight function-ordered points")
    lines, validation = [], []
    for function, (point, dimensions, review) in enumerate(zip(points, DIMENSIONS, reviews), 1):
        point = np.asarray(point, float)
        if (point.shape != (dimensions,) or not np.isfinite(point).all()
                or not ((point >= 0) & (point < 1)).all() or not np.array_equal(point, point.round(6))):
            raise ValueError(f"Invalid Function {function} coordinates")
        line = "-".join(f"{x:.6f}" for x in point)
        if tuple(point) in set(map(tuple, review["known"].round(6))):
            raise ValueError(f"Function {function} repeats a known official/peer input")
        for prior in prior_files:
            earlier = prior.read_text(encoding="utf-8").splitlines()
            if len(earlier) != 8:
                raise ValueError("An earlier official submission is malformed")
            if line == earlier[function-1]:
                raise ValueError(f"Function {function} repeats an earlier official submission")
        lines.append(line)
        validation.append(dict(function=function, known_rows_checked=len(review["known"]),
                               prior_submission_files_checked=len(prior_files), duplicate=False,
                               nearest_known_input_distance=float(cdist(point[None], review["known"]).min())))
    return "\n".join(lines)+"\n", validation


def apply_submission_confirmation(record, confirmation):
    """Retain owner-confirmed submission when regenerating the selection record."""
    if (confirmation["week"] != record["week"]
            or confirmation["status"] != "submitted_awaiting_results"
            or confirmation["source"] != "project_owner_confirmation"
            or confirmation["results_status"] != "pending"
            or confirmation["submission_file"] != record["submission_file"]
            or confirmation["latest_confirmed_feedback_round"] != record["feedback_through_round"]
            or confirmation["official_observations"] != record["official_observations"]
            or confirmation["remaining_queries_per_function_after_submission"]
                != record["future_opportunities_after_week_06_if_submitted"]):
        raise ValueError("Submission confirmation does not match the retained inputs, feedback or query budget")
    record.update(status=confirmation["status"], submission_confirmation=confirmation,
                  remaining_queries_per_function_after_submission=confirmation["remaining_queries_per_function_after_submission"],
                  portal_submission_performed=True, portal_submission_performed_by_this_workflow=False)
    return record


REASONS = {
    2: "Retain the pooled separate-length-scale GP candidate. Official-target RMSE improves from 0.12335237524474416 to 0.11137906268912874; the neural comparator is weaker. Weight and regularisation sensitivity limit confidence.",
    3: "Retain the pooled shared-length-scale GP candidate under the declared RMSE rule. Its official-target RMSE is 0.04869707389884704 versus 0.07390043231177344 for the official selected GP. The last low-x3 failure and continued extrapolation remain risks; one result does not isolate an x3 effect.",
    4: "Retain the official GP local candidate near the new confirmed positive incumbent. GP validation is materially better than the neural fit, and peer training does not improve the declared error. The fitted mean remains below the measured best.",
    7: "Retain the GP-UCB local candidate. GP RMSE is lower than the neural ensemble's; the more optimistic neural mean alternative lacks enough validation support to replace the established policy.",
    8: "Retain the full-input GP-UCB candidate, with a predicted mean above the incumbent. GP validation is better than the neural comparator; full eight-input extrapolation and weight sensitivity remain material limitations.",
}


def build_review():
    reviews = [read_review(f) for f in range(1, 9)]
    points = [np.asarray(r["summary"]["point"]) for r in reviews]
    points[0], f1 = review_f1(reviews[0])
    if (reviews[1]["summary"]["portal_input"] != "0.700520-0.000114"
            or reviews[1]["summary"]["ucb_weight"] != 1.96):
        raise ValueError("The agreed Function 2 input or uncertainty weight changed")
    f2 = dict(decision="retained_gp_ucb", status="agreed", effective_ucb_weight=1.96,
              portal_input="0.700520-0.000114", rationale=REASONS[2],
              weight_interpretation="An explicit exploration/exploitation choice, not calibrated coverage or a guarantee of improvement.")
    if (reviews[2]["summary"]["portal_input"] != "0.721213-0.830963-0.001845"
            or reviews[2]["summary"]["ucb_weight"] != 1.96):
        raise ValueError("The agreed Function 3 input or uncertainty weight changed")
    f3 = dict(decision="retained_gp_ucb", status="agreed", effective_ucb_weight=1.96,
              portal_input="0.721213-0.830963-0.001845", rationale=REASONS[3],
              relies_on_retained_peer_data=True,
              peer_data_caveat="The selected peer-assisted GP has lower declared official-target RMSE, but the official-only GP "
                               "proposes a completely different region. There is no cross-scope agreement on the low-x3 region. "
                               "The recommendation depends on retained self-reported peer evidence; the recent low-x3 failure and extrapolation limit confidence.")
    if (reviews[3]["summary"]["portal_input"] != "0.408963-0.412546-0.416464-0.438393"
            or reviews[3]["summary"]["ucb_weight"] != 1.96):
        raise ValueError("The agreed Function 4 input or uncertainty weight changed")
    f4 = dict(decision="retained_gp_ucb", status="agreed", effective_ucb_weight=1.96,
              portal_input="0.408963-0.412546-0.416464-0.438393", rationale=REASONS[4],
              gp_forecast_caveat="The GP predicts about -0.371831 at the already measured +0.650130 incumbent. "
                                 "At the agreed point its mean is about -0.408796 and UCB about +0.418308, both below the measured best. "
                                 "The choice is a small local test guided by finite-pool ranking, not a demonstrated predicted improvement or calibrated local forecast.",
              gp_at_measured_incumbent=point_record(reviews[3], reviews[3]["fitted"]["incumbent"]))
    points[4], f5, f5_pool = review_f5(reviews[4])
    if not np.array_equal(points[4], [0., 0., .999999, .999999]):
        raise ValueError("The agreed Function 5 exploration input changed")
    f5.update(status="agreed", portal_input="0.000000-0.000000-0.999999-0.999999",
              one_off_exploration_query=True, exploration_queries_committed=1)
    points[5], f6 = review_f6(reviews[5])
    if not np.array_equal(points[5], [.398303, .269220, .627336, .768441, 0.]):
        raise ValueError("The agreed Function 6 neural candidate changed")
    f6.update(status="agreed", portal_input="0.398303-0.269220-0.627336-0.768441-0.000000",
              one_focused_neural_led_test=True, focused_tests_committed=1,
              agreed_caveats="The overall neural RMSE gain is only about 4%, while error on the five weekly targets is worse. "
                             "The proposed mean gain over the incumbent is only about 0.0152, far below held-out error. "
                             "The GP mean at this point remains below the incumbent and the GP prefers its own mean-only candidate. "
                             "The full-input point extrapolates on x5=0. Neural seed variation is not posterior uncertainty; the 1.96 multiplier does not select this point.")
    if (reviews[6]["summary"]["portal_input"] != "0.000000-0.135108-0.370684-0.306265-0.311144-0.649736"
            or reviews[6]["summary"]["ucb_weight"] != 1.96):
        raise ValueError("The agreed Function 7 input or uncertainty weight changed")
    f7 = dict(decision="retained_gp_ucb", status="agreed", effective_ucb_weight=1.96,
              portal_input="0.000000-0.135108-0.370684-0.306265-0.311144-0.649736", rationale=REASONS[7],
              agreed_caveats="The GP mean at the point is below the measured incumbent; UCB is an optimistic ranking score, not a returned-value forecast. "
                             "The full six-input point is outside the sampled hull on x1=0. Posterior SD is conditional on the fitted kernel and regularisation. "
                             "The exact point changes with the uncertainty weight, and no optimal multiplier or guaranteed improvement is established.")
    if (reviews[7]["summary"]["portal_input"] != "0.022327-0.096114-0.237034-0.202848-0.982361-0.880106-0.159960-0.678158"
            or reviews[7]["summary"]["ucb_weight"] != 1.96):
        raise ValueError("The agreed Function 8 input or uncertainty weight changed")
    f8 = dict(decision="retained_gp_ucb", status="agreed", agreed_on="2026-10-31", effective_ucb_weight=1.96,
              portal_input="0.022327-0.096114-0.237034-0.202848-0.982361-0.880106-0.159960-0.678158", rationale=REASONS[8],
              agreed_caveats="Only 45 observations support this eight-dimensional fit. The full-input point lies outside the sampled hull "
                             "and is about 0.754759 from its nearest observation. The GP mean gain of about 0.213760 is below held-out RMSE, "
                             "and the neural mean margin is tiny. The exact point is weight-sensitive and another candidate has a near-identical UCB. "
                             "Posterior SD is conditional on the fitted model; UCB is an optimistic ranking score, not a forecast or guarantee of improvement. "
                             "Neither a uniquely best input nor an optimal 1.96 multiplier is established.")
    prior_files = sorted((STAGE / "submissions").glob("Week_0[1-5]/submissions.txt"))
    if len(prior_files) != 5:
        raise ValueError("Expected all five earlier weekly submission files")
    text, validation = validate_lines(points, reviews, prior_files)
    candidates = []
    for f, (review, point, checked) in enumerate(zip(reviews, points, validation), 1):
        summary = review["summary"]
        scored = point_record(review, point)
        if f == 1:
            basis, reason = f1["decision"], f1["rationale"]
        elif f == 5:
            basis, reason = "global_information_priority", f5["rationale"]
        elif f == 6:
            basis, reason = "neural_seed_mean_candidate_with_gp_crosscheck", f6["rationale"]
        else:
            basis, reason = "retained_gp_ucb", REASONS[f]
        candidates.append(dict(function=f, dimensions=DIMENSIONS[f-1], portal_input=scored["portal_input"],
            point=point.tolist(), selection_policy=basis, data_scope=summary["training_scope"], reason=reason,
            decision_status="agreed",
            predictions=scored, prediction_units=summary["target_units"],
            original_model_notebook_proposal=summary["portal_input"],
            changed_from_notebook_proposal=summary["point"] != point.tolist(),
            evidence=dict(official_observations=summary["observations"],
                best_official_raw_output=summary["best_observed_raw_output"],
                gp_selection_metric=summary["selection_metric"], gp_official_target_error=summary["official_target_error"],
                neural_ensemble_errors=summary["neural_ensemble_errors"]), sources=review["sources"], validation=checked))
    record = dict(week=6, prepared_on="2026-10-30", status="prepared_not_submitted", feedback_through_round=5,
                  agreed_functions=[1, 2, 3, 4, 5, 6, 7, 8], selections_finalised_on="2026-10-31",
                  official_observations=215, query_opportunities_per_function_including_week_06=8,
                  future_opportunities_after_week_06_if_submitted=7,
                  submission_file=dict(path="stage-2-bbo/submissions/Week_06/submissions.txt",
                                       sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()),
                  validation=dict(line_count=8, dimensions=list(DIMENSIONS), coordinate_range="[0, 1)",
                      decimal_places=6, separator="-", spaces=False, duplicates=False,
                      prior_submission_files_checked=5), candidates=candidates,
                  function_1_review=f1, function_2_review=f2, function_3_review=f3, function_4_review=f4,
                  function_5_review=f5, function_6_review=f6, function_7_review=f7, function_8_review=f8,
                  selection_source=dict(path=Path(__file__).relative_to(ROOT).as_posix(), sha256=fingerprint(__file__)),
                  earlier_submissions=[dict(path=p.relative_to(ROOT).as_posix(), sha256=fingerprint(p)) for p in prior_files],
                  new_course_evaluations=0, portal_submission_performed=False,
                  portal_submission_performed_by_this_workflow=False)
    confirmation_path = STAGE / "submissions/Week_06/submission-confirmation.json"
    if confirmation_path.exists():
        apply_submission_confirmation(record, json.loads(confirmation_path.read_text(encoding="utf-8")))
    return record, text, f5_pool


def prepare_submission():
    record, text, _ = build_review()
    folder = STAGE / "submissions/Week_06"
    target = folder / "submissions.txt"
    if target.exists() and target.read_bytes() != text.encode("utf-8"):
        raise FileExistsError("An existing Week 6 submission file differs; review it before replacement")
    folder.mkdir(exist_ok=True)
    target.write_bytes(text.encode("utf-8"))
    (folder / "proposals.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if target.read_bytes() != text.encode("utf-8"):
        raise ValueError("Submission readback mismatch")
    return record


if __name__ == "__main__":
    prepared = prepare_submission()
    print(f"Verified eight Week 6 inputs. Status: {prepared['status']}.")

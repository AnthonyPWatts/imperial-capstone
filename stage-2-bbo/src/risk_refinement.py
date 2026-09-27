"""Verify the conservative choice on a larger pool and retain a complete record."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from function_2_risk import score_queries, simulate_remaining_queries
from multifunction_risk import (ensure_risk_results, restore_study, candidate_pool,
                                _scored_models, _save, fit_model, posterior_on_pool)


def ensure_refined_results(data, function_id, output, *, progress=None):
    saved = ensure_risk_results(data, function_id, output, progress=progress)
    fingerprint = hashlib.sha256((saved["fingerprint"] + Path(__file__).read_text()).encode()).hexdigest()
    if saved.get("refinement_fingerprint") == fingerprint:
        return saved
    report = progress if progress is not None else lambda message: None
    report(f"Function {function_id}: full decision-risk check on the larger pool")
    started = perf_counter()
    study = restore_study(data, output)
    inputs = data[[name for name in data if name.startswith("x")]].to_numpy()
    previous = study["pool"][study["robust_index"]]
    pool = np.unique(np.vstack([candidate_pool(inputs, data.y.to_numpy(), fine=True),
                                study["pool"][study["candidate_indices"]]]), axis=0)
    with threadpool_limits(limits=2):
        models = _scored_models(study["fits"], pool, risk_draws=4096)
    residual = np.stack([m["current"]["risk"] - m["gains"] for m in models])
    after = residual.max(axis=0)
    before = max(m["current"]["risk"] for m in models)
    best = int(np.argmin(after))
    old = int(np.flatnonzero(np.all(pool == previous, axis=1))[0])
    retained = (before - after[old]) / (before - after[best])
    # Retain a choice with at least 95% of the larger-pool gain; this tolerance
    # limits switches caused by very small numerical/modelled differences.
    chosen = pool[best] if retained < .95 else previous
    reason = "Larger-pool conservative KG" if retained < .95 else "Conservative KG, confirmed on the larger pool"
    if function_id == 1:
        chosen = np.array([.468429, .435440])
        reason = "Original mean-coverage proposal retained after an inconclusive raw-output KG check"
    match = np.flatnonzero(np.all(study["pool"] == chosen, axis=1))
    if not len(match):
        # Never mix old-pool risk/coverage statistics with a new rollout pool.
        raise ValueError("The refined query is absent from the comparison pool. "
                         "Include it in the base study before comparing rollouts.")
    chosen_index = int(match[0])
    if chosen_index not in study["candidate_indices"]:
        study["candidate_names"].append("Larger-pool choice")
        study["candidate_indices"].append(chosen_index)
        from multifunction_risk import coverage_scores
        coverage, _ = coverage_scores(inputs, chosen[None, :], power=16, seed=101)
        study["checked_coverage"] = np.r_[study["checked_coverage"], coverage]
        with threadpool_limits(limits=2):
            for model in study["models"]:
                addition = simulate_remaining_queries(model["posterior"], [chosen_index], len(study["pool"]), trials=64)
                for key in ("losses", "mean", "mc_se"):
                    model["rollout"][key] = np.concatenate([model["rollout"][key], addition[key]], axis=0)
    # Keep sensitivity ratios aligned with the actual recommendation.
    if chosen_index != study["robust_index"]:
        with threadpool_limits(limits=2):
            for row in study["noise_checks"]:
                fit = fit_model(inputs, data.y.to_numpy(), row["model"], row["noise_fraction"])
                posterior = posterior_on_pool(fit, study["pool"])
                gains = score_queries(posterior, len(study["pool"]))
                row["recommended_gain"] = float(gains[chosen_index])
    study["recommended_index"] = chosen_index
    study["recommendation_reason"] = reason
    study["refinement"] = {
        "pool_count": len(pool), "previous": previous, "fine_choice": pool[best],
        "worst_current_risk": before, "fine_after": float(after[best]),
        "previous_after": float(after[old]), "retained_fraction": float(retained),
        "fine_model_gains": {m["name"]: float(m["gains"][best]) for m in models},
        "fine_current_risks": {m["name"]: m["current"] for m in models},
        "seconds": perf_counter() - started,
    }
    study["refinement_fingerprint"] = fingerprint
    _save(study, output)
    report(f"Function {function_id}: recommendation ready")
    return json.loads((Path(output) / "risk-study.json").read_text(encoding="utf-8"))

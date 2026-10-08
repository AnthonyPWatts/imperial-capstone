"""Collect the verified first-query studies into a portable local reading pack."""

from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

stage = Path(__file__).resolve().parents[1]
root = stage.parent
sys.path.insert(0, str(stage / "src"))
from initial_exploration import load_initial_data
from coverage_optimisation import MeanCoverage2D
from report_provenance import verify_report_provenance


REASONS = {
    1: "Retain average coverage. Raw-output KG is dominated by one negative extreme and lacks held-out support.",
    2: "Upper-right query from the finer local-model KG search; model uncertainty is more important than geometric coverage alone.",
    3: "Test whether the relatively good region around ranks 3 and 4 extends to a nearby combination; model support remains weak.",
    4: "Test a combination around the better observed settings. Held-out prediction strongly supports using fitted structure.",
    5: "Probe high x2, x3 and x4 near the large positive response. The larger-pool check changes the original proposal.",
    6: "Retain relatively high x4 and low x5 while testing a different combination of the other three inputs.",
    7: "Increase x4 by 0.1 at the best observation, keeping the other coordinates. A controlled local hypothesis test with weak model support.",
    8: "Test an extreme combination with low x1 and x3. Strong additive predictive structure supports modelling, while the conservative local-model choice is stable to noise.",
}


def main():
    # Check every input before creating or overwriting any pack/proposal files.
    reports = {}
    for function in range(1, 9):
        for suffix in ("initial-analysis", "risk"):
            key = f"function-{function}-{suffix}"
            reports[key] = verify_report_provenance(stage, function, suffix)
    destination = root / ".runtime/first-query-review"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for function in range(1, 9):
        data = load_initial_data(stage, function)
        inputs = data[[name for name in data if name.startswith("x")]].to_numpy()
        study = json.loads((root / f".runtime/function-{function}-risk/risk-study.json").read_text(encoding="utf-8"))
        if function == 2:
            selected = next(item for item in study["grid_checks"] if item["model"] == "Local 2D")
            point = np.array(selected["point"])
            loo = json.loads((root / ".runtime/function-2-initial-analysis/selection-study.json").read_text())["loo_rmse"]
            error_reduction = 1 - min(loo[:2]) / loo[2]
            geometry = MeanCoverage2D(inputs)
            coverage = 1 - geometry(point) / geometry.mean_before
            evidence = {"criterion": "Local 2D knowledge gradient on the 41 x 41 pool",
                        "one_query_gain": selected["gain"], "noise_sd": .05,
                        "note": "The existing thirteen-query comparison used the nearby coarse-grid choice."}
        else:
            index = study["recommended_index"]
            point = np.array(study["pool"][index])
            slot = study["candidate_indices"].index(index)
            coverage = study["checked_coverage"][slot]
            error_reduction = 1 - min(item["rmse"] for item in study["checks"][1:]) / study["checks"][0]["rmse"]
            fine = study["refinement"]
            evidence = {"criterion": study["recommendation_reason"], "noise_sd": study["noise_sd"],
                        "comparison_pool_size": len(study["pool"]), "larger_pool_size": fine["pool_count"],
                        "main_pool_model_gains": {model["name"]: model["gains"][index] for model in study["models"]},
                        "larger_pool_kg_candidate": fine["fine_choice"],
                        "larger_pool_worst_risk_before": fine["worst_current_risk"],
                        "larger_pool_worst_risk_after_kg": fine["fine_after"],
                        "note": "Function 1 retains coverage rather than the reported KG candidate." if function == 1 else
                                "Conditional finite-pool expected shortfall; not a probability or continuous-domain bound."}
        portal = "-".join(f"{value:.6f}" for value in point)
        # Submission-facing checks belong here, away from the reading notebooks.
        assert len(point) == inputs.shape[1]
        assert ((point >= 0) & (point < 1)).all()
        assert not np.any(np.all(np.round(inputs, 6) == point, axis=1))
        assert np.linalg.norm(inputs - point, axis=1).min() > 1e-5
        rows.append({"function": function, "dimensions": len(point), "initial_observations": len(data),
                     "point": point.tolist(), "portal_input": portal, "status": "proposed_not_submitted",
                     "rationale": REASONS[function], "coverage_gain": float(coverage),
                     "best_model_rmse_reduction_vs_mean": float(error_reduction), "evidence": evidence})
        for suffix in ("initial-analysis", "risk"):
            target = destination / f"function-{function}-{suffix}"
            target.mkdir(exist_ok=True)
            shutil.copy2(root / f".runtime/function-{function}-{suffix}/analysis.html", target / "analysis.html")
    sources = sorted([*stage.glob("src/*.py"), *stage.glob("scripts/*.py"),
                      *stage.glob("notebooks/Function_*/Week_01/*.ipynb")])
    record = {"generated_utc": datetime.now(timezone.utc).isoformat(), "round": 1,
              "status": "proposals_only_no_portal_submissions",
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_state": "Working-tree source hashes verified against executed report manifests.",
              "report_provenance": reports,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
              "source_sha256": {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sources}, "proposals": rows}
    proposal_path = stage / "submissions/round-01-proposals.json"
    proposal_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    shutil.copy2(proposal_path, destination / "round-01-proposals.json")

    figure, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), layout="constrained")
    functions = [row["function"] for row in rows]
    error_changes = np.array([row["best_model_rmse_reduction_vs_mean"] for row in rows]) * 100
    axes[0].bar(functions, error_changes, color=["#287a69" if value > 0 else "#bd531d" for value in error_changes])
    axes[0].axhline(0, color="#35434a", linewidth=.8)
    axes[0].set(title="How much predictive structure did we find?", ylabel="Held-out RMSE reduction versus mean (%)")
    axes[1].bar(functions, [100 * row["coverage_gain"] for row in rows], color="#26769a")
    axes[1].set(title="How much coverage would the proposed point add?", ylabel="Mean-distance reduction (%)")
    for axis in axes:
        axis.set(xlabel="Function", xticks=functions)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.savefig(destination / "comparison.png", dpi=160)
    plt.close(figure)
    import base64
    chart = base64.b64encode((destination / "comparison.png").read_bytes()).decode()
    cards = []
    markdown = ["# First-query proposals", "", "Eight proposed first inputs, before any portal evaluation.", "",
                "**Start by reducing ignorance, finish by exploiting what we have learned.**", "",
                "| Function | Portal input |", "| --- | --- |"]
    for row in rows:
        function = row["function"]
        markdown.append(f"| {function} | `{row['portal_input']}` |")
        cards.append(f'''<section id="function-{function}"><h2>Function {function} <small>{row['dimensions']} inputs · {row['initial_observations']} observations</small></h2>
<p class="point"><code>{escape(row['portal_input'])}</code></p><p>{escape(row['rationale'])}</p>
<p class="links"><a href="function-{function}-initial-analysis/analysis.html">Data exploration</a> · <a href="function-{function}-risk/analysis.html">Knowledge gradient and justification</a></p></section>''')
    markdown += ["", "## Interpretation", "",
                 "Function 1 retains its coverage proposal after an inconclusive raw-output KG check. Function 2 retains the finer-grid point from its existing study. Functions 3–8 use the conservative three-model KG comparison with noise, candidate-pool and thirteen-query checks. The larger-pool check revises Function 5's proposed point.", "",
                 "Prediction is strongest in Functions 4, 6 and 8, and weakest in Functions 1, 3 and 7. These differences reflect function behaviour and sample size as well as dimension. Numerical KG values are conditional expected reductions in decision shortfall, in original output units, not probabilities or guarantees. Six decimals express portal precision.", ""]
    for row in rows:
        markdown += [f"## Function {row['function']}", "", row["rationale"], "",
                     f"Estimated full-space coverage gain: {100 * row['coverage_gain']:.4f}%. Best fitted-model held-out RMSE reduction versus the training-mean baseline: {100 * row['best_model_rmse_reduction_vs_mean']:.1f}%.", ""]
    markdown += ["## Reproduction and reading copies", "",
                 "The individual notebooks retain the computational rationale. Run `stage-2-bbo/scripts/run_initial_analysis.py --function N` and the same command with `--decision-risk` using the repository's Python environment. Then run `stage-2-bbo/scripts/build_first_query_review.py` to rebuild the local pack.", "",
                 "The standalone index and portable ZIP are generated under `.runtime/first-query-review/` and `.runtime/capstone-first-query-pack.zip`. The proposal JSON records the base commit, verified report manifests, source hashes, settings and runtime versions. No submission or returned observation is recorded.", ""]
    (stage / "results/first-query-proposals.md").write_text("\n".join(markdown), encoding="utf-8")
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Capstone — first-query review</title><style>
body{{max-width:1100px;margin:auto;padding:32px;color:#24343f;background:#fff;font:17px/1.55 system-ui,sans-serif}}
h1{{font-size:2.1em;line-height:1.2}} h2{{margin-bottom:.45em}} small{{font-size:.6em;font-weight:normal;color:#536571;margin-left:12px}}
section{{border-top:1px solid #d9e1e5;padding:12px 0 24px}} .point{{background:#edf3f6;padding:14px;border-radius:5px;overflow-x:auto}}
code{{font-size:15px;white-space:nowrap}} img{{max-width:100%;height:auto}} a{{color:#15648c}} nav a{{margin-right:20px}}
.note{{background:#f6f4ec;padding:16px}} @media(max-width:600px){{body{{padding:16px}}small{{display:block;margin-left:0}}}}
</style></head><body><h1>Capstone — first-query review</h1>
<p><strong>Eight suggested inputs, with data exploration and computational justification.</strong> Prepared from the supplied starter data on 24 September 2026. These are proposals; none has been submitted.</p>
<p><strong>Taper-waper: start by reducing ignorance, finish by exploiting what we have learned.</strong></p>
<nav>{' '.join(f'<a href="#function-{function}">{function}</a>' for function in functions)}</nav>
<h2>What changed as the functions grew?</h2>
<p>Dimension alone did not determine difficulty. The eight-input function showed much clearer predictive structure than the two-input Function 1 or three-input Function 3. The starting sample sizes and function behaviours also differ, so this is not an isolated experiment on dimensionality.</p>
<img src="data:image/png;base64,{chart}" alt="Predictive evidence and coverage benefit by function">
<p>The left chart compares each function's best checked predictor with its held-out training-mean baseline. Negative values mean that even the best fitted model was worse. These small-sample results are descriptive; the model sets also differ for Function 2. The right chart shows geometric coverage benefit. Knowledge-gradient choices can add little uniform-volume coverage while investigating a consequential hypothesis.</p>
<p class="note"><strong>Read the strength of the evidence alongside each number.</strong> Functions 4, 6 and 8 have stronger held-out predictive support. Functions 1 and 3 fail that check; Function 7 has only modest support and its most cautious model is a weak predictor. Function 1 therefore retains its original coverage proposal. Function 5 changes after the larger-pool check.</p>
<p>KG values are conditional expected improvements in a final decision, in each function's own output units. They are not probabilities or guarantees. The thirteen-query experiments test a declared taper policy; they do not solve an optimal thirteen-step plan. Six decimal places are the portal's format.</p>
{''.join(cards)}
<p><a href="round-01-proposals.json">Structured proposals, settings and source hashes</a></p>
<p>All reading copies embed their figures and need no Python installation. The ZIP contains this index, the sixteen reports and the proposal record; extract it before opening the index.</p>
</body></html>'''
    (destination / "index.html").write_text(html, encoding="utf-8")
    archive = root / ".runtime/capstone-first-query-pack.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.suffix in (".html", ".json"):
                bundle.write(path, path.relative_to(destination))
    print(f"Review: {destination / 'index.html'}")
    print(f"Portable pack: {archive} ({archive.stat().st_size:,} bytes)")
    print("Eight proposals checked for dimensions, six-decimal formatting, bounds and duplicate inputs.")


if __name__ == "__main__":
    try:
        main()
    except ValueError as error:
        raise SystemExit(str(error)) from error

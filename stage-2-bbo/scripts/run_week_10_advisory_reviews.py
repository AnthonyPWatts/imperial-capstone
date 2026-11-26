
"""Build and execute the eight Week 10 advisory notebooks and proposal summary."""

import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from textwrap import dedent

import nbformat
from nbclient import NotebookClient


STAGE = Path(__file__).resolve().parents[1]
OUTPUT = STAGE / "submissions/Week_10"
SHARED = STAGE / "notebooks/Shared"
F1_SHARED = STAGE / "notebooks/Function_1/FunctionSpecificShared"
sys.path[:0] = [str(SHARED), str(F1_SHARED)]

from cohort_advisory_bo import SETTINGS  # noqa: E402
from week_10_decision_record import decision_markdown  # noqa: E402


NOTES = {
    1: "A narrow positive source leads the official search. The archive locates the high region precisely. "
       "Inspect its nearest shortlist point and the small differences in predicted gain before choosing a refinement.",
    2: "Recent official scores favour the x1 ≈ 0.70 band. Eight archived returns at (0.50, 0.50) "
       "make a separate region worth reviewing. The gap table shows whether the official shortlist can test it.",
    3: "The official best is close to zero, and smooth extrapolation can assign positive scores in weakly sampled regions. "
       "Compare any broader winner with the local candidate on the official high-score slice before spending a query.",
    4: "The official positives define a local hill. Compare the shortlisted move against the earlier x3-only midpoint "
       "to distinguish predicted upside from the value of a controlled measurement.",
    5: "The permitted upper corner is already measured. The 0.02 separation rule makes the next query primarily a "
       "test of the response away from that corner. Look at the raw expected gain before interpreting the normalised rank.",
    6: "The official x5 = 0 face remains competitive. The 2024 data identify a distinct interior region around x5 ≈ 0.11. "
       "A shortlist winner on the face leaves that regional question open.",
    7: "The official gains clustered near x1 = 0. The strongest archived region also shifts x3. A boundary winner "
       "tests refinement of the established region and leaves the interior regional comparison open.",
    8: "Use the starter-only separable quadratic while it reproduces the nine later returns to numerical precision. "
       "The Gaussian process and cohort advisers give sensitivity views around the fitted peak.",
}

MAIN = """
## Expectations

Use the official accumulated observations to fit a Gaussian process in raw score units. For Function 1, a stable official positive-source fit supplies the mean function and the GP adjusts the residuals. Expected improvement (EI) is the expected positive difference between one future return and the official incumbent. The acquisition uses the GP's predictive observation uncertainty. The fitted GP supplies the same uncertainty to both advisers.

The official model screens a seeded Sobol pool across the domain and around high official points. The frozen shortlist includes the leading EI candidates, candidates from different search regions and coordinate directions, and the previous owner-first proposal when eligible. Official inputs and prior submissions are excluded at portal precision. Function 5 retains its 0.02 spacing rule.

Current-cohort returns and the 2024 archive fit **separate residual GPs**. Each learns the difference between its reported score and the official GP prediction. Repeated inputs contribute their median and their measured spread. Official coordinates are excluded from each adviser, including during replay. When a coordinate occurs in both external groups, the current cohort supplies the evidence at that coordinate. The 2024 source includes its extra starters and archived returns.

An adviser's residual adjusts the owner mean for its own EI calculation. Divide the resulting change in EI by the best owner EI and cap it at ±1. The ranking is **0.80 × normalised owner EI + 0.10 × current change + 0.10 × archival change**. A point with no nearby external support gets little adjustment. Weights are fixed judgement calls. The weight sweep shows their effect, including the official-only decision. This is a bounded shortlist adviser inspired by transfer Bayesian optimisation. [Transfer Bayesian optimisation](https://arxiv.org/abs/1802.02219)

## Results

The input fingerprints identify the local files reproduced below. The extended sources provide comparison and advice for this experiment. Course permission to rely on those scores for an assessed submission remains unresolved. [Course data-use review](../../../docs/capstone-data-policy.md) · [2024 intake and transfer review](../../../side-quests/historical-cohort-2024/README.md) · [Official weekly notebook](weekly_review.ipynb)
"""

REPLAY = """
### Replay on the nine official returns

Remove each returned coordinate from official training in turn and refit. The external snapshots are frozen at their current versions and exclude every official coordinate. The replay measures **score prediction** retrospectively. It uses neither the historical information timeline nor a live optimisation experiment, so a lower replay error alone gives limited evidence about the next query. The 10% columns apply each advisory mean shift at the declared weight.
"""

PLOTS = """
### Pairwise fitted surfaces

For each pair, hold the other coordinates at the advisory experiment's winner. The star marks that acquisition winner. The circle projects the incumbent and the plus projects the best archived point. These reference markers may come from a different slice. The top-right panel shows the owner GP's latent uncertainty. For F8 this is the uncertainty of the GP comparator, while the mean panel shows the verified quadratic. Lower panels show the two signed adjustments to the GP mean in raw output units using the same colour scale within each figure. The capped acquisition contributions appear in the shortlist table.
"""

LIFECYCLE = """
### Course workflow

1. **Define the goal and scope** – improve the best future official score with four queries remaining, including Week 10.
2. **Gather the data** – verify the official, current-cohort and 2024 snapshot fingerprints.
3. **Explore the data** – compare the best external measurements and their distance to the official shortlist.
4. **Clean and preprocess the data** – enforce portal bounds, remove official overlaps from the advisers, and group repeated external coordinates.
5. **Select and engineer features** – retain every coordinate. Use the F1 source structure and the F8 separable quadratic where the official evidence supports them.
6. **Define the machine-learning task** – fit score regression surrogates for a sequential optimisation decision.
7. **Partition the data** – group by official portal coordinate for replay and keep the external groups separate.
8. **Select and train candidate methods** – fit the owner GP and two separately regularised advisory GPs.
9. **Evaluate and interpret the results** – use the replay, weight sweep, comparison distances and conditional pairwise surfaces.
10. **Deploy and iterate** – review the experimental input, then add the next official return and build a new owner shortlist.
"""


def markdown(value, cell_id):
    return nbformat.v4.new_markdown_cell(dedent(value).strip(), id=cell_id)


def code(value, cell_id):
    return nbformat.v4.new_code_cell(dedent(value).strip(), id=cell_id)


def notebook_for(function):
    setup = f"function = {function}\n" + dedent("""
        from pathlib import Path
        import json
        import sys

        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import display

        here = Path.cwd().resolve()
        stage = next(path for path in (here, *here.parents) if (path / "notebooks/Shared").is_dir())
        sys.path[:0] = [str(stage / "notebooks/Shared"),
                        str(stage / "notebooks/Function_1/FunctionSpecificShared")]

        from cohort_advisory_bo import SETTINGS, WEIGHTS, load_data, review
        from cohort_advisory_plots import pairwise_slice
        from weekly_peak_review import portal_point

        owner, current, archive, columns, fingerprints = load_data(stage, function)
        result = review(stage, function, owner, current, archive, columns)
        print(f"{len(owner)} official rows, {result['pool_size']:,} eligible candidates, "
              f"{len(result['table'])} on the frozen shortlist.")
        display(pd.Series(fingerprints, name="Intake evidence").to_frame())
        display(result["comparison"].round(6))
    """)
    validation = """
        replay = result["replay"]
        methods = ["Official only", "Current only", "2024 only", "10% each"]
        error = replay[methods].sub(replay["Actual"], axis=0)
        metrics = pd.DataFrame({"MAE": error.abs().mean(), "RMSE": error.pow(2).mean().pow(.5)})
        display(metrics.round(8))
        display(replay.round(8))
        if result["structural"] is not None:
            fit = result["structural"]
            print(f"Official quadratic maximum: {fit['predicted_maximum']:.10g}, "
                  f"maximum absolute replay error: {fit['later_max_abs_error']:.3g}.")
    """
    predictions = """
        weight_rows = pd.DataFrame(WEIGHTS, index=["Official", "Current", "2024"]).T
        display(weight_rows.assign(Input=pd.Series(result["winners"])))
        table = result["table"]
        front = [result["chosen"], result["winners"]["Official only"], result["previous"], result["local_winner"]]
        candidates = table[table["Input"].isin(front)].copy()
        candidates.insert(0, "Ensemble rank", candidates.index + 1)
        display(candidates.round(7))
        display(table.round(7))
    """
    plot = """
        for pair in SETTINGS[function][4]:
            figure = pairwise_slice(function, result, pair)
            plt.show()
            plt.close(figure)
    """
    # The build reads this tagged display as its verified machine-readable output.
    emit = """
        summary = {
            "function": function,
            "incumbent": portal_point(result["incumbent"]),
            "incumbent_y": result["incumbent_y"],
            "selected": result["chosen"],
            "official_only": result["winners"]["Official only"],
            "older_proposal": result["previous"],
            "local": result["local_winner"],
            "weight_sweep": result["winners"],
            "selected_in_radius": bool(table.loc[table["Input"].eq(result["chosen"]), "Within owner radius"].iloc[0]),
            "mae_official": float(metrics.loc["Official only", "MAE"]),
            "mae_advisory": float(metrics.loc["10% each", "MAE"]),
            "closest_2024_best": float(result["comparison"].loc[1, "Distance to nearest shortlist point"]),
        }
        print(json.dumps(summary, allow_nan=False))
    """
    opening = f"# Function {function} · Week 10 cohort advice\n\n" + NOTES[function]
    return nbformat.v4.new_notebook(cells=[
        markdown(opening + "\n" + MAIN, "expectations"),
        code(setup, "load"),
        markdown(REPLAY, "replay-intro"),
        code(validation, "replay"),
        markdown("## Predictions\n\nCompare the official choice, the prior proposal and the bounded ensemble. "
                 "A local alternative is shown when the wider shortlist wins. For F8 the verified official "
                 "quadratic takes precedence over the ranking.", "predictions-intro"),
        code(predictions, "ranking"),
        markdown(decision_markdown(function), "prepared-decision"),
        code("""
            from week_10_decision_record import decision_comparison

            with pd.option_context("display.max_colwidth", None):
                display(decision_comparison(stage, function, owner, result["model"], result["structural"]))
        """, "prepared-inputs"),
        markdown(PLOTS, "plots-intro"),
        code(plot, "surfaces"),
        code(emit, "build-summary"),
        markdown(LIFECYCLE, "workflow"),
    ], metadata={
        "kernelspec": {"display_name": "Python (Capstone .venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })


def validate_executed(notebook, function):
    nbformat.validate(notebook)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if any(cell.execution_count is None or any(output.output_type == "error" for output in cell.outputs)
           for cell in cells):
        raise ValueError(f"Notebook execution incomplete for Function {function}")
    panels = cells[-2].outputs
    count = sum("image/png" in output.get("data", {}) for output in panels)
    if count != len(SETTINGS[function][4]):
        raise ValueError(f"Missing conditional figures for Function {function}")
    summary = json.loads(cells[-1].outputs[0].text)
    dimension = len(SETTINGS[function][0])
    text = summary["selected"]
    pattern = r"0\.\d{6}(?:-0\.\d{6}){" + str(dimension - 1) + "}"
    if summary["function"] != function or not re.fullmatch(pattern, text):
        raise ValueError(f"Invalid portal coordinates for Function {function}")
    if not math.isfinite(summary["mae_advisory"]):
        raise ValueError(f"Invalid replay error for Function {function}")
    return summary


def build_report(summaries):
    lines = [
        "# Week 10 Bayesian optimisation with cohort advice", "",
        "**Experimental batch for review.** The official model generates the shortlist and carries 80% of the ranking weight. "
        "The two adviser contributions are capped at 10% each. F8 uses its verified starter-only peak. "
        "The input file is ordered F1–F8.", "",
        "[Copy-ready experimental inputs](advisory-submissions.txt) · "
        "[Initial proposal](initial-proposal.md) · [Current decision](README.md) · [Course data-use review](../../docs/capstone-data-policy.md)", "",
        "| Function | Official-only acquisition winner | Experimental input | Official replay MAE → advisory | "
        "Nearest shortlisted candidate to 2024 best |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for row in summaries:
        f = row["function"]
        destination = f"../../notebooks/Function_{f}/Week_10/cohort_advisory_review.ipynb"
        mae = f"{row['mae_official']:.5g} → {row['mae_advisory']:.5g}"
        lines.append(f"| [F{f}]({destination}) | `{row['official_only']}` | `{row['selected']}` | "
                     f"{mae} | {row['closest_2024_best']:.6f} |")
    lines.extend(["", "Distances are Euclidean on the portal input axes. The replay holds out each official return "
                  "in turn using the latest frozen adviser snapshots. It measures predictive error retrospectively.", "",
                  "The strongest archived regions for F2, F6 and F7 lie outside the official shortlist. "
                  "F3's broader candidate relies on extrapolated upside. Its notebook exposes the local "
                  "alternative alongside the blended winner. Each notebook now displays the two prepared, unsubmitted portal paths separately from that frozen experiment.", "",
                  "Regenerate from the capstone root with "
                  "`./run-python.ps1 stage-2-bbo/scripts/run_week_10_advisory_reviews.py`"
                  ".", ""])
    return "\n".join(lines)


def main():
    summaries = []
    books = []
    with TemporaryDirectory(prefix="week-10-advisory-") as work:
        for function in SETTINGS:
            directory = STAGE / f"notebooks/Function_{function}/Week_10"
            notebook = notebook_for(function)
            NotebookClient(
                notebook, timeout=120, kernel_name="imperial-capstone",
                resources={"metadata": {"path": str(directory)}},
            ).execute()
            summary = validate_executed(notebook, function)
            summaries.append(summary)
            name = "cohort_advisory_review.ipynb"
            staged = Path(work) / f"{function}-{name}"
            nbformat.write(notebook, staged)
            books.append((staged, directory / name))
            print(f"F{function}: {summary['selected']}", flush=True)

        for staged, destination in books:
            destination.write_bytes(staged.read_bytes())
        batch = "\n".join(row["selected"] for row in summaries) + "\n"
        (OUTPUT / "advisory-submissions.txt").write_text(batch, encoding="utf-8")
        (OUTPUT / "advisory-review.md").write_text(build_report(summaries), encoding="utf-8")


if __name__ == "__main__":
    main()

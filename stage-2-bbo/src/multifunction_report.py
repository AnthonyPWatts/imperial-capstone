"""Short tables, charts and data-specific prose for the multi-function studies."""

from itertools import combinations
from pathlib import Path

from IPython.display import Markdown, display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

from initial_exploration import _coordinate_axes, _finish
from multidimensional_exploration import plot_coordinate_pairs
from multifunction_risk import fit_model


FUNCTION_NOTES = {
    4: "The three best observations all have x4 below 0.25. Their other coordinates also differ, so this does not isolate an x4 effect. Across the sample, lower x1, x2 and x4 tend to accompany higher outputs; x3 has a weaker marginal association. The best point is not at the lowest value of every input, so a rule to minimise every coordinate would go beyond the evidence.",
    5: "The three best observations all have x2 above 0.77 and x4 above 0.84. Rank 1 is more than twice as large as rank 2, making its neighbourhood particularly consequential. Higher x4 has the strongest positive rank association in this sample. The course's typically unimodal description motivates looking for one promising region, but these observations do not establish unimodality.",
    6: "The three best observations combine relatively high x4 (about 0.69–0.86) with low x5 (about 0.01–0.33). Higher x4 and lower x5 have the clearest marginal rank associations. The first three coordinates still differ substantially, so they cannot be declared irrelevant or fixed without testing their contribution.",
    7: "Rank 1 is more than twice the runner-up. The best two inputs are widely separated in x1 (about 0.058 and 0.882), while both have x6 close to 0.731. That shared coordinate is a hypothesis to examine, not evidence that x6 alone controls quality. Lower x5 has the clearest negative rank association, but the overall sample leaves considerable room for interactions and separate peaks.",
    8: "Lower x1 and lower x3 have the strongest rank associations with higher outputs. The best observation has all four of its first coordinates near zero, but the runner-up has x2 near 0.63 and x4 near 0.49. The data therefore support investigating low x1 and x3 more strongly than simply setting every coordinate low. Forty points are still sparse in eight dimensions.",
}

ANALOGIES = {
    4: "warehouse-model tuning with expensive evaluations",
    5: "chemical yield, described as typically unimodal",
    6: "recipe quality with negative contributions",
    7: "six model hyperparameters",
    8: "eight-parameter optimisation",
}


def number(value):
    return f"{value:.3e}" if 0 < abs(value) < .001 or abs(value) >= 10000 else f"{value:.4f}"


def coordinates(point, precision=6):
    return "(" + ", ".join(f"{value:.{precision}f}" for value in point) + ")"


def describe_values(data):
    columns = [name for name in data if name.startswith("x")]
    best = data.loc[data.y.idxmax()]
    positive, negative = int((data.y > 0).sum()), int((data.y < 0).sum())
    return Markdown(
        f"The outputs range from **{number(data.y.min())} to {number(data.y.max())}**. "
        f"There are **{positive} positive and {negative} negative** values. "
        f"The highest supplied output is **{number(best.y)}** at "
        f"**{coordinates(best[columns].to_numpy())}**. This is a reference for improvement, "
        "not a known target or ceiling. Every observation remains in the analysis."
    )


def describe_scale(data):
    magnitudes = np.abs(data.y.to_numpy())
    nonzero = magnitudes[magnitudes > 0]
    ratio = nonzero.max() / nonzero.min()
    orientation = ("All outputs are positive, so logarithmic magnitude preserves their ordering."
                   if (data.y > 0).all() else
                   "All outputs are negative, so larger logarithmic magnitude means a worse output."
                   if (data.y < 0).all() else
                   "Magnitude alone does not preserve the quality ordering when signs differ.")
    return Markdown(f"Non-zero magnitudes span **{ratio:.2f} times**, or **{np.log10(ratio):.2f} "
                    f"orders of magnitude**. {orientation} The log view is a display comparison; "
                    "the optimisation objective remains the supplied output.")


def plot_pair_pages(data, *, gaps=False, output):
    columns = [name for name in data if name.startswith("x")]
    pairs = list(combinations(columns, 2))
    maximum = None
    if gaps:
        centres = (np.arange(121) + .5) / 121
        grid = np.array(np.meshgrid(centres, centres)).reshape(2, -1).T
        maximum = max(cKDTree(data[list(pair)]).query(grid)[0].max() for pair in pairs)
    labelled = None if len(data) <= 20 else {1, 2, 3, len(data) - 2, len(data) - 1, len(data)}
    for start in range(0, len(pairs), 6):
        subset = pairs[start:start + 6]
        figure = plot_coordinate_pairs(data, gaps=gaps, pairs=subset,
                                        distance_max=maximum, label_ranks=labelled)
        figure.suptitle(f"{'Projected sampling gaps' if gaps else 'Observed output ranks'}: "
                        f"pairs {start + 1}–{start + len(subset)} of {len(pairs)}", fontsize=16)
        name = f"{'gaps' if gaps else 'ranks'}-{start // 6 + 1}.png"
        _finish(figure, Path(output) / name)
        display(figure)


def full_space_coverage(data):
    columns = [name for name in data if name.startswith("x")]
    inputs = data[columns].to_numpy()
    locations = qmc.Sobol(len(columns), scramble=True, seed=101).random_base2(16)
    distance = cKDTree(inputs).query(locations)[0]
    nearest, indices = cKDTree(inputs).query(inputs, k=2)
    first = np.argmin(nearest[:, 1])
    second = indices[first, 1]
    return Markdown(
        f"A uniform space-filling sample of **65,536 locations in all {len(columns)} "
        f"dimensions** gives a mean nearest-observation distance of **{distance.mean():.3f}**. "
        f"Ten per cent of those locations are farther than **{np.quantile(distance, .9):.3f}** "
        "from any observation. These numerical approximations describe geometry, not model uncertainty.\n\n"
        f"The closest supplied pair in the full space is ranks **{int(data.iloc[first]['rank'])} "
        f"and {int(data.iloc[second]['rank'])}**, separated by **{nearest[first, 1]:.3f}**; "
        f"their outputs differ by **{number(abs(data.iloc[first].y - data.iloc[second].y))}**. "
        "Distances grow naturally with dimension, so comparing these raw distances across "
        "functions is not by itself a measure of worsening coverage."
    )


def association_table(data):
    columns = [name for name in data if name.startswith("x")]
    rows = [{"Input": name, "Pearson": data[name].corr(data.y),
             "Spearman": data[name].corr(data.y, method="spearman")} for name in columns]
    return pd.DataFrame(rows).style.format({"Pearson": "{:.2f}", "Spearman": "{:.2f}"}).hide(axis="index")


def fit_table(study):
    return pd.DataFrame([{"Explanation": item["name"], "Held-out RMSE": number(item["rmse"]),
                          "Held-out MAE": number(item["mae"])} for item in study["checks"]]).style.hide(axis="index")


def fit_findings(study):
    baseline = study["checks"][0]["rmse"]
    winners = [row for row in study["checks"][1:] if row["rmse"] < baseline]
    if winners:
        best = min(winners, key=lambda row: row["rmse"])
        message = (f"**{best['name']}** has the lowest held-out RMSE, "
                   f"**{number(best['rmse'])}**, versus **{number(baseline)}** for the training-mean "
                   f"baseline ({100 * (1 - best['rmse'] / baseline):.1f}% lower). "
                   "That supports some predictive structure, but does not validate extrapolation into empty regions.")
    else:
        message = ("**None of the three models beats the training-mean baseline on held-out RMSE.** "
                   "The query calculation is therefore a conditional experiment under uncertain models; "
                   "its numerical precision is not evidence of predictive accuracy.")
    return Markdown(message)


def current_risk_table(study):
    return pd.DataFrame([{"Explanation": model["name"],
                          "Expected shortfall": number(model["current"]["risk"]),
                          "Simulation SE": number(model["current"]["mc_se"])}
                         for model in study["models"]]).style.hide(axis="index")


def prediction_table(data, study):
    inputs = data[[name for name in data if name.startswith("x")]].to_numpy()
    point = np.array(study["pool"][study.get("recommended_index", study["robust_index"])])[None, :]
    rows = []
    for name in ("Additive", "Local", "Flexible"):
        fit = fit_model(inputs, data.y.to_numpy(), name)
        mean, deviation = fit["model"].predict(point, return_std=True)
        rows.append({"Explanation": name, "Predicted mean": number(mean[0]),
                     "Latent response SD": number(deviation[0]),
                     "New observation SD": number(np.sqrt(deviation[0] ** 2 + fit["noise_sd"] ** 2))})
    return pd.DataFrame(rows).style.hide(axis="index")


def kernel_table(study):
    return pd.DataFrame([{"Explanation": model["name"], "Fitted kernel": model["kernel"],
                          "Fit warnings": len(model["warnings"])}
                         for model in study["models"]]).style.hide(axis="index")


def candidate_table(study):
    rows = []
    for slot, (label, index) in enumerate(zip(study["candidate_names"], study["candidate_indices"])):
        row = {"Query": f"Q{slot + 1}: {label}", "Coordinates": coordinates(study["pool"][index], 3),
               "Coverage gain": f"{100 * study['checked_coverage'][slot]:.2f}%"}
        row.update({m["name"] + " KG": number(m["gains"][index]) for m in study["models"]})
        rows.append(row)
    return pd.DataFrame(rows).style.hide(axis="index")


def risk_findings(study):
    index = study["robust_index"]
    before = max(m["current"]["risk"] for m in study["models"])
    after = study["worst_after"][index]
    active = max(study["models"], key=lambda m: m["current"]["risk"] - m["gains"][index])
    best_fit = min(study["checks"][1:], key=lambda row: row["rmse"])["name"]
    return Markdown(
        "Minimising the **largest model-specific expected shortfall after one query** selects "
        f"**{coordinates(study['pool'][index])}**. On this criterion, expected shortfall falls "
        f"from **{number(before)} to {number(after)}**, a reduction of **{number(before - after)} "
        "output units**.\n\n"
        f"The **{active['name']}** model determines the largest remaining risk at this choice. "
        f"The best held-out predictor among the fitted models is **{best_fit}**. "
        "Protecting against the most uncertain included model is a conservative decision rule, "
        "not evidence that this model is correct. No probabilities have been assigned to the models. "
        "Changing the model set could change this recommendation."
    )


def noise_table(study):
    rows = [{"Explanation": item["model"], "Noise / sample SD": f"{item['noise_fraction']:.0%}",
             "Best tested coordinates": coordinates(item["point"], 3),
             "Best KG": number(item["gain"]),
             "Proposed query / best KG": f"{item['recommended_gain'] / item['gain']:.0%}"}
            for item in study["noise_checks"]]
    return pd.DataFrame(rows).style.hide(axis="index")


def grid_table(study):
    return pd.DataFrame([{"Explanation": row["model"],
                          "Initial pool winner": coordinates(row["coarse_point"], 3),
                          "Larger pool winner": coordinates(row["fine_point"], 3),
                          "Initial KG": number(row["coarse_gain"]),
                          "Larger-pool KG": number(row["fine_gain"])}
                         for row in study["grid_checks"]]).style.hide(axis="index")


def refinement_findings(study):
    fine = study["refinement"]
    moved = not np.array_equal(fine["previous"], fine["fine_choice"])
    return Markdown(
        f"The main comparison contains **{len(study['pool'])} points**; the final larger-pool "
        f"check contains **{fine['pool_count']}**. On the latter, the conservative KG choice is "
        f"**{coordinates(fine['fine_choice'])}**. Its largest model-specific expected shortfall "
        f"falls from **{number(fine['worst_current_risk'])} to {number(fine['fine_after'])}**. "
        f"The earlier choice retains **{fine['retained_fraction']:.1%}** of this expected reduction.\n\n"
        + ("The location changes with the larger pool. " if moved else "The same location wins this larger-pool check. ")
        + "Our stability rule keeps the earlier input if it retains at least 95% of the larger-pool "
        "gain; otherwise it adopts the new choice. This is a practical tolerance, not a confidence "
        "level. Function 1's final coverage decision is a separate judgement about model credibility. "
        "Risk magnitudes change when the decision pool changes, so the checks do not establish "
        "convergence to the continuous-domain risk."
    )


def recommendation_findings(study):
    index = study.get("recommended_index", study["robust_index"])
    point = study["pool"][index]
    slot = study["candidate_indices"].index(index)
    function = study["function_id"]
    notes = {
        1: "The raw-output KG comparison is inconclusive about a better experiment. Retain the earlier geometrically justified coverage query.",
        3: "This tests the region around ranks 3 and 4. It values whether those relatively good outcomes extend into nearby three-input combinations. The failure to beat the held-out mean baseline keeps the interpretation tentative.",
        4: "The proposal investigates the region around the better observed settings while changing their combination. Held-out prediction provides substantially stronger support for modelling here than in Function 3.",
        5: "The larger-pool check moves the proposal towards high x2 as well as high x3 and x4. This probes the high-output region near the current best, with x1 changed rather than simply repeating the incumbent.",
        6: "The point keeps relatively high x4 and low x5 while testing a different combination of the other inputs. This follows the clearest observed pattern without declaring the remaining coordinates irrelevant.",
        7: "This is a controlled perturbation of the best supplied point: x4 increases by 0.1, with the other coordinates retained to six decimal places. The additive model driving this conservative choice has weak held-out performance, so treat this as a hypothesis test around an unusually good result.",
        8: "The corner preserves low x1 and x3 while testing the other coordinates jointly, including high x5. This is a decision-value choice rather than an average-coverage optimum. The local model retains this point across the noise checks, although the other explanations can prefer different inputs.",
    }
    return Markdown(
        f"**Proposed input: {coordinates(point)}**\n\n"
        "Portal format: `" + "-".join(f"{value:.6f}" for value in point) + "`\n\n"
        f"**Basis: {study['recommendation_reason']}.** This is candidate **Q{slot + 1}** in the "
        f"comparison, including the paired thirteen-query simulation. Its estimated full-space "
        f"coverage improvement is **{100 * study['checked_coverage'][slot]:.2f}%**.\n\n" + notes[function]
    )


def final_risk_table(study):
    rows = []
    for slot, name in enumerate(study["candidate_names"]):
        row = {"First query": f"Q{slot + 1}: {name}"}
        for model in study["models"]:
            row[model["name"]] = (number(model["rollout"]["mean"][slot][-1]) + " ± "
                                  + number(model["rollout"]["mc_se"][slot][-1]))
        rows.append(row)
    return pd.DataFrame(rows).style.hide(axis="index")


def rollout_findings(study):
    selected = study["candidate_indices"].index(study.get("recommended_index", study["robust_index"]))
    rows = []
    for model in study["models"]:
        losses = np.asarray(model["rollout"]["losses"])
        difference = losses[0, :, -1] - losses[selected, :, -1]
        error = difference.std(ddof=1) / np.sqrt(len(difference))
        rows.append(f"**{model['name']}: {number(difference.mean())} ± {number(error)}**")
    return Markdown(
        "The paired final-shortfall difference, **coverage first minus the proposed KG query first**, "
        "is " + "; ".join(rows) + ". Positive values favour the KG start; negative values favour "
        "coverage. The ± values are one simulation standard error, not uncertainty about the real "
        "function. Small differences relative to these errors do not establish a preferred first move "
        "over thirteen rounds. A one-step KG recommendation is not an optimal thirteen-step plan."
    )


def plot_query_values(study, *, save_to=None):
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), layout="constrained", sharey=True)
    indices = study["candidate_indices"]
    for axis, model in zip(axes, study["models"]):
        values = np.asarray(model["gains"])[indices]
        colours = ["#d67d24" if i == study.get("recommended_index", study["robust_index"])
                   else "#26769a" for i in indices]
        axis.bar(np.arange(len(indices)) + 1, values, color=colours)
        axis.set(title=model["name"], xlabel="First-query candidate", xticks=np.arange(len(indices)) + 1,
                 xticklabels=[f"Q{i + 1}" for i in range(len(indices))])
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Knowledge gradient (output units)")
    return _finish(figure, save_to)


def plot_rollouts(study, *, save_to=None):
    figure, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), layout="constrained", sharey=True)
    chosen = study["candidate_indices"].index(study.get("recommended_index", study["robust_index"]))
    candidates = list(dict.fromkeys([0, 1, chosen]))
    colours = ("#287a69", "#6e6c83", "#d67d24")
    upper = 0
    for axis, model in zip(axes, study["models"]):
        mean = np.asarray(model["rollout"]["mean"])
        error = np.asarray(model["rollout"]["mc_se"])
        upper = max(upper, np.max(mean[candidates] + error[candidates]))
        for index, colour in zip(candidates, colours):
            axis.errorbar([1, 4, 13], mean[index], yerr=error[index], capsize=3, marker="o",
                          label=f"Q{index + 1}", color=colour)
        axis.set(title=model["name"], xlabel="Queries completed", xticks=[1, 4, 13])
        axis.grid(axis="y", alpha=.2)
    axes[0].set_ylim(0, 1.08 * upper)
    axes[0].set_ylabel("Simulated expected shortfall")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3)
    return _finish(figure, save_to)


def plot_cube_kg(study, *, save_to=None):
    if study["dimensions"] != 3:
        raise ValueError("Cube slices require three inputs.")
    axis_values = np.round(np.linspace(0, .999999, 9), 6)
    first, second = np.meshgrid(axis_values, axis_values)
    levels = axis_values[[1, 4, 7]]
    tree = cKDTree(study["pool"])
    locations = [np.c_[first.ravel(), second.ravel(), np.full(first.size, level)] for level in levels]
    indices = []
    for points in locations:
        distance, index = tree.query(points)
        if distance.max() > 1e-8:
            raise ValueError("The plotted slice must consist of evaluated KG candidates.")
        indices.append(index)
    maximum = max(max(model["gains"]) for model in study["models"])
    figure, axes = plt.subplots(3, 3, figsize=(12.6, 11.8), layout="constrained")
    for row, model in enumerate(study["models"]):
        for column, (level, index) in enumerate(zip(levels, indices)):
            axis = axes[row, column]
            heat = axis.pcolormesh(axis_values, axis_values, np.asarray(model["gains"])[index].reshape(9, 9),
                                   vmin=0, vmax=maximum, shading="nearest", cmap="viridis")
            _coordinate_axes(axis, ("x1", "x2"))
            axis.set_title(f"{model['name']} · x3 = {level:.3f}")
    figure.colorbar(heat, ax=list(axes.ravel()), shrink=.6, label="Knowledge gradient (output units)")
    return _finish(figure, save_to)

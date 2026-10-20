"""Reproducible Function 1 magnitude-target experiment, confined to the side quest."""

import nbformat


def add_magnitude_predictions(notebook):
    notebook.cells = [cell for cell in notebook.cells if not cell.id.startswith("cohort-f1-activity-")]
    for cell in notebook.cells:
        if cell.cell_type != "markdown":
            continue
        cell.source = cell.source.replace("the other 12", "the other 18")
        if cell.source.startswith("## 11. Target sensitivity"):
            cell.source = cell.source.replace("Five negative", "Ten negative").replace("eight positives", "nine positives").replace("all 13", "all 19")
        cell.source = cell.source.replace(
            "For Week 4, the [exploration notebook's coverage sequence](exploration.ipynb) remains the query decision;",
            "The [exploration notebook's original coverage sequence](exploration.ipynb) records the submitted Week 4 decision;")
    section = [
        nbformat.v4.new_markdown_cell(r"""## Activity hypothesis: predict absolute magnitude

The [cohort exploration](exploration.ipynb) asks whether both signs indicate
activity. Here we **refit** the same five methods to a second target:

$$g(y)=\log_{10}(1+|y|/s)=|h(y)|,\qquad s=10^{-200}.$$

This retains the previous log compression across almost 200 orders of magnitude.
It folds each measured negative value upwards **before fitting**; it is not the
absolute value of the old fitted surface. Every non-zero output contributes,
without inventing a detection threshold. Large negative outputs now count as
strong activity, whereas the course still rewards the highest signed output.

Inputs, observations, leave-one-out folds, model settings and candidate pool
stay the same. Neither target is clipped. Compare each target with **its own**
mean baseline: a lower error for g does not establish better signed-score
prediction, source detection or distance estimation.
"""),
        nbformat.v4.new_code_cell('''from cohort_magnitude_models import (
    plot_relative_validation, plot_target_shapes, plot_target_slices, plot_activity_search,
)

magnitude_y = np.abs(height_y)
magnitude_comparison, magnitude_held_out, magnitude_fitted = compare_models(
    make_models(), inputs, magnitude_y,
)
target_validation = pd.concat({
    "Signed h": comparison[["LOO MAE", "MAE / baseline"]],
    "Magnitude g": magnitude_comparison[["LOO MAE", "MAE / baseline"]],
}, axis=1)
display(target_validation.round(3))
display(plot_relative_validation(comparison, magnitude_comparison,
                                save_to=figures / "magnitude-relative-validation.png"))
magnitude_gp_name = magnitude_comparison.loc[gp_names, "LOO MAE"].idxmin()
magnitude_gp = magnitude_fitted[magnitude_gp_name]
print(f"Lowest-error magnitude GP: {magnitude_gp_name}")
print("LOO is an exploratory comparison on 19 pooled records, not a separate final test.")
'''),
        nbformat.v4.new_markdown_cell("""### What happens to the fitted shape?

Read across each row to change the target while holding the method fixed.
The signed surfaces must turn between large negative and positive heights
at nearby inputs. The magnitude surfaces can join those readings into a
broader high-activity region. Shared colours and axes make the upward fold
visible. The quadratic and the two GPs still impose different shapes.
The magnitude quadratic retains a north-east emphasis; the local GP favours
a smaller south-west peak, while the broad GP joins them into a central band.
The local signed GP also overshoots into deep troughs between measurements:
those extremes are predictions, not additional observed readings.

The cross-sections then hold x2 at the **best signed observation's coordinate**
in both fits. Their bands describe the declared GP assumptions; they are not
probabilities of a source being present.
"""),
        nbformat.v4.new_code_cell('''display(plot_target_shapes(inputs, raw_y, fitted, magnitude_fitted,
                           save_to=figures / "signed-versus-magnitude-models.png"))
display(plot_target_slices(inputs, height_y, fitted, magnitude_fitted, anchor,
                           save_to=figures / "signed-versus-magnitude-slices.png"))
'''),
        nbformat.v4.new_code_cell('''magnitude_grid = prediction_grid(magnitude_gp, inputs, uncertainty=True)
magnitude_fitted_terrain = fitted_terrain(
    magnitude_grid, inputs, magnitude_y, raw_y,
    title=f"Function 1 cohort: magnitude fit — {magnitude_gp_name}",
    target_label="Magnitude height g",
)
magnitude_fitted_terrain.data[-1].name = "Largest observed |y|"
magnitude_terrain_path = figures / "magnitude-fitted-terrain.html"
magnitude_fitted_terrain.write_html(magnitude_terrain_path, config=PLOT_CONFIG)
magnitude_fitted_terrain.show(config=PLOT_CONFIG)
'''),
        nbformat.v4.new_markdown_cell("""### Where does the activity objective send us?

Score the **same finite pool** used above, with all coordinates rounded before
prediction. Compare each GP with itself under h and g, then consider the GP
with the lowest held-out error for each target. This separates the effect of
changing the target from the effect of changing the preferred length scale.

Weight 0 seeks the highest predicted mean. Weight 1.96 adds an uncertainty
allowance (UCB); weights 1 and 3 show sensitivity. Coordinates below are
unevaluated proposals, not new observations or altered submissions. Nearest
distance uses both inputs on the unit square; inside the sampled hull does
not guarantee good coverage or a reliable prediction.
"""),
        nbformat.v4.new_code_cell('''from scipy.spatial import Delaunay

candidate_hull = Delaunay(inputs)
activity_rows = []
for target_name, target_models in (("Signed h", fitted), ("Magnitude g", magnitude_fitted)):
    for model_name in gp_names:
        candidate_scores = score_query_candidates(target_models[model_name], query_candidates)
        for weight in (0.0, 1.0, 1.96, 3.0):
            # Reuse predictions so every weight sees exactly the same pool.
            index = (candidate_scores.posterior_mean + weight * candidate_scores.posterior_sd).idxmax()
            row = candidate_scores.loc[index]
            point = row[input_columns].to_numpy(dtype=float)
            activity_rows.append({
                "Target": target_name, "GP": model_name, "Weight": weight,
                "x1": point[0], "x2": point[1], "Mean": row.posterior_mean,
                "SD": row.posterior_sd, "UCB": row.posterior_mean + weight * row.posterior_sd,
                "Inside hull": bool(candidate_hull.find_simplex(point) >= 0),
                "Nearest observed distance": np.linalg.norm(inputs - point, axis=1).min(),
            })
activity_proposals = pd.DataFrame(activity_rows)
display(activity_proposals.loc[activity_proposals.Weight.isin([0, 1.96])]
        .set_index(["Target", "GP", "Weight"]).round(6))
print("All four uncertainty weights, retaining both GP assumptions:")
display(activity_proposals.set_index(["Target", "GP", "Weight"])[
    ["x1", "x2", "Inside hull", "Nearest observed distance"]
].round(6))

def activity_proposal(target_name, model_name, weight=1.96):
    return activity_proposals.loc[
        activity_proposals.Target.eq(target_name) & activity_proposals.GP.eq(model_name)
        & activity_proposals.Weight.eq(weight)
    ].iloc[0]

selected_activity = activity_proposal("Magnitude g", magnitude_gp_name)
local_activity = activity_proposal("Magnitude g", "GP: local (l=0.15)")
display(plot_activity_search(
    magnitude_grid, inputs, raw_y, selected_activity[input_columns].to_numpy(float),
    local_activity[input_columns].to_numpy(float),
    save_to=figures / "magnitude-exploration-areas.png",
))
'''),
        nbformat.v4.new_code_cell('''from IPython.display import Markdown

signed_activity = activity_proposal("Signed h", proposal_gp_name)
peak_activity = activity_proposal("Magnitude g", magnitude_gp_name, 0.0)
best_magnitude_input = inputs[np.argmax(magnitude_y)]
display(Markdown(f"""### What this identifies in the pooled snapshot

- **Central follow-up:** the preferred magnitude GP ({magnitude_gp_name}) has its
  finite-pool mean peak at **({peak_activity.x1:.6f}, {peak_activity.x2:.6f})**.
  Adding the standard uncertainty allowance moves it to
  **({selected_activity.x1:.6f}, {selected_activity.x2:.6f})**. This is close to
  the strong signed return, in the region also containing large negative readings.
  It is a local follow-up for examining the sign transition and activity hypothesis.
- **South-west extension:** the shorter-scale magnitude GP suggests
  **({local_activity.x1:.6f}, {local_activity.x2:.6f})**, about
  **{local_activity['Nearest observed distance']:.3f}** from the nearest observation.
  That is an alternative, more sparsely measured area, dependent on the kernel assumption.
- **Changed emphasis:** the signed-target policy suggested
  **({signed_activity.x1:.6f}, {signed_activity.x2:.6f})**. Magnitude shifts attention
  towards and to the south-west of the central negative cluster, whose strongest
  observed |y| is at **({best_magnitude_input[0]:.6f}, {best_magnitude_input[1]:.6f})**.

The magnitude GP's held-out MAE is
**{magnitude_comparison.loc[magnitude_gp_name, 'MAE / baseline']:.1%} of its own mean
baseline**, compared with **{comparison.loc[proposal_gp_name, 'MAE / baseline']:.1%}**
for the preferred signed GP. The folded target is easier to predict in this
sample. Both targets were assessed on the same small snapshot used to choose
between methods; these are not independent test results.

The central UCB point is only **{selected_activity['Nearest observed distance']:.3f}**
from the nearest observation. It refines a known neighbourhood rather than
opening an unexplored part of the square. Magnitude targets are non-negative,
but these unconstrained models can still predict below zero near the boundary.

The local GP's weight-3 magnitude proposal moves to an unobserved boundary corner.
That is uncertainty-driven extrapolation, not evidence for another source.
None of these fits establishes a shared source, an ECG-like response, or that
large magnitude is close to a source. A magnitude query can return a very
negative value and perform poorly on the signed maximisation objective.
"""))
'''),
    ]
    for index, cell in enumerate(section):
        cell.id = f"cohort-f1-activity-{index}"
    # Preserve the existing closing note and original signed analyses.
    notebook.cells[-1:-1] = section
    notebook.cells[0].source = notebook.cells[0].source.replace(
        "The methods, target treatments and search settings are retained.",
        "The original methods and search settings are retained. A separate magnitude-target\ncomparison tests the activity hypothesis alongside the signed models.")

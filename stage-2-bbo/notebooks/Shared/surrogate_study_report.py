"""Generate discussion evidence directly from saved numerical study results."""

import json
from pathlib import Path

import numpy as np


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _row(rows, name):
    return next(row for row in rows if row["Model"] == name)


def discussion_evidence(summary):
    reg = summary["regression_comparison"]
    cls = summary["classification_comparison"]
    nn_reg = _row(reg, summary["regression_representative"])
    nn_cls = _row(cls, summary["classification_representative"])
    gp = _row(reg, summary["reference_gp"])
    linear, logistic, svm = _row(reg, "Linear regression"), _row(cls, "Logistic regression"), _row(cls, "RBF SVM")
    seed = summary["regression_seed_comparison"][0]
    class_seed = summary["classification_seed_comparison"][0]
    top_inputs = sorted(summary["input_gradients"], key=lambda row: row["NN sampled mean absolute gradient"], reverse=True)
    influential = ", ".join(f"{row['Input']} ({row['NN sampled mean absolute gradient']:.4g})" for row in top_inputs[:3])
    near = ", ".join(f"{row['observation_id']} (class {int(row['Good class'])}, decision score {row['Decision score']:.3f})"
                     for row in summary["rbf_support_examples"][:3])
    ascent, descent = summary["gradient_paths"]["Ascent (maximise surrogate)"], summary["gradient_paths"]["Descent (minimise surrogate)"]
    warnings = sum(sum(counts.values()) for task in summary["fit_warnings"].values() for counts in task.values())
    reg_better = seed["mean"] < gp["LOO RMSE"]
    class_better = class_seed["mean"] > logistic["LOO balanced accuracy"]
    boundary_guidance = (
        f"The RBF classifier misses all **{svm['False negatives']}** good held-out "
        f"results, with **{svm['LOO balanced accuracy']:.1%}** leave-one-out balanced "
        "accuracy. Its current fitted boundary has not earned a role guiding "
        "queries. This predictive failure does not rule out every classification approach."
        if summary["held_out_good_count"] > 0
        and svm["False negatives"] == summary["held_out_good_count"]
        else "Nearby uncertain classifications can motivate exploration only where "
        "held-out predictive evidence supports using the classifier."
    )
    return f"""## Evidence for the seven reflection points

### 1. Inputs acting as support vectors

The full-fit RBF SVM identifies **{summary['rbf_support_count']} of {summary['observations']}**
observations as actual support vectors. Examples nearest its decision boundary
in absolute decision score are {near}. The linear SVM uses
**{summary['linear_support_count']}** support vectors. Their full input profiles
are retained in the notebook table and figure. Kernel decision scores are not
Euclidean distances from the boundary. These points support the fitted classifier,
not a known true boundary of the hidden function. A large support-vector fraction
alone neither validates nor invalidates the boundary. {boundary_guidance}

![Actual support-vector input profiles.](support-vector-profiles.png)

### 2. Surrogate gradients and directions

The study trains neural regressors and backpropagates their output derivatives
to the original input coordinates, including both fitted scalers. Independent
finite-difference checks pass for the representative regressor and classifier.
The bounded ascent demonstration accepts **{len(ascent)-1} steps**, changing the
NN prediction from **{ascent[0]['prediction']:.6g} to {ascent[-1]['prediction']:.6g}**;
descent accepts **{len(descent)-1} steps**, changing it from
**{descent[0]['prediction']:.6g} to {descent[-1]['prediction']:.6g}**. Zero accepted
steps mean no improving step was found under the declared step limits.
The project maximises its signed score, so ascent is the relevant query direction;
descent is included to address the assignment's minimisation wording. All these
predictions use **{summary['target_units']}**, and no new course evaluations occur.

![Bounded surrogate ascent and descent.](gradient-paths.png)

### 3. Classification framing and exploration trade-offs

"Good" means a result at least as high as the training history's 75th percentile.
Every validation fold estimates its threshold from training outcomes only.
The final full-data threshold is **{summary['classification_threshold']:.6g}** in
{summary['target_units']}, giving **{summary['full_good_count']} good** and
**{summary['observations']-summary['full_good_count']} bad** labels. Fold thresholds
vary; held-out metrics evaluate this training-relative rule rather than one fixed
oracle boundary. The representative NN misses **{nn_cls['False negatives']}** good
held-out results and falsely labels **{nn_cls['False positives']}** bad results as
good. False negatives risk excluding promising regions; false positives spend
scarce queries. Labels discard output magnitude, so this is supporting evidence
alongside numerical maximisation, not a replacement objective.

![Held-out confusion matrices.](classification-confusions.png)

### 4. Choice of model and interpretability

The representative NN's LOO RMSE is **{nn_reg['LOO RMSE']:.6g}**, compared with
**{linear['LOO RMSE']:.6g}** for linear regression and **{gp['LOO RMSE']:.6g}** for
the reference GP (**{summary['reference_gp']}**). Averaging all three seeds of
the selected NN architecture gives RMSE **{seed['mean']:.6g}** with seed SD
**{seed['std']:.6g}**. The architecture is selected by this mean, and seed 42
is the fixed representative rather than the luckiest seed. The average network
{'improves on' if reg_better else 'does not improve on'} the reference GP on these
observed validation targets. Linear coefficients are easier to explain; neural
curves and input gradients provide local interpretation. GP posterior uncertainty
and variation across neural initialisations answer different questions.

![All declared model comparisons.](validation-overview.png)

### 5. Input influence and priorities for exploration

The largest NN mean absolute gradients across the observed input locations are
**{influential}**, in {summary['target_units']} per unit input. These sampled
sensitivities average the selected architecture's three fitted seeds; they are
neither causal effects nor global importance. The separate signed gradients at
the incumbent and probability gradients near a fitted classification boundary
retain all inputs and can suggest different priorities. Every coordinate shares
the original 0–1 range. Larger, seed-stable gradients can motivate controlled
local perturbations, while small or unstable gradients still warrant coverage
checks rather than permanent exclusion of an input.

![Input gradients and seed variation.](input-gradients.png)

### 6. Neural decision boundary and backpropagation

The representative neural classifier has held-out balanced accuracy
**{nn_cls['LOO balanced accuracy']:.3f}**, good-result recall **{nn_cls['Good recall']:.3f}**
and Brier score **{nn_cls['Brier score']:.3f}**. Its full-fit training balanced
accuracy is **{nn_cls['Training balanced accuracy']:.3f}**, so the training versus
held-out gap remains visible. Logistic regression scores
**{logistic['LOO balanced accuracy']:.3f}** and the RBF SVM
**{svm['LOO balanced accuracy']:.3f}** on the same folds.
Backpropagation supplies weight gradients to the L-BFGS training optimiser;
the later input backpropagation differentiates the fixed trained model's
P(good). The latter is checked against finite differences and used to select
two informative plotting axes. Other coordinates remain fixed in higher-dimensional
slices, and observations away from that slice are omitted. These plots visualise
an estimated boundary, not measured truth throughout the domain. Softmax or
sigmoid confidence is not automatically calibrated uncertainty about unsampled regions.

![Conditional classifier decision boundaries.](classification-boundaries.png)

### 7. Added neural flexibility versus simpler models

For regression, the selected architecture's mean RMSE is **{seed['mean']:.6g}**,
with individual seeds spanning **{seed['min']:.6g}–{seed['max']:.6g}**. The
representative fit has training RMSE **{nn_reg['Training RMSE']:.6g}**, compared
with held-out RMSE **{nn_reg['LOO RMSE']:.6g}**. For classification, its selected
architecture averages balanced accuracy **{class_seed['mean']:.3f}** with seed SD
**{class_seed['std']:.3f}**, which {'exceeds' if class_better else 'does not exceed'}
the logistic baseline's **{logistic['LOO balanced accuracy']:.3f}**. Regression and
classification can therefore give different answers about the value of flexibility.
The bounded study retains **{warnings} fitting warnings**, including iteration
limits where present, rather than increasing the budget until a network wins.
There are only {summary['observations']} observations for {summary['dimensions']}
inputs. All configuration selection reuses these small validation samples, so
the comparison is exploratory and supplies no independent final-test guarantee.
The submitted Week 5 inputs remain unchanged; useful future decisions require
new returned evidence rather than lower training error alone.

![Variation across the three declared neural initialisations.](seed-stability.png)
"""


def write_study_report(summary, folder):
    folder = Path(folder)
    evidence = discussion_evidence(summary)
    introduction = f"""# Function {summary['function']} — Week 5 neural-network and surrogate study

**Study date: {summary['study_date']}.** This is a retrospective discussion
experiment using {summary['observations']} official observations through round 4.
The fifth query is already submitted and its result is pending. This experiment
does not explain or replace the earlier submission decision. Sources and settings
are recorded in [results.json](results.json); the executed notebook is
[surrogate_comparison.ipynb](../surrogate_comparison.ipynb).

Three architectures (tanh-8, tanh-16 and ReLU-8-8) each use seeds 11, 42 and 73
for regression and classification. Input scaling and regression target scaling
are fitted inside every leave-one-out fold. Function 1 uses its existing fixed,
order-preserving signed-log transform; all other functions use original score
units. Neural training uses L-BFGS, backpropagated gradients, L2 alpha 0.1,
400 iterations at most and no early-stopping holdout. Neither depth nor higher
input dimension is assumed to make a network preferable.

SVM probability calibration is not fitted to these tiny class samples. Its
decision scores support labels, ROC AUC and boundary plots; its Brier score and
log loss are unavailable. Probability near 0.5 may reflect noisy ambiguity as
well as lack of information, so it is not by itself a rule for spending a query.

The common candidate comparison and gradient paths evaluate fitted models only.
Variation across neural initialisations measures optimisation sensitivity, not
calibrated posterior uncertainty. Support-vector examples identify observations
by ID; no additional source arrays are exported to the result record.

"""
    lifecycle = """
## Course lifecycle

| Step | Application |
| --- | --- |
| 1. Define the goal and scope | Gather comparative evidence for the seven reflection criteria, using existing observations. |
| 2. Gather the data | Read the independent official Week 5 snapshot through round 4; add no peer or portal returns. |
| 3. Explore the data | Inspect input dimensions, output range, class balance and the existing incumbent. |
| 4. Clean and preprocess the data | Validate the snapshot; fit scaling inside folds and preserve Function 1's fixed signed-log ordering. |
| 5. Select and engineer features | Retain all original inputs; keep the existing Function 2 x1-only GP as an explicit comparison. |
| 6. Define the machine-learning task | Compare score regression with a separate good/bad classification framing. |
| 7. Partition the data | Use the same leave-one-out rows for all methods; fit each classification threshold on training outcomes only. |
| 8. Select and train candidate methods | Run the declared linear, kernel, GP and small neural models with fixed seeds and budgets. |
| 9. Evaluate and interpret the results | Compare held-out errors, seed stability, support vectors, checked input gradients and conditional boundaries. |
| 10. Deploy and iterate | Deployment is not applicable to this retrospective study. The submitted inputs remain fixed; revisit after new feedback. |
"""
    (folder / "report.md").write_text(introduction + evidence + lifecycle, encoding="utf-8", newline="\n")
    return evidence


def summarise_studies(root):
    """Refresh the eight-function reading index from saved results, without fitting."""
    root = Path(root)
    studies = []
    for number in range(1, 9):
        folder = root / f"stage-2-bbo/notebooks/Function_{number}/Week_05"
        if number <= 4:
            folder /= "official"
        path = folder / "surrogate-study/results.json"
        if not path.is_file():
            return None
        studies.append((folder, json.loads(path.read_text(encoding="utf-8"))))
    text = """# Week 5 neural-network and surrogate comparison

**Study date: 29 October 2026.** Eight executed notebooks compare numerical-score
surrogates and good/bad classifiers using **207 official observations through
round 4**. These retrospective studies follow the fifth-query submission. Its
results are pending, its input file is unchanged and none of the hypothetical
queries has been evaluated by the course portal.

The [final Week 5 discussion draft](../docs/week-05-discussion-draft.md) records
the posted reflection and the interpretation of the weak F3 classifier and
suspected F5 boundary maximum, with their limitations.

## Regression: compare every seed

The NN column reports the architecture with the lowest mean leave-one-out RMSE
across seeds 11, 42 and 73. The spread is one seed standard deviation, not a
confidence interval. Seed 42 is the fixed representative for gradients and plots.
The reference is the lowest-RMSE non-neural method in the declared matrix.
Function 1 uses signed-log score units; other rows use original score units.
Compare methods within each row rather than raw errors between functions.

| Function | NN architecture | NN mean RMSE ± seed SD | Best non-neural method | Non-neural RMSE | NN / non-neural error |
| --- | --- | ---: | --- | ---: | ---: |
"""
    fits, neural_fits, winners = 0, 0, []
    for folder, summary in studies:
        relative = folder.relative_to(root / "stage-2-bbo").as_posix()
        nn = summary["regression_seed_comparison"][0]
        other = next(row for row in summary["regression_comparison"] if not row["Model"].startswith("NN "))
        ratio = nn["mean"] / other["LOO RMSE"]
        if ratio < 1:
            winners.append(str(summary["function"]))
        text += f"| [{summary['function']}](../{relative}/surrogate_comparison.ipynb) | {nn['architecture']} | {nn['mean']:.6g} ± {nn['std']:.3g} | {other['Model']} | {other['LOO RMSE']:.6g} | {ratio:.3f} |\n"
        fits += summary["estimator_fits"]
        neural_fits += 2*len(summary["settings"]["architectures"])*len(summary["settings"]["seeds"])
    text += "\nThe selected neural architecture's mean RMSE beats the best declared non-neural alternative for **Function " + ", ".join(winners) + "**.\n"
    text += """
Some individual seeds rank better than their architecture average. Those
outcomes remain visible rather than being promoted as stable model improvements.

## Classification: good/bad boundaries

Each fold labels a test result good if it reaches the training outcomes' 75th
percentile, estimated without that held-out outcome. Full-fit boundary plots
use a separately recorded full-data threshold. Balanced accuracy gives the
two classes equal weight; the notebooks also retain recall, precision,
false negatives/positives and probability metrics where available.

| Function | NN architecture | NN mean balanced accuracy ± seed SD | Logistic regression | RBF SVM |
| --- | --- | ---: | ---: | ---: |
"""
    for _, summary in studies:
        nn = summary["classification_seed_comparison"][0]
        logistic = _row(summary["classification_comparison"], "Logistic regression")
        svm = _row(summary["classification_comparison"], "RBF SVM")
        text += f"| {summary['function']} | {nn['architecture']} | {nn['mean']:.3f} ± {nn['std']:.3f} | {logistic['LOO balanced accuracy']:.3f} | {svm['LOO balanced accuracy']:.3f} |\n"
    text += """
The classification comparison can favour a network even where a GP gives better
numerical predictions. Classification discards score magnitude and remains a
secondary framing of maximisation. A flexible fitted boundary is not a known
true boundary, and a probability near 0.5 can reflect noisy ambiguity as well
as lack of information.

## Inputs, support vectors and seven reflection sections

All studies identify actual SVM support vectors and check neural input gradients
against finite differences. The input ranking below averages absolute regression
gradients across observed locations and the selected architecture's three seeds.
It measures sampled model sensitivity, rather than causal or global importance.

| Function | Largest sampled NN sensitivities | RBF support vectors / observations | Full report |
| --- | --- | ---: | --- |
"""
    for folder, summary in studies:
        relative = folder.relative_to(root / "stage-2-bbo").as_posix()
        rows = sorted(summary["input_gradients"], key=lambda row: row["NN sampled mean absolute gradient"], reverse=True)
        inputs = ", ".join(row["Input"] for row in rows[:3])
        text += f"| {summary['function']} | {inputs} | {summary['rbf_support_count']} / {summary['observations']} | [Seven reflection sections and figures](../{relative}/surrogate-study/report.md) |\n"
    text += f"\nThe bounded experiment ran **{neural_fits} final neural fits** and **{fits:,} estimator fits** including validation and other methods.\n"
    text += """
## Limits and reproducibility

The declared neural configurations are tanh-8, tanh-16 and ReLU-8-8 with three
initialisations each, L-BFGS, L2 alpha 0.1 and a maximum of 400 iterations.
Regression also includes the mean baseline, linear/ridge methods, kernel ridge,
RBF SVR and GPs with shared/separate length scales. Function 2 retains its x1-only
GP hypothesis. Input and regression target scaling are fitted inside folds.

Convergence and kernel-bound warnings remain in the saved results. These are
warning-message counts, not failed-fit counts: one GP fit can warn about several
bounded coordinates. Iteration-limited networks may behave differently with a
larger training budget; this study does not prove that an architecture is
intrinsically unsuitable. Neural seed variation is optimisation sensitivity,
not calibrated posterior uncertainty.

The same validation outcomes are reused to compare architectures and reference
models. There is no independent final test set or statistical guarantee for the
selected configuration. The observed inputs were collected adaptively, and
higher-dimensional boundary views are conditional slices with other inputs
held fixed. Off-slice observations are not projected onto them.

Run from the Capstone root:

```powershell
.\\run-python.ps1 stage-2-bbo/scripts/run_surrogate_studies.py
```

Use `--functions 3 8` for a targeted rerun. Each study saves an executed notebook,
seven PNG figures, a source-bound result record and a report covering all seven
reflection criteria. The runner refreshes this index once all eight results are
available. See the [execution guide](../docs/reproducing-the-analyses.md#week-5-discussion-surrogate-comparisons).
"""
    destination = root / "stage-2-bbo/results/week-05-surrogate-comparison.md"
    destination.write_text(text, encoding="utf-8", newline="\n")
    return destination

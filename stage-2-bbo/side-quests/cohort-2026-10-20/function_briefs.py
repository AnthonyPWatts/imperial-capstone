"""Attach the dated course-description review without clearing notebook outputs."""

from pathlib import Path

import nbformat


STUDY = Path(__file__).resolve().parent
SOURCE = "https://classroom.emeritus.org/courses/18642/modules/items/3237303"

# These are concise paraphrases of the course's sample applications. The
# observations and model interpretations describe the fixed 20 October snapshot.
BRIEFS = {
    1: {
        "description": "Locate possible contamination sources in a two-input field; the lesson associates non-zero readings with proximity. Maximise the supplied signed output.",
        "observations": """### What the cohort observations suggest

The **19 observations** contain strong negative readings and a much smaller
positive incumbent in the broad central region:

| Reading | Input | Signed output |
| --- | --- | ---: |
| Highest signed output | `(0.540351, 0.524565)` | `+1.180859e-13` |
| Largest absolute magnitude | `(0.419675, 0.463269)` | `-0.006137883` |

The strongest readings of opposite signs are about **0.135** apart on the unit
square, and their magnitudes differ by **10.7 orders of magnitude**. All current
readings are non-zero, including extremely small values. Literal zero versus
non-zero therefore supplies no useful detection threshold for this snapshot.

Both signs could help identify an active region. A response with positive and
negative lobes, resembling an ECG trace, is a hypothesis worth examining; the
observations do not establish that shape, a shared source, or a distance-to-source
relationship. Triangles joining observations do not measure a ridge through gaps.

### Exploration implication

Retain signed views for the maximisation objective and magnitude views for the
separate activity hypothesis. The central sign transition merits examination;
large negative readings remain poor signed scores. See
[model predictions](model_predictions.ipynb) for the separate magnitude refits.
""",
        "models": r"""### What the fitted comparisons add

The separate activity target is
$$g(y)=\log_{10}(1+|y|/10^{-200})=|h(y)|.$$
It folds measured negative targets upwards **before refitting**, retaining the
existing log compression. It is not the absolute value of an old fitted surface.
The magnitude fits form a broader central band of predicted activity; the
quadratic and the two GP length scales still disagree about its detailed shape.

The broad magnitude GP has the lowest held-out error: **31.9% of its own mean
baseline MAE**, versus **96.0%** for the preferred signed GP. These ratios compare
predictability of different targets; they do not establish better signed-score
prediction or validate the source hypothesis.

On the common candidate pool, the magnitude broad-GP UCB point is
**`(0.563545, 0.515620)`**, only about **0.025** from an observation. It refines a
known central neighbourhood. The local magnitude GP suggests the less-sampled
south-west extension **`(0.321717, 0.357282)`** instead. Its weight-3 proposal moves
to a boundary corner through uncertainty, not measured evidence for another source.
""",
    },
    2: {
        "description": "Tune two inputs to improve a noisy log-likelihood score. Multiple local peaks may occur; the objective is to maximise the returned score.",
        "observations": """### What the cohort observations suggest

The **18 observations** provide support near **x1 = 0.67–0.73**. The three highest
scores occur at `(0.702637, 0.926564)` → **0.611205**,
`(0.690829, 0.926564)` → **0.592732**, and
`(0.665800, 0.123969)` → **0.538996**.

High scores at very different x2 values suggest two potentially good
neighbourhoods. A continuous ridge and separate high regions are both plausible;
the observations do not establish a high ridge all the way between them. The
upper neighbourhood has more nearby published support than the lower one.

The brief explicitly allows noise. A lower nearby return can reflect noise,
real variation or both; there are no repeated full input combinations here to
measure repeat-evaluation noise independently. The supplied best remains the
pooled incumbent.

### Exploration implication

Investigate whether the upper and lower high regions connect, for example by
varying x2 while keeping x1 near 0.7. Keep the possibility of additional local
peaks open. Model the supplied signed score directly; it already represents a
log-likelihood, so another logarithm is not required by the description.
""",
        "models": """### What the fitted comparisons add

The local GP has held-out **MAE 0.1398**, versus **0.2112** for the mean baseline.
This supports useful predictive structure in this snapshot. Its conditional x2
slice predicts a dip between the upper and lower high regions, but that dip is
a model hypothesis rather than a measured continuous profile.

The broad GP performs worse than the mean baseline. Smooth appearance alone
therefore does not favour its explanation. The current UCB proposal,
**`(0.595377, 0.993143)`**, is outside the sampled hull; its mean **0.840** exceeds
the best observation **0.611**. That is explicit extrapolation, not a confirmed
new peak. The assumed GP noise and latent-function bands do not measure the
unknown repeat-evaluation noise described by the course.
""",
    },
    3: {
        "description": "Choose amounts of three compounds to reduce adverse reactions. The competition uses a transformed output for which higher is better.",
        "observations": """### What the cohort observations suggest

All **23 outputs** are negative. The best remains the supplied
`(0.492581, 0.611593, 0.340176)` → **-0.034835**; the second best is
`(0.600097, 0.725136, 0.066089)` → **-0.036378**. The best peer addition,
`(0.492581, 0.861593, 0.399583)` → **-0.047594**, adds support without improving
the incumbent.

These points suggest examining moderate x1, moderately high x2 and low-to-moderate
x3. The compound-combination analogy makes interactions worth testing, without
establishing them. A high-x3 query also returned a reasonably good **-0.054209**,
while a near-zero-x3 peer query returned **-0.138436**. Because the other inputs
differed, these observations do not justify a blanket rule based on x3 alone.

Less negative is better. Absolute magnitude would reverse the observed ranking.
The exact transformation is not specified, so a score such as -0.035 is not a
literal fractional count of adverse reactions.

### Exploration implication

Refine the neighbourhood of the best complete combination and vary one compound
at a time with the other two held fixed. A two-input projection can hide a
different third coordinate; use true conditional slices when interpreting effects.
""",
        "models": """### What the fitted comparisons add

The local GP has held-out **MAE 0.03686**, versus **0.05953** for the mean baseline.
The quadratic is worse than that baseline. This favours local modelling over
assuming one broad quadratic shape among the methods examined.

The prediction error is much larger than the gap between the two best measured
scores, so their fine ranking does not locate a resolved optimum. The current
UCB point **`(0.658460, 0.558234, 0.006012)`** explores a low-x3 region outside the
full 3D sampled hull. Its predicted mean is **-0.031223** and SD is **0.078202**;
the positive UCB **0.122053** is an acquisition score, not a predicted positive
return. The description supplies no exact noise level or proven attainable optimum.
""",
    },
    4: {
        "description": "Tune four hyperparameters of an ML approximation to expensive warehouse calculations. The score compares with the expensive baseline; local optima are expected.",
        "observations": """### What the cohort observations suggest

The **39 observations** contain a concentrated promising neighbourhood. The four
highest reported outputs have **x1 = 0.41–0.46**, **x2 = 0.37–0.42**,
**x3 = 0.36–0.43** and **x4 = 0.38–0.42**, approximately centred on
**`(0.44, 0.40, 0.39, 0.40)`**.

| Evidence | Best output |
| --- | ---: |
| Shared initial observations | -4.025542 |
| Own returned queries | -0.129918 |
| Published cohort | +0.712146 |

The positive peer-reported return shows that zero is not a ceiling in this
snapshot. The baseline wording does not define enough physical detail to
translate the score into a particular operational improvement. The inputs are
model tuning settings, rather than known physical warehouse coordinates.

### Exploration implication

Refine the strong four-input cluster while retaining some search elsewhere.
Multiple local optima are a stated possibility; the current cluster is not
established as globally best. The observations do not support a general rule
such as reducing every input independently.
""",
        "models": """### What the fitted comparisons add

The GP's held-out **RMSE is 1.553**, versus **9.466** for the mean baseline, an
approximately **84% reduction**. The mean-predictor benchmark is distinct from
the expensive application baseline mentioned in the course description.

The GP still underestimates the strongest observations; its held-out prediction
for the **0.712146** incumbent is about **-0.617**. Kernel ridge misses the summit
more severely. Broad predictive usefulness therefore does not establish precise
resolution near the best settings.

The current exploratory UCB proposal has mean **-1.390**, below the observed
incumbent, and SD **0.958**. It investigates uncertainty rather than promising a
new record. The GP noise SD is assumed from the training output spread; the
warehouse narrative does not empirically establish that noise level or time drift
in the synthetic function.
""",
    },
    5: {
        "description": "Maximise chemical-process yield over four inputs. The lesson describes the function as typically unimodal, suggesting one dominant peak.",
        "observations": """### What the cohort observations suggest

The **28 observations** strongly favour the upper corner. The best is
**8662.405** with all four inputs at **0.999999**. A nearby query,
`(0.996065, 0.996814, 0.999999, 0.999999)`, returned **8525.753**, only **1.6%**
lower. Together they support a high-yield neighbourhood rather than one isolated
exceptional reading.

Own queries occupy the first three places in this collected snapshot. The best
peer addition is **2605.263**. Strong records consistently have high x3 and x4;
the own results suggest substantial further gains when x1 and x2 are also high,
without establishing monotonic effects throughout the domain.

### Exploration implication

A single peak need not be in the middle of the allowed box. The unimodal clue
supports concentrating on the upper boundary, but is not proof of a global
optimum or a symmetric hill. Test whether slightly reducing one input, while
holding the other three fixed, can improve on the corner. The nearby follow-up
changed x1 and x2 together, so their separate effects remain unresolved.
Profile labels of 1.000 are rounded displays of valid 0.999999 coordinates.
""",
        "models": """### What the fitted comparisons add

The raw-output GP has the lowest held-out **RMSE, 375.8**, versus **2346.4** for
the mean baseline, about an **84% reduction**. The log-target GP has lower MAE
(**212.5** versus **286.8**), but higher RMSE. The differing rankings reflect the
emphasis on large errors versus typical absolute errors, not different observations.

The raw GP predicted the latest held-out **8525.753** as **8573.040**, within
about **0.6%**. That is encouraging local evidence, but the fit has a very close
training neighbour at the upper corner; it is not a test of the entire domain.

All four tested exploration weights select the same nearby candidate,
`(0.999515, 0.989282, 0.999999, 0.999999)`, from this pool and fitted model. Its
mean **8507.7** is below the incumbent; its UCB exceeds the incumbent through
uncertainty. The boundary's detailed shape and exact maximum remain unresolved.
""",
    },
    6: {
        "description": "Balance five ingredient settings against recipe penalties for flavour, consistency, calories, waste and cost. The lesson describes a negative score improving towards zero.",
        "observations": """### What the cohort observations suggest

The **28 observations** favour combinations with **low x5**, **low x2** and
**fairly high x4**. Among the top five, x5 is roughly 0–0.10, x2 is 0.07–0.26
and x4 is 0.71–1.00. Ingredient names in the brief are illustrative rather than
a verified mapping to the input columns.

The own first query remains the pooled best:
**`(0.344331, 0.260596, 0.575965, 0.745346, 0.066311)` → -0.378691**.
The best peer return is **-0.462480**. Low x5 is promising, but zero has not been
established as optimal: an own zero-x5 query returned **-1.124281**, with other
settings changed as well. The best recipe also keeps x4 below its upper limit.

### Scoring interpretation and exploration implication

The lesson's final sentence suggests that negating an already-negative total is
equivalent to moving it towards zero. Those operations reverse the ranking.
Follow the explicit competition rule: **maximise the returned signed score**.
Zero is an ideal in the penalty analogy, not an observed or proven attainable
optimum; no new output bound is imposed by this review.

Adding several assessment criteria together does not make ingredient effects
independent. A useful local comparison varies x5 around the best recipe while
holding its other four settings fixed, then examines x4 separately.
""",
        "models": """### What the fitted comparisons add

The GP has held-out **RMSE 0.2475**, versus **0.6119** for the mean baseline,
about a **60% reduction**. Its retrospective held-out prediction of the latest
query was **-0.5574**, against **-0.5514** observed. This supports useful
structure without resolving the exact best recipe.

Every tested exploration weight chooses x5 = 0 in the current candidate pool,
but the other settings change. This is sensitivity within one fitted model,
not independent evidence that zero x5 is optimal. The standard UCB candidate
has mean **-0.5084**, below the **-0.3787** incumbent, and SD **0.1762**; it is
an information-gathering proposal. Keep all five inputs available and use local
comparisons to separate their effects. The sign interpretation above is unchanged.
""",
    },
    7: {
        "description": "Tune six ML hyperparameters to maximise a performance score. The lesson suggests established modelling practice as a possible guide to the search.",
        "observations": """### What the cohort observations suggest

The **38 observations** show a strong measured neighbourhood. The eight highest
records have **x1 ≤ 0.058** and **x6 ≈ 0.70–0.78**. The best two keep x4 near
**0.285**, with intermediate settings for x2, x3 and x5. Low x1 alone does not
establish a good six-input combination.

| Own evidence | Best output |
| --- | ---: |
| Shared initial observations | 1.364968 |
| Round 1 | 1.405759 |
| Round 2 | 2.075222 |
| Round 3 | 2.552418 |

The latest two own returns are the highest in this pooled snapshot. At portal
precision, the first query changed only x4, from **0.218118 to 0.318118**, raising
the score by **0.040791**. Later larger gains changed several coordinates together
and cannot be attributed to a single input. Several peer queries also target
the initial strong neighbourhood; this selected cluster does not cover the domain.

### Exploration implication

Continue examining the low-x1 neighbourhood, including the balance among x2,
x3 and x5 with x6 around 0.7. Literature would be more actionable with the actual
model identity and parameter definitions: the examples do not identify x1 as a
learning rate or supply a column mapping. Scores above 1 also mean the returned
scale is not literal unscaled accuracy or F1. No single-peak guarantee is given.
""",
        "models": """### What the fitted comparisons add

The GP has held-out **RMSE 0.2519**, versus **0.6564** for the mean baseline,
about a **62% reduction**. It underpredicted the latest strong return as
**2.1785**, against **2.5524** observed. Closely neighbouring cohort inputs also
make these errors incomplete evidence about less-sampled regions.

The current UCB proposal extends the local neighbourhood with x1 near zero,
x6 near 0.7 and changes to the middle settings. Its mean **2.4744** is below
the incumbent, with SD **0.2068**. Pure exploitation offers only about **0.0073**
predicted improvement, much less than its SD. The model therefore supplies a
reason to investigate locally, not a clear prediction of another record or proof
of a global maximum. Hyperparameter names and score bounds remain unspecified.
""",
    },
    8: {
        "description": "Optimise eight influential inputs. The lesson favours practical strong local maxima when global search is difficult in the high-dimensional domain.",
        "observations": """### What the cohort observations suggest

The five strongest of **49 records** keep x1–x4 low, especially x1 and x4.
The other four settings vary considerably. This suggests a promising region;
compensating effects could explain similarly good outputs from different
combinations, so those inputs cannot be declared irrelevant.

| Evidence | Output |
| --- | ---: |
| Best shared initial observation | 9.598482 |
| Best peer-reported observation | 9.890878 |
| Best own returned query | 9.518300 |
| Latest own query | 8.798303 |

Pooling improves the known incumbent; own returns have not beaten the supplied
best. A controlled comparison between own rounds 1 and 2 changed only x8 from
**0 to 0.999999**, increasing output by approximately **0.020**. That is a small
effect for this particular combination, not proof of global irrelevance. Raising
x4 and x6 together from the round 1 settings reduced output by about **0.700**;
their separate contributions remain unresolved.

### Exploration implication

Refine the low-x1–x4 neighbourhood around the peer's 9.891 result while keeping
the remaining inputs available. The brief says all eight affect output, without
implying equal influence. Its named hyperparameters and 0–1 accuracy are examples,
not verified input types or the actual score scale. Actual outputs are roughly
5.6–9.9, and neither 1 nor 10 is imposed as a ceiling. Strong local maxima are a
practical aim; 49 selected observations do not map the full eight-dimensional box.
""",
        "models": """### What the fitted comparisons add

The GP has held-out **RMSE 0.2921**, versus **1.1721** for the mean baseline,
about a **75% reduction**. However, it predicted the latest corner query as
**9.7222**, against **8.7983** observed. That miss contributes about **20%** of
its total held-out squared error. Good overall validation does not establish
reliable predictions at distant corners.

The standard UCB winner is another corner, with mean **9.4188** below the
**9.8909** incumbent and SD **0.6928**. It earns its place through uncertainty.
A local alternative near the peer incumbent has mean **10.0663** and SD
**0.2242**; pure exploitation selects another nearby alternative. These are
conditional predictions, not evidence for an actual score above 10.

The local alternatives have stronger nearby measured support. The broad corner
is a deliberate information-gathering alternative that merits caution after
the latest miss. This interpretation leaves the saved policy and submissions intact.
""",
    },
}


def add_function_brief(notebook, number, *, prediction):
    """Insert or refresh one Markdown cell; preserve every other cell verbatim."""
    brief = BRIEFS[number]
    prefix = f"cohort-f{number}-brief-"
    cells = [cell for cell in notebook.cells if not cell.id.startswith(prefix)]
    heading = "Function description and modelling clues" if prediction else "Function description and cohort clues"
    text = f"""## {heading}

**Course description, paraphrased.** {brief['description']}

Source: [Mini-lesson 12.8: function descriptions]({SOURCE}), reviewed
20 October 2026. The notes below interpret the fixed cohort snapshot collected
on that date. Peer outputs are published reports, not independently verified
portal returns. Every official objective remains maximisation of signed y.

"""
    if prediction:
        companion = "exploration.ipynb" if number <= 3 else "observation_review.ipynb"
        text += f"Read the [cohort evidence and exploration clues]({companion}) alongside these model results.\n\n"
        text += brief["models"]
        text += "\nHeld-out comparisons and method selection use the same small snapshot; they are not independent final-test results. The notes do not change targets, settings or query policies.\n"
    else:
        text += brief["observations"]
        text += "\nThese patterns describe selected observations, not controlled causal effects or a known true surface. Exploration suggestions are hypotheses for the side quest, not submissions.\n"
    cell = nbformat.v4.new_markdown_cell(text)
    cell.id = prefix + ("models" if prediction else "observations")
    cells.insert(1, cell)
    changed = notebook.cells != cells
    notebook.cells = cells
    return changed


def update_existing():
    """Refresh the existing reading notes without rebuilding or rerunning models."""
    changed = 0
    for number in BRIEFS:
        week = STUDY / f"notebooks/Function_{number}/Week_04"
        observed = "exploration.ipynb" if number <= 3 else "observation_review.ipynb"
        for name in (observed, "model_predictions.ipynb"):
            path = week / name
            notebook = nbformat.read(path, as_version=4)
            if add_function_brief(notebook, number, prediction=name == "model_predictions.ipynb"):
                nbformat.validate(notebook)
                nbformat.write(notebook, path)
                changed += 1
    print(f"Updated description-and-clue notes in {changed} notebooks; existing code and outputs retained.")
    return changed


if __name__ == "__main__":
    update_existing()


"""Present the Week 10 decision record against the saved official acquisition model."""

import re

import numpy as np
import pandas as pd

from cohort_advisory_bo import SETTINGS, fit_owner, owner_posterior
from weekly_peak_review import separable_quadratic


BATCH_FILES = {
    "Advisory experiment": "advisory-submissions.txt",
    "Revised cohort-informed": "submissions.txt",
    "Official-data fallback": "official-data-submissions.txt",
}

RATIONALE = {
    1: (
        "The revised input uses the official seven-positive source fit constrained to the diagonal. "
        "The archived peak supports that constraint and its local fit predicts about 1.999997. "
        "The fallback uses the unconstrained official source peak. A plateau of near-2 returns "
        "would support the local source and leave distant peaks unresolved.",
        "Refine the better-supported local shape if the return improves. Use a distinct nearby control "
        "or a second-source test if it falls short.",
    ),
    2: (
        "The revised input repeats the published archive centre exactly. Its eight archived returns "
        "span 0.75571 to 0.91987, all above the official 0.72306. The owner GP strongly prefers "
        "the familiar x1≈0.70 band and has no official evaluation at the centre. The fallback uses "
        "the leading official two-query candidate in the established band. One transfer return "
        "tests the region and provides limited evidence about its mean.",
        "After a competitive centre return, sample a neighbouring point or an allowed repeat to "
        "separate location from variability. After a weak result, prioritise the established x1 band.",
    ),
    3: (
        "The revised input changes only x2, placing it 0.005 from Aby's near-zero measurement. "
        "The advisory experiment ranked a distant x2≈0.95 point first on an extrapolated positive "
        "mean. The official fallback follows the measured direction from x2=0.58 to 0.61 and "
        "tests x2=0.65 on the same x1/x3 line. Zero remains a hypothesis about the ceiling.",
        "Tighten x2 spacing on the controlled slice if the chosen side improves the score. A weak "
        "return favours the best measured setting and a carefully spaced check.",
    ),
    4: (
        "Both batches use the midpoint between the two strongest official x3 values, with x1, x2 "
        "and x4 held fixed. The earlier batch used a different reference slice and the advisory "
        "winner moves several coordinates. The prepared midpoint gives a clean comparison against "
        "0.65013 and 0.66609 on the same line.",
        "If the midpoint improves the incumbent, bracket the high part of the x3 slice. Otherwise use "
        "the measured better endpoint when testing another coordinate.",
    ),
    5: (
        "Both batches keep the single-axis x4 experiment at the upper corner. It clears the 0.020 "
        "spacing rule and the official surrogate expects a lower score than the 8,662.405 incumbent. "
        "The useful outcome is a measurement of the corner slope and its persistence at the larger step.",
        "Use the result to decide whether the corner loss is consistent with previous probes. Spend "
        "further queries where they provide evidence beyond another step down the same arm.",
    ),
    6: (
        "The revised input is the centroid of four strong archived interior points from one reporting "
        "ID. Their outputs range from −0.065 to −0.006. The bounded advisory ranking and the "
        "official-data fallback keep x5=0, where the official evidence is concentrated. This is a "
        "deliberate transfer test outside the shortlist, with one participant supplying the tight cluster.",
        "Refine the interior only after a competitive official return. A weak result favours the "
        "confirmed face for the next query.",
    ),
    7: (
        "The revised input is the coordinatewise median of six distinct archival points above 3.3 "
        "and one independent corroboration. It shifts x1 and x3 into the stronger measured region. "
        "The fallback changes x3 on the official x1=0 face. The owner GP retains that face because "
        "the broader interior basin lacks nearby official coverage.",
        "Use a competitive interior return to justify a local follow-up. Otherwise use the face as a "
        "control and reassess with the remaining budget.",
    ),
    8: (
        "Both routes use the six-decimal stationary point of the official starter-only separable "
        "quadratic. It predicted the nine later official returns within 2×10⁻¹⁴. An independent "
        "campaign sampled this coordinate, which was recovered here from the starter data.",
        "If the portal returns 10 to the expected precision, use any required subsequent query to "
        "check local curvature. Investigate any material mismatch before spending a refinement.",
    ),
}

STATUS = (
    "**Prepared and unsubmitted as of 25 November 2026.** Round 9 is the latest confirmed feedback. "
    "Four portal queries remain per function, including Week 10. Use the cohort-informed route only "
    "if the course permits that use, including the exact published F2 repeat. The official-data "
    "fallback is ready if the rules remain unclear near the portal deadline."
)

REFERENCES = (
    "[Week 10 course check and exact batches](../../../submissions/Week_10/README.md) · "
    "[Cohort advice and replay](../../../submissions/Week_10/advisory-review.md)"
)

SCORE_NOTE = (
    "The estimates below use the advisory review's official-data GP comparator and predictive return "
    "standard deviation, which includes modelled observation noise. F8 uses the validated starter "
    "quadratic and omits a standard deviation because that fit supplies no probabilistic interval. "
    "The rows are proposed inputs with model estimates."
)


def decision_inputs(stage, function):
    """Load the four recorded query paths in their original order."""
    folder = stage / "submissions/Week_10"
    table = (folder / "initial-proposal.md").read_text(encoding="utf-8")
    initial = dict(re.findall(r"(?m)^\|\s*([1-8])\s*\|\s*\x60([0-9.-]+)\x60\s*\|", table))
    batches = {"Initial proposal": [initial[str(index)] for index in SETTINGS]}
    for label, filename in BATCH_FILES.items():
        batches[label] = (folder / filename).read_text(encoding="utf-8").splitlines()
    dimension = len(SETTINGS[function][0])
    pattern = rf"0\.\d{{6}}(?:-0\.\d{{6}}){{{dimension-1}}}"
    if any(len(batch) != len(SETTINGS) or not re.fullmatch(pattern, batch[function-1])
           for batch in batches.values()):
        raise ValueError(f"Invalid saved decision batch for Function {function}")
    return {name: batch[function-1] for name, batch in batches.items()}


def decision_comparison(stage, function, official, owner_model=None, structural=None):
    """Compare forecasts at the recorded locations using the official observations."""
    inputs = decision_inputs(stage, function)
    points = np.array([point.split("-") for point in inputs.values()], dtype=float)
    axes = [f"x{axis+1}" for axis in range(points.shape[1])]
    x, y = official[axes].to_numpy(), official["y"].to_numpy()
    rows = pd.DataFrame({"Stage": list(inputs), "Portal input": list(inputs.values())}).set_index("Stage")
    if function == 8:
        fit = structural if structural is not None else separable_quadratic(x, y, official["source_round"].gt(0))
        deviations = points - np.asarray(fit["point"])
        rows["Starter quadratic (score)"] = np.round(fit["predicted_maximum"] -
                                                     deviations**2 @ np.asarray(fit["curvature"]), 7)
    else:
        owner = owner_model if owner_model is not None else fit_owner(function, x, y)
        gp, _, scale, _ = owner[0]
        mean, latent = owner_posterior(owner, points)
        rows["Official GP mean (score)"] = np.round(mean, 7)
        rows["Predictive return SD"] = np.round(np.hypot(latent, scale * np.sqrt(gp.kernel_.k2.noise_level)), 7)
    return rows


def decision_markdown(function):
    rationale, follow_up = RATIONALE[function]
    return (
        "### Prepared Week 10 decision\n\n"
        f"{STATUS}\n\n{rationale}\n\n"
        f"**Week 11 depends on the return.** {follow_up}\n\n"
        f"{REFERENCES}\n\n{SCORE_NOTE}"
    )

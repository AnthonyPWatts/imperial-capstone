"""Nearby-pair evidence and conditional GP noise fits for any input dimension."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import WhiteKernel


def nearby_output_differences(data, *, input_columns, output_column="y",
                              cutoffs=(.015, .03, .05, .1), precision=6):
    """Return full-space pairs and cutoff sensitivity, without assuming a flat function.

    RMS difference / sqrt(2) estimates noise SD only for independent, equal-variance
    same-input repeats. At distinct inputs it also includes true response variation.
    Exact-input groups are repeat candidates; independence requires provenance.
    Submission-precision collisions are counted separately, never promoted to repeats.
    """
    inputs = data[list(input_columns)].to_numpy(dtype=float)
    target = data[output_column].to_numpy(dtype=float)
    cutoffs = np.asarray(cutoffs, dtype=float)
    if (inputs.ndim != 2 or inputs.shape[1] < 1 or len(inputs) < 2
            or not np.isfinite(inputs).all() or not np.isfinite(target).all()
            or not data.observation_id.is_unique or data.observation_id.isna().any()
            or cutoffs.ndim != 1 or not len(cutoffs)
            or not np.isfinite(cutoffs).all() or (cutoffs <= 0).any()):
        raise ValueError("Require complete distinct IDs, finite observations and positive distance cutoffs")
    left, right = np.triu_indices(len(target), k=1)
    distance = np.linalg.norm(inputs[left] - inputs[right], axis=1)
    delta = target[left] - target[right]
    ids = data.observation_id.to_numpy()
    pairs = pd.DataFrame({"Left": ids[left], "Right": ids[right], "Distance": distance,
                          "Absolute output difference": np.abs(delta),
                          "Half squared difference": delta ** 2 / 2})
    exact = (inputs[left] == inputs[right]).all(axis=1)
    rounded = (inputs[left].round(precision) == inputs[right].round(precision)).all(axis=1)
    rows = []
    for cutoff in cutoffs:
        keep = distance <= cutoff
        rows.append({"Maximum distance": cutoff, "Pairs": int(keep.sum()),
                     "Distinct observations": len(np.unique(np.r_[left[keep], right[keep]])),
                     "Noise SD proxy": float(np.sqrt(np.mean(delta[keep] ** 2 / 2))) if keep.any() else np.nan})
    # Pool within-group sums of squares with n-1 degrees of freedom per exact group.
    _, groups = np.unique(inputs, axis=0, return_inverse=True)
    sse, degrees, repeat_groups = 0., 0, 0
    for group in np.unique(groups):
        values = target[groups == group]
        if len(values) > 1:
            sse += float(np.sum((values-values.mean()) ** 2))
            degrees += len(values) - 1
            repeat_groups += 1
    summary = dict(observations=len(target), input_dimensions=inputs.shape[1],
                   exact_input_pairs=int(exact.sum()), submission_precision_pairs=int(rounded.sum()),
                   exact_input_groups=repeat_groups, repeat_degrees_of_freedom=degrees,
                   repeat_candidate_sd=float(np.sqrt(sse/degrees)) if degrees else None,
                   output_sd=float(target.std(ddof=0)))
    return dict(pairs=pairs.sort_values("Distance", kind="stable").reset_index(drop=True),
                cutoffs=pd.DataFrame(rows), summary=summary)


def fit_noise_diagnostics(models, inputs, target, *, model_names,
                           initial_noise_variance=.01, noise_variance_bounds=(1e-6, 1.),
                           jitter=1e-10):
    """Fit RBF + independent white-noise variants without modifying proposal models.

    Noise bounds and fitted variance use normalised target units. Convert its
    square root back to output SD. These are conditional in-sample likelihood
    fits, not validated noise measurements or replacements for the declared policy.
    """
    inputs, target = np.asarray(inputs), np.asarray(target)
    if (target.ndim != 1 or len(target) != len(inputs) or not np.isfinite(target).all()
            or target.std(ddof=0) == 0):
        raise ValueError("Noise fits need finite, non-constant targets matching the inputs")
    rows = []
    for name in model_names:
        estimator = clone(models[name])
        gp = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
        if not gp.normalize_y:
            raise ValueError("Noise fits require normalised GP targets")
        gp.set_params(kernel=clone(gp.kernel) + WhiteKernel(initial_noise_variance, noise_variance_bounds), alpha=jitter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(inputs, target)
        variance = float(gp.kernel_.k2.noise_level)
        fraction = float(np.sqrt(variance))
        rows.append({"Model": name, "Noise SD": fraction * float(target.std(ddof=0)),
                     "Noise / output SD": fraction, "Normalised noise variance": variance,
                     "At noise lower bound": bool(np.isclose(variance, noise_variance_bounds[0], rtol=1e-4, atol=0)),
                     "At noise upper bound": bool(np.isclose(variance, noise_variance_bounds[1], rtol=1e-4, atol=0)),
                     "Fitted kernel": str(gp.kernel_),
                     "Fit warnings": " | ".join(dict.fromkeys(str(item.message) for item in caught))})
    return pd.DataFrame(rows)

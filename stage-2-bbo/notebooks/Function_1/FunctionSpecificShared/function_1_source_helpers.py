"""Small log-quadratic source hypotheses for Function 1's official observations."""

import numpy as np
from scipy.optimize import lsq_linear


def _inputs(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Supply a finite two-column input array")
    if ((values < 0) | (values >= 1)).any():
        raise ValueError("Inputs must lie in [0, 1)")
    return values


def _design(inputs, anisotropic):
    quadratic = -(inputs**2) if anisotropic else -np.sum(inputs**2, axis=1, keepdims=True)
    return np.column_stack([np.ones(len(inputs)), inputs, quadratic])


def fit_gaussian_source(inputs, outputs, *, positive_only=True, positive_count=5,
                        anisotropic=False, minimum_curvature=1e-8):
    """Fit log10 response using a concave circular or axis-aligned quadratic.

    Positive-source fits select their strongest positive training readings only.
    The global-envelope control instead fits every absolute training output.
    Selection happens inside each training fold, never using its held-out target.
    This shape is a hypothesis; it cannot represent negative responses.
    """
    inputs, outputs = _inputs(inputs), np.asarray(outputs, dtype=float)
    if outputs.shape != (len(inputs),) or not np.isfinite(outputs).all() or (outputs == 0).any():
        raise ValueError("Supply matching finite, non-zero training outputs")
    if not np.isfinite(minimum_curvature) or minimum_curvature <= 0:
        raise ValueError("Minimum curvature must be positive and finite")
    if positive_count is not None and (not isinstance(positive_count, int) or positive_count < 1):
        raise ValueError("Positive count must be a positive integer or None")
    if positive_only:
        selected = np.flatnonzero(outputs > 0)
        selected = selected[np.argsort(-outputs[selected], kind="stable")]
        if positive_count is not None:
            selected = selected[:positive_count]
    else:
        if positive_count is not None:
            raise ValueError("The all-magnitude envelope must use positive_count=None")
        selected = np.arange(len(inputs))
    design = _design(inputs[selected], anisotropic)
    parameters = 5 if anisotropic else 4
    if len(selected) < parameters or np.linalg.matrix_rank(design) < parameters:
        raise ValueError("Too few independent readings for the declared source shape")
    target = np.log10(np.abs(outputs[selected]))
    curvature_count = 2 if anisotropic else 1
    lower = np.r_[np.full(3, -np.inf), np.full(curvature_count, minimum_curvature)]
    fitted = lsq_linear(design, target, bounds=(lower, np.full(parameters, np.inf)), tol=1e-12)
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError(f"Source fit failed: {fitted.message}")
    curvature = fitted.x[3:] if anisotropic else np.repeat(fitted.x[3], 2)
    stationary = fitted.x[1:3] / (2 * curvature)
    return dict(coefficients=fitted.x.tolist(), anisotropic=bool(anisotropic),
                positive_only=bool(positive_only), positive_count=positive_count,
                selected_indices=selected.tolist(), selected_count=len(selected),
                parameters=parameters, residual_degrees_of_freedom=len(selected)-parameters,
                stationary_point=stationary.tolist(), curvature=curvature.tolist(),
                sigma=(1 / np.sqrt(2 * curvature * np.log(10))).tolist(),
                curvature_at_floor=bool(np.any(curvature <= minimum_curvature * 1.01)),
                design_condition_number=float(np.linalg.cond(design)),
                training_log10_mae=float(np.mean(np.abs(design @ fitted.x-target))),
                minimum_curvature=minimum_curvature)


def predict_source_log10(fit, inputs):
    """Predict in log10 magnitude units, retaining tails below float64 raw range."""
    return _design(_inputs(inputs), fit["anisotropic"]) @ np.asarray(fit["coefficients"])


def source_peak(fit, *, precision=6):
    """Find this assumed concave shape's permitted maximum, without a candidate search."""
    if not isinstance(precision, int) or precision < 1:
        raise ValueError("Declare a positive portal precision")
    stationary = np.asarray(fit["stationary_point"])
    point = np.clip(stationary, 0, 1-10.**-precision).round(precision)
    log10_height = float(predict_source_log10(fit, point[None])[0])
    height = float(10.**log10_height) if log10_height <= np.log10(np.finfo(float).max) else None
    return dict(point=point.tolist(), portal_input="-".join(f"{x:.{precision}f}" for x in point),
                predicted_log10_height=log10_height, predicted_height=height,
                stationary_point_inside_domain=bool(((stationary >= 0) & (stationary < 1)).all()),
                curvature_at_floor=fit["curvature_at_floor"])


def source_target_predictions(log10_values, *, height_scale=1e-200):
    """Keep positive source predictions distinct from signed and magnitude observations.

    Overflowing raw predictions remain explicit infinities for validation to flag;
    the log-domain predictions stay usable. No observed value is clipped or changed.
    """
    log10_values = np.asarray(log10_values, dtype=float)
    if not np.isfinite(log10_values).all() or not np.isfinite(height_scale) or height_scale <= 0:
        raise ValueError("Require finite log predictions and a positive target scale")
    heights = np.logaddexp(0, log10_values*np.log(10)-np.log(height_scale))/np.log(10)
    with np.errstate(over="ignore", under="ignore"):
        raw = np.power(10., log10_values)
    return dict(signed=heights, magnitude=heights.copy(), raw_magnitude=raw,
                raw_signed=raw.copy(), log10_magnitude=log10_values.copy())


def source_leave_one_out(inputs, outputs, **settings):
    """Refit selection and source parameters without each official observation."""
    inputs, outputs = _inputs(inputs), np.asarray(outputs, dtype=float)
    predicted = np.empty(len(outputs))
    folds = []
    for held in range(len(outputs)):
        training = np.delete(np.arange(len(outputs)), held)
        fitted = fit_gaussian_source(inputs[training], outputs[training], **settings)
        predicted[held] = predict_source_log10(fitted, inputs[held:held+1])[0]
        folds.append(dict(held_out_index=held,
                          selected_training_indices=training[fitted["selected_indices"]].tolist(),
                          peak=source_peak(fitted), selected_count=fitted["selected_count"],
                          training_log10_mae=fitted["training_log10_mae"],
                          curvature_at_floor=fitted["curvature_at_floor"]))
    full = fit_gaussian_source(inputs, outputs, **settings)
    return dict(fit=full, peak=source_peak(full), log10_predictions=predicted,
                folds=folds, settings=settings)


def leave_one_out_baselines(values):
    """Return mean and median forecasts fitted on each target's other official rows."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Supply at least two finite targets")
    return {name: np.asarray([statistic(np.delete(values, held)) for held in range(len(values))])
            for name, statistic in (("mean", np.mean), ("median", np.median))}

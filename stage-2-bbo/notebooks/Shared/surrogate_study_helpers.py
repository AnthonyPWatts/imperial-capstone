"""Bounded surrogate comparisons and input backpropagation for weekly studies."""

from collections import Counter
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, mean_absolute_error, precision_score,
                             recall_score, roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from model_review_helpers import build_review_candidates


def make_study_models(n_inputs, *, architectures, seeds, alpha=.1, max_iter=400,
                      max_fun=15000, tolerance=1e-6, gp_restarts=3, gp_feature_sets=None):
    """Declare the entire comparison before validation; no adaptive tuning occurs."""
    if n_inputs < 1 or not seeds or not architectures or len(set(seeds)) != len(seeds):
        raise ValueError("Require input dimensions, distinct seeds and architectures")

    def scaled(estimator):
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])

    def regression(estimator):
        return TransformedTargetRegressor(regressor=scaled(estimator), transformer=StandardScaler())

    gp = GaussianProcessRegressor(
        kernel=ConstantKernel(1., (.01, 100.)) * RBF(.3, (.03, 2.)),
        alpha=.1 ** 2, normalize_y=True, n_restarts_optimizer=gp_restarts, random_state=42,
    )
    regressors = {
        "Mean baseline": DummyRegressor(),
        "Linear regression": regression(LinearRegression()),
        "Ridge": regression(Ridge(alpha=1.)),
        "Kernel ridge": regression(KernelRidge(alpha=.1, kernel="rbf", gamma=1/n_inputs)),
        "RBF SVR": regression(SVR(C=1., epsilon=.1, kernel="rbf", gamma="scale")),
        "GP shared length scale": gp,
        "GP separate length scales": clone(gp).set_params(
            kernel=ConstantKernel(1., (.01, 100.)) * RBF([.3] * n_inputs, (.03, 2.))),
    }
    # Retain known projected hypotheses without reducing the inputs to other methods.
    if gp_feature_sets:
        from sklearn.compose import ColumnTransformer
        for name, axes in gp_feature_sets.items():
            if not axes or any(axis not in range(n_inputs) for axis in axes):
                raise ValueError("Projected GP axes must be valid")
            regressors[name] = Pipeline([
                ("coordinates", ColumnTransformer([("inputs", "passthrough", axes)])),
                ("model", clone(gp)),
            ])
    classifiers = {
        "Class-prior baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": scaled(LogisticRegression(C=1., max_iter=1000, random_state=42)),
        "Linear SVM": scaled(SVC(C=1., kernel="linear")),
        "RBF SVM": scaled(SVC(C=1., kernel="rbf", gamma="scale")),
    }
    metadata = {}
    for specification in architectures:
        for seed in seeds:
            name = f"NN {specification['name']} seed {seed}"
            settings = dict(hidden_layer_sizes=tuple(specification["layers"]),
                            activation=specification["activation"], solver="lbfgs", alpha=alpha,
                            max_iter=max_iter, max_fun=max_fun, tol=tolerance,
                            random_state=seed, early_stopping=False)
            regressors[name] = regression(MLPRegressor(**settings))
            classifiers[name] = scaled(MLPClassifier(**settings))
            metadata[name] = {"architecture": specification["name"], "seed": seed,
                              "layers": list(specification["layers"]), "activation": specification["activation"]}
    return regressors, classifiers, metadata


def training_threshold(target, quantile=.75):
    """Define good relative to the training history, without consulting held-out y."""
    target = np.asarray(target, dtype=float)
    if target.ndim != 1 or len(target) < 3 or not np.isfinite(target).all() or not 0 < quantile < 1:
        raise ValueError("Require finite training targets and a quantile inside (0, 1)")
    threshold = float(np.quantile(target, quantile))
    if len(np.unique(target >= threshold)) != 2:
        raise ValueError("The declared threshold must produce both training classes")
    return threshold


def _fit(estimator, inputs, target, warning_counts):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = clone(estimator).fit(inputs, target)
    for warning in caught:
        warning_counts[f"{warning.category.__name__}: {warning.message}"] += 1
    return fitted


def evaluate_regressors(models, inputs, target, *, verbose=True):
    """Fit complete pipelines per held-out row; keep all seeds and fitting warnings."""
    inputs, target = np.asarray(inputs, float), np.asarray(target, float)
    rows, predictions, fitted, warning_details = [], {}, {}, {}
    for name, estimator in models.items():
        counts, predicted = Counter(), np.empty(len(target))
        for train, test in LeaveOneOut().split(inputs):
            fold = _fit(estimator, inputs[train], target[train], counts)
            predicted[test] = fold.predict(inputs[test])
        full = _fit(estimator, inputs, target, counts)
        training = full.predict(inputs)
        if not np.isfinite(predicted).all() or not np.isfinite(training).all():
            raise ValueError(f"Non-finite predictions from {name}")
        rank = float(spearmanr(target, predicted).statistic) if np.ptp(predicted) else np.nan
        rows.append({"Model": name, "LOO RMSE": root_mean_squared_error(target, predicted),
                     "LOO MAE": mean_absolute_error(target, predicted), "Rank correlation": rank,
                     "Training RMSE": root_mean_squared_error(target, training),
                     "Fit warnings": sum(counts.values())})
        predictions[name], fitted[name], warning_details[name] = predicted, full, dict(counts)
        if verbose:
            print(f"Regression complete: {name}; warnings={sum(counts.values())}", flush=True)
    table = pd.DataFrame(rows).sort_values("LOO RMSE").reset_index(drop=True)
    baseline = float(table.set_index("Model").loc["Mean baseline", "LOO RMSE"])
    table["RMSE / baseline"] = table["LOO RMSE"] / baseline if baseline else np.nan
    return dict(table=table, predictions=predictions, fitted=fitted, warnings=warning_details)


def _classifier_scores(model, inputs):
    if hasattr(model, "predict_proba"):
        positive = int(np.flatnonzero(model.classes_ == 1)[0])
        return model.predict_proba(inputs)[:, positive], True
    return np.asarray(model.decision_function(inputs)), False


def evaluate_classifiers(models, inputs, target, *, quantile=.75, verbose=True):
    """LOO thresholds and scaling use only each fold's training observations.

    Held-out labels mean 'at least the upper-quartile training score'. Thresholds
    consequently vary by fold. Full-fit plots use the separately saved full-data
    threshold; they are not an independently known black-box boundary.
    """
    inputs, target = np.asarray(inputs, float), np.asarray(target, float)
    thresholds, truth = np.empty(len(target)), np.empty(len(target), dtype=int)
    folds = list(LeaveOneOut().split(inputs))
    for train, test in folds:
        thresholds[test] = training_threshold(target[train], quantile)
        truth[test] = target[test] >= thresholds[test]
    full_threshold = training_threshold(target, quantile)
    full_labels = (target >= full_threshold).astype(int)
    rows, predictions, scores, fitted, warning_details = [], {}, {}, {}, {}
    for name, estimator in models.items():
        counts, predicted, values = Counter(), np.empty(len(target), dtype=int), np.empty(len(target))
        for train, test in folds:
            labels = (target[train] >= thresholds[test][0]).astype(int)
            fold = _fit(estimator, inputs[train], labels, counts)
            predicted[test] = fold.predict(inputs[test])
            fold_scores, probabilities = _classifier_scores(fold, inputs[test])
            values[test] = fold_scores
        full = _fit(estimator, inputs, full_labels, counts)
        training = full.predict(inputs)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite classification scores from {name}")
        rows.append({"Model": name, "LOO balanced accuracy": balanced_accuracy_score(truth, predicted),
                     "LOO accuracy": accuracy_score(truth, predicted),
                     "Good precision": precision_score(truth, predicted, zero_division=0),
                     "Good recall": recall_score(truth, predicted, zero_division=0),
                     "Good F1": f1_score(truth, predicted, zero_division=0),
                     "ROC AUC": roc_auc_score(truth, values) if len(np.unique(truth)) == 2 else np.nan,
                     "Brier score": brier_score_loss(truth, values) if probabilities else np.nan,
                     "Log loss": log_loss(truth, np.clip(values, 1e-12, 1-1e-12), labels=[0, 1]) if probabilities else np.nan,
                     "Training balanced accuracy": balanced_accuracy_score(full_labels, training),
                     "False negatives": int(((truth == 1) & (predicted == 0)).sum()),
                     "False positives": int(((truth == 0) & (predicted == 1)).sum()),
                     "Fit warnings": sum(counts.values())})
        predictions[name], scores[name], fitted[name], warning_details[name] = predicted, values, full, dict(counts)
        if verbose:
            print(f"Classification complete: {name}; warnings={sum(counts.values())}", flush=True)
    table = pd.DataFrame(rows).sort_values(["LOO balanced accuracy", "Good recall"], ascending=False).reset_index(drop=True)
    return dict(table=table, predictions=predictions, scores=scores, fitted=fitted,
                warnings=warning_details, truth=truth, fold_thresholds=thresholds,
                threshold=full_threshold, labels=full_labels, quantile=quantile)


def summarise_network_seeds(run, metadata, *, task, representative_seed=42):
    """Select architecture by average validation evidence, never the best seed."""
    table = run["table"].set_index("Model")
    rows = []
    for name, settings in metadata.items():
        values = table.loc[name]
        rows.append({"Model": name, **settings, **values.to_dict()})
    networks = pd.DataFrame(rows)
    metric = "LOO RMSE" if task == "regression" else "LOO balanced accuracy"
    grouped = networks.groupby("architecture")[metric].agg(["mean", "std", "min", "max"]).reset_index()
    grouped = grouped.sort_values(["mean", "architecture"], ascending=[task == "regression", True]).reset_index(drop=True)
    architecture = grouped.iloc[0].architecture
    names = [name for name, spec in metadata.items() if spec["architecture"] == architecture]
    representative = next((name for name in names if metadata[name]["seed"] == representative_seed), None)
    if representative is None:
        raise ValueError("The representative seed must be declared before fitting")
    return dict(table=grouped, metric=metric, architecture=architecture,
                names=names, representative=representative)


def network_input_gradient(model, inputs):
    """Backpropagate a scalar regression output or P(good) to original input units.

    Uses fitted public coefs_/intercepts_. ReLU uses its zero subgradient at a
    kink. Regression target scaling and input scaling are both differentiated.
    """
    inputs = np.atleast_2d(np.asarray(inputs, float))
    target_scale = 1.
    if isinstance(model, TransformedTargetRegressor):
        target_scale = float(model.transformer_.scale_[0])
        model = model.regressor_
    scaler, network = model.named_steps["scale"], model.named_steps["model"]
    if not isinstance(network, (MLPRegressor, MLPClassifier)):
        raise ValueError("Supply a fitted scalar MLP pipeline")
    activation = scaler.transform(inputs)
    hidden = []
    for weights, bias in zip(network.coefs_[:-1], network.intercepts_[:-1]):
        linear = activation @ weights + bias
        if network.activation == "tanh":
            activation = np.tanh(linear)
        elif network.activation == "relu":
            activation = np.maximum(linear, 0)
        else:
            raise ValueError("This study declares only tanh and ReLU hidden activations")
        hidden.append(activation)
    output = activation @ network.coefs_[-1] + network.intercepts_[-1]
    if network.out_activation_ == "logistic":
        probability = expit(output)
        gradient = probability * (1-probability)
    elif network.out_activation_ == "identity":
        gradient = np.ones_like(output)
    else:
        raise ValueError("Require scalar regression or binary classification")
    for layer in range(len(network.coefs_)-1, -1, -1):
        gradient = gradient @ network.coefs_[layer].T
        if layer:
            values = hidden[layer-1]
            gradient *= (1-values**2) if network.activation == "tanh" else (values > 0)
    return gradient / scaler.scale_ * target_scale


def finite_difference_gradient(predict, inputs, *, step=1e-5, bounded=True):
    """Check input gradients, using one-sided differences at domain boundaries."""
    inputs = np.atleast_2d(np.asarray(inputs, float))
    result = np.empty_like(inputs)
    for axis in range(inputs.shape[1]):
        lower, upper = inputs.copy(), inputs.copy()
        lower[:, axis] -= step
        upper[:, axis] += step
        if bounded:
            lower[:, axis] = np.maximum(0, lower[:, axis])
            upper[:, axis] = np.minimum(.999999, upper[:, axis])
        result[:, axis] = (np.asarray(predict(upper))-np.asarray(predict(lower))) / (upper[:, axis]-lower[:, axis])
    return result


def bounded_gradient_path(model, start, *, direction=1, steps=8, step_size=.05):
    """Demonstrate ascent/descent of the fitted model without course evaluations."""
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 for ascent or -1 for descent")
    point = np.clip(np.asarray(start, float), 0, .999999)
    rows = []
    for iteration in range(steps+1):
        value = float(model.predict(point[None])[0])
        rows.append({"step": iteration, "prediction": value, **{f"x{i+1}": x for i, x in enumerate(point)}})
        gradient = network_input_gradient(model, point)[0] * direction
        norm = np.linalg.norm(gradient)
        if norm < 1e-10 or iteration == steps:
            break
        for fraction in (1., .5, .25):
            candidate = np.clip(point + fraction * step_size * gradient/norm, 0, .999999).round(6)
            proposed = float(model.predict(candidate[None])[0])
            if direction * (proposed-value) > 1e-10:
                point = candidate
                break
        else:
            break
    return pd.DataFrame(rows)


def common_study_candidates(inputs, anchor, *, broad_power=12, local_power=10, local_radius=.15):
    columns = [f"x{i+1}" for i in range(inputs.shape[1])]
    return build_review_candidates(inputs, anchor, np.empty((0, inputs.shape[1])),
        input_columns=columns, broad_power=broad_power, local_power=local_power,
        broad_seed=42, local_seed=43, local_radius=local_radius)


def svm_support_review(model, data, *, input_columns, labels, threshold, maximum=8):
    """Identify actual full-fit support vectors; kernel scores are not distances."""
    svm = model.named_steps["model"]
    if not isinstance(svm, SVC):
        raise ValueError("Supply a fitted SVC pipeline")
    support = data.iloc[svm.support_].copy()
    support["Good class"] = np.asarray(labels)[svm.support_]
    support["Decision score"] = model.decision_function(support[list(input_columns)].to_numpy())
    support["Threshold"] = threshold
    order = support["Decision score"].abs().sort_values().index
    return support.loc[order].head(maximum).reset_index(drop=True), len(svm.support_)


def interpret_study(regression, classification, reg_networks, class_networks, inputs,
                    target, data, submitted_point, *, input_columns,
                    candidate_settings=None, gradient_step=1e-5, path_steps=8, path_step_size=.05):
    """Collect gradient, support-vector and hypothetical-query evidence without I/O."""
    inputs, target = np.asarray(inputs, float), np.asarray(target, float)
    anchor = inputs[int(np.argmax(target))]
    reg_name, class_name = reg_networks["representative"], class_networks["representative"]
    reg_nn, class_nn = regression["fitted"][reg_name], classification["fitted"][class_name]
    gp_name = regression["table"].loc[regression["table"].Model.str.startswith("GP "), "Model"].iloc[0]
    gp = regression["fitted"][gp_name]
    candidates = common_study_candidates(inputs, anchor, **(candidate_settings or {}))
    points = candidates[list(input_columns)].to_numpy(dtype=float)
    probabilities = class_nn.predict_proba(points)[:, 1]
    boundary_point = points[int(np.argmin(np.abs(probabilities-.5)))]
    boundary_gradient = network_input_gradient(class_nn, boundary_point)[0]
    varied_axes = tuple(np.argsort(-np.abs(boundary_gradient), kind="stable")[:2])
    local_gradients = {
        "NN backpropagation": network_input_gradient(reg_nn, anchor)[0],
        "GP finite difference": finite_difference_gradient(gp.predict, anchor, step=gradient_step)[0],
        "SVR finite difference": finite_difference_gradient(regression["fitted"]["RBF SVR"].predict, anchor, step=gradient_step)[0],
    }
    absolute_gradients = np.array([np.abs(network_input_gradient(regression["fitted"][name], inputs)).mean(axis=0)
                                   for name in reg_networks["names"]])
    gradient_table = pd.DataFrame({"Input": input_columns, **local_gradients,
        "NN sampled mean absolute gradient": absolute_gradients.mean(axis=0),
        "NN seed SD of mean absolute gradient": absolute_gradients.std(axis=0, ddof=1),
        "NN P(good) gradient near boundary": boundary_gradient})
    check_points = np.random.default_rng(47).uniform(.1, .9, (3, inputs.shape[1]))
    gradient_checks = []
    for label, model, predict in ((reg_name, reg_nn, reg_nn.predict),
            (class_name, class_nn, lambda values: class_nn.predict_proba(values)[:, 1])):
        analytical = network_input_gradient(model, check_points)
        numerical = finite_difference_gradient(predict, check_points, step=gradient_step)
        error = np.abs(analytical-numerical)
        normalised_error = float((error/(1+np.abs(numerical))).max())
        if normalised_error > .001:
            raise ValueError(f"Input backpropagation check failed for {label}: {normalised_error}")
        gradient_checks.append({"Model": label, "Maximum absolute error": float(error.max()),
                                "Maximum normalised error": normalised_error, "Passed": True})
    support, support_count = svm_support_review(classification["fitted"]["RBF SVM"], data,
        input_columns=input_columns, labels=classification["labels"], threshold=classification["threshold"])
    linear_support, linear_support_count = svm_support_review(classification["fitted"]["Linear SVM"], data,
        input_columns=input_columns, labels=classification["labels"], threshold=classification["threshold"])
    candidate_rows, submitted_rows = [], []
    for name in ("Linear regression", "RBF SVR", gp_name, reg_name):
        model = regression["fitted"][name]
        means = np.asarray(model.predict(points))
        winner = int(np.argmax(means))
        point = points[winner]
        candidate_rows.append({"Method": name + ": mean only", "Predicted score": float(means[winner]),
            "Input": "-".join(f"{x:.6f}" for x in point),
            "Nearest measured input": float(np.linalg.norm(inputs-point, axis=1).min())})
        submitted_rows.append({"Model": name, "Prediction at submitted input": float(model.predict(np.asarray(submitted_point)[None])[0])})
    seed_predictions = np.array([regression["fitted"][name].predict(points) for name in reg_networks["names"]])
    ensemble_mean, ensemble_spread = seed_predictions.mean(axis=0), seed_predictions.std(axis=0, ddof=1)
    winner = int(np.argmax(ensemble_mean))
    ensemble_point = points[winner]
    candidate_rows.append({"Method": "NN seed ensemble: mean only", "Predicted score": float(ensemble_mean[winner]),
        "Input": "-".join(f"{x:.6f}" for x in ensemble_point),
        "Nearest measured input": float(np.linalg.norm(inputs-ensemble_point, axis=1).min())})
    paths = {"Ascent (maximise surrogate)": bounded_gradient_path(reg_nn, anchor, direction=1, steps=path_steps, step_size=path_step_size),
             "Descent (minimise surrogate)": bounded_gradient_path(reg_nn, anchor, direction=-1, steps=path_steps, step_size=path_step_size)}
    parameters = []
    for name in reg_networks["names"]:
        network = regression["fitted"][name].regressor_.named_steps["model"]
        parameters.append({"Model": name, "Trainable parameters": sum(array.size for array in [*network.coefs_, *network.intercepts_]),
                           "Observations": len(target), "Final iterations": network.n_iter_})
    return dict(anchor=anchor, gp_name=gp_name, regression_nn=reg_name, classification_nn=class_name,
        local_gradients=local_gradients, gradient_table=gradient_table,
        gradient_checks=pd.DataFrame(gradient_checks), seed_gradient_mean=absolute_gradients.mean(axis=0),
        seed_gradient_sd=absolute_gradients.std(axis=0, ddof=1), boundary_point=boundary_point,
        boundary_probability=float(class_nn.predict_proba(boundary_point[None])[0, 1]),
        boundary_gradient=boundary_gradient, boundary_axes=varied_axes,
        support=support, support_count=support_count, linear_support=linear_support,
        linear_support_count=linear_support_count, candidate_count=len(points),
        candidate_review=pd.DataFrame(candidate_rows), submitted_predictions=pd.DataFrame(submitted_rows),
        nn_ensemble_seed_spread_at_mean_winner=float(ensemble_spread[winner]), paths=paths,
        parameter_counts=pd.DataFrame(parameters))

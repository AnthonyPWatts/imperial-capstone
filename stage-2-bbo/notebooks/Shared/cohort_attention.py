"""A fixed-feature tabular self-attention benchmark for the small cohort pools."""

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cohort_surrogate_study import grouped_official_folds


ATTENTION = "Frozen transformer + ridge"
ABLATION = "No-attention tokens + ridge"
SEEDS = (11, 42, 73)
WIDTH = 16
ALPHA = 1.0


def _normalise_tokens(tokens):
    centred = tokens - tokens.mean(axis=-1, keepdims=True)
    return centred / np.sqrt(np.mean(centred ** 2, axis=-1, keepdims=True) + 1e-6)


def _token_features(inputs, seed):
    """Return matched representations with and without cross-coordinate attention."""
    rng = np.random.default_rng(seed)
    shape = (inputs.shape[1], WIDTH)
    tokens = (2 * inputs[..., None] - 1) * rng.normal(size=shape) + rng.normal(size=shape)
    query, key, value, output, expand, contract = rng.normal(size=(6, WIDTH, WIDTH)) / np.sqrt(WIDTH)

    normalised = _normalise_tokens(tokens)
    weights = softmax((normalised @ query) @ (normalised @ key).swapaxes(-2, -1) / np.sqrt(WIDTH), axis=-1)
    attended = tokens + (weights @ (normalised @ value)) @ output

    def features(hidden):
        layer = _normalise_tokens(hidden) @ expand
        encoded = hidden + np.maximum(layer, 0) @ contract
        return encoded.reshape(len(inputs), -1)

    return {ATTENTION: features(attended), ABLATION: features(tokens)}


def compare_cohort_attention(review, peer_target, study):
    """Score the matched models on the cohort study's grouped official LOO folds."""
    columns = review["input_columns"]
    official = review["own_data"]
    own = official[columns].to_numpy(dtype=float)
    peers = review["data"].loc[review["peer_mask"], columns].to_numpy(dtype=float)
    target = np.asarray(study["official_targets"], dtype=float)
    peer_target = np.asarray(peer_target, dtype=float)
    if (target.shape != (len(own),) or peer_target.shape != (len(peers),)
            or study["official_ids"] != official["observation_id"].tolist()):
        raise ValueError("Attention targets must match the cohort study's official and peer rows")

    portal = np.vstack([own, peers]).round(6)
    target_all = np.r_[target, peer_target]
    later = official["source_round"].to_numpy() > 0
    designs = [_token_features(np.vstack([own, peers]), seed) for seed in SEEDS]
    scores = {name: np.full((len(SEEDS), len(own)), np.nan) for name in (ATTENTION, ABLATION)}
    withheld = []
    for train, test in grouped_official_folds(own, np.arange(len(own))):
        peer_matches = np.all(portal[len(own):, None] == portal[test], axis=-1).any(axis=1)
        allowed = np.r_[train, len(own) + np.flatnonzero(~peer_matches)]
        withheld.append(int(peer_matches.sum()))
        for seed_index, design in enumerate(designs):
            for name, features in design.items():
                readout = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA))
                readout.fit(features[allowed], target_all[allowed])
                scores[name][seed_index, test] = readout.predict(features[test])

    def evaluate(name, predicted):
        residual = target - predicted
        def error(values):
            return float(np.mean(np.abs(values)) if study["metric"] == "MAE"
                         else np.sqrt(np.mean(values ** 2)))
        return dict(method=name, error=error(residual), returned_error=error(residual[later]),
                    predictions=predicted.tolist())

    rows = [evaluate(study["reference_model"], np.asarray(study["loo_predictions"][study["reference_model"]]))]
    for name, values in scores.items():
        if not np.isfinite(values).all():
            raise ValueError("An official attention fold did not produce a finite prediction")
        rows.append(evaluate(name, values.mean(axis=0)))
    return dict(metric=study["metric"], seeds=list(SEEDS), width=WIDTH, ridge_alpha=ALPHA,
                official_targets=len(own), peer_training_rows=len(peers),
                matching_peer_rows_withheld=sum(withheld), results=rows)

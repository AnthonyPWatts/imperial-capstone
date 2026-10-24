"""Snapshot alignment and like-for-like cohort comparisons for any input count."""

import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from exploration_review_helpers import load_exploration_review
from model_helpers import score_query_candidates
from output_helpers import summarise_output_observations


def load_cohort_review(data_path, official_path, *, input_columns, output_column="y",
                       round_column="source_round", expected_round_counts=None,
                       expected_peer_count=None, peer_prefix="peer-", peer_round_max=None):
    """Require unchanged official records; align their positions by ID, not row order.

    Peer round labels belong to their sources. Only the official history determines
    the latest-return highlight; all retained outputs contribute to pooled ranks.
    """
    settings = dict(input_columns=input_columns, output_column=output_column, round_column=round_column)
    own = load_exploration_review(official_path, expected_round_counts=expected_round_counts, **settings)
    review = load_exploration_review(data_path, **settings)
    data, own_data = review["data"], own["data"]
    indexed = data.set_index("observation_id", drop=False)
    if not own_data.observation_id.isin(indexed.index).all():
        raise ValueError("The pooled snapshot is missing official observations")
    aligned = indexed.loc[own_data.observation_id].reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(aligned, own_data.reset_index(drop=True), check_exact=True)
    except AssertionError as error:
        raise ValueError("The pooled snapshot changed official observations") from error
    own_mask = data.observation_id.isin(own_data.observation_id).to_numpy()
    peer_mask = ~own_mask
    peers = data.loc[peer_mask]
    if not peers.observation_id.str.startswith(peer_prefix).all():
        raise ValueError("Unexpected non-official observation IDs")
    if expected_peer_count is not None and len(peers) != expected_peer_count:
        raise ValueError("Unexpected peer observation count")
    if peer_round_max is not None and (peers[round_column] > peer_round_max).any():
        raise ValueError("Peer rounds exceed the declared snapshot cutoff")
    # Neutralise per-source round labels only for the timeline summary, not stored data.
    timeline = data.copy(deep=True)
    timeline.loc[peer_mask, round_column] = 0
    summary = summarise_output_observations(timeline, **settings)
    review.update({key: summary[key] for key in
                   ("latest_round", "latest_mask", "prior_count", "latest_submissions", "report")})
    positions = indexed.index.get_indexer(own_data.observation_id)
    review.update(own_data=own_data, own_positions=positions, own_mask=own_mask, peer_mask=peer_mask)
    return review


def paired_target_comparison(own_run, pooled_run, own_target, *, own_positions, peer_count):
    """Compare identical official LOO targets with and without peer training rows."""
    own_target, own_positions = np.asarray(own_target), np.asarray(own_positions)
    if (own_positions.shape != own_target.shape or len(np.unique(own_positions)) != len(own_target)
            or not np.issubdtype(own_positions.dtype, np.integer) or (own_positions < 0).any()):
        raise ValueError("Supply one distinct pooled position per official target")
    count = len(own_target) - 1
    rows = []
    for name, predictions in own_run["predictions"].items():
        pooled = np.asarray(pooled_run["predictions"][name])[own_positions]
        rows.append({"Model": name,
            f"Own-target RMSE: {count} training rows": root_mean_squared_error(own_target, predictions),
            f"Own-target RMSE: {count} own + {peer_count} peer training rows": root_mean_squared_error(own_target, pooled),
            "Own-target MAE: own training": mean_absolute_error(own_target, predictions),
            "Own-target MAE: pooled training": mean_absolute_error(own_target, pooled)})
    return pd.DataFrame(rows)


def read_saved_proposal(notebook_path, *, function, week, model_run, observation_count):
    """Validate the saved official proposal against a fresh fit of its snapshot."""
    matches = []
    for cell in nbformat.read(Path(notebook_path), as_version=4).cells:
        for output in cell.get("outputs", []):
            try:
                value = json.loads(output.get("text", ""))
            except (ValueError, TypeError):
                continue
            if (isinstance(value, dict) and value.get("function") == function
                    and value.get("week") == week and "portal_input" in value):
                matches.append(value)
    if len(matches) != 1:
        raise ValueError("Require exactly one saved official proposal summary")
    summary = matches[0]
    if (summary["observations"] != observation_count
            or summary["query_model"] != model_run["selected_name"]
            or not np.isclose(summary["gp_rmse"], model_run["selected_rmse"], rtol=1e-10, atol=1e-12)):
        raise ValueError("Saved official summary does not match the refitted snapshot")
    return summary


def read_saved_analysis(notebook_path, *, function, week, analysis):
    """Read exactly one dated analysis summary, including multi-target studies."""
    matches = []
    for cell in nbformat.read(Path(notebook_path), as_version=4).cells:
        for output in cell.get("outputs", []):
            try:
                value = json.loads(output.get("text", ""))
            except (ValueError, TypeError):
                continue
            if (isinstance(value, dict) and value.get("function") == function
                    and value.get("week") == week and value.get("analysis") == analysis):
                matches.append(value)
    if len(matches) != 1:
        raise ValueError("Require exactly one saved matching analysis summary")
    return matches[0]


def common_candidate_comparison(fits, point_sets, observed, *, input_columns, weight=1.96, precision=6):
    """Score a frozen union of candidate sets using every declared fit."""
    points = np.unique(np.clip(np.vstack(point_sets).round(precision), 0, 1-10.**-precision), axis=0)
    keys = set(map(tuple, np.asarray(observed).round(precision)))
    points = points[[tuple(point) not in keys for point in points]]
    rows = []
    for label, run in fits.items():
        top = score_query_candidates(run["selected_model"], points, input_columns=input_columns,
                                     weight=weight, precision=precision).iloc[0]
        rows.append({"Fit": label, "Model": run["selected_name"], **top.to_dict(), "Candidates": len(points)})
    return pd.DataFrame(rows)


def compare_proposal_predictions(fits, proposals, *, input_columns, weight=1.96):
    """Apply each fit to the same actual proposals, retaining all coordinates."""
    rows = []
    for point_label, point in proposals.items():
        point = np.asarray(point)
        for fit_label, run in fits.items():
            mean, std = run["selected_model"].predict(point[None, :], return_std=True)
            rows.append({"Point": point_label, "Fit": fit_label, **dict(zip(input_columns, point)),
                         "Mean": mean[0], "SD": std[0], "UCB": mean[0] + weight * std[0]})
    return pd.DataFrame(rows)

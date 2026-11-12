"""Verify the agreed Week 7 inputs and reproduce the portal input file."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from review_week_06_queries import DIMENSIONS, ROOT, STAGE, fingerprint, point_record, score_points, validate_lines
from cohort_review_helpers import load_cohort_review
from exploration_review_helpers import load_exploration_review
from prediction_review_helpers import make_candidate_review


FOLDER = STAGE / "submissions/Week_07"
SUBMISSION_PATH = FOLDER / "submissions.txt"
FEEDBACK_PATH = STAGE / "submissions/round-06-results.json"


def source_entry(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=fingerprint(path))


def read_review(function):
    """Load the saved fit with its pinned official and retained peer snapshots."""
    week = STAGE / f"notebooks/Function_{function}/Week_07"
    folder = week / "official" if function <= 4 else week
    summary_path = folder / "model-review.json"
    official_path = folder / "observations.csv"
    explorer_path = folder / "figures/neural-network-explorer.html"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    columns = [f"x{i}" for i in range(1, DIMENSIONS[function - 1] + 1)]
    if (summary["function"], summary["week"], summary["snapshot_sha256"]) != (function, 7, fingerprint(official_path)):
        raise ValueError(f"Function {function} model and official snapshot differ")

    html = explorer_path.read_text(encoding="utf-8")
    match = re.search(r'<script id="model-data" type="application/json">(.*?)</script>', html, re.S)
    if match is None:
        raise ValueError(f"Function {function} has no saved explorer fit")
    fitted = json.loads(match[1])
    if (fitted["input_columns"] != columns or fitted["proposal"] != summary["point"]
            or fitted["selected_architecture"] != summary["selected_neural_architecture"]
            or fitted["scope"] != summary["training_scope"]):
        raise ValueError(f"Function {function} saved fit and model review differ")

    sources = [official_path, summary_path, folder / "model_predictions.ipynb", explorer_path]
    counts = {0: summary["official_observations"] - 6, **dict.fromkeys(range(1, 7), 1)}
    if function <= 4:
        pooled_path = week / "cohort/observations.csv"
        audit_path = week / "cohort/data-audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (fingerprint(pooled_path) != audit["cohort_csv_sha256"]
                or fingerprint(audit_path) != summary["cohort_audit_sha256"]):
            raise ValueError(f"Function {function} retained peer snapshot differs")
        pooled = load_cohort_review(pooled_path, official_path, input_columns=columns,
                                    expected_round_counts=counts,
                                    expected_peer_count=summary["retained_peer_observations"])
        data, known = pooled["own_data"], pooled["data"]
        sources.extend([pooled_path, audit_path])
    else:
        data = load_exploration_review(official_path, input_columns=columns, expected_round_counts=counts)["data"]
        known = data
    review = dict(summary=summary, fitted=fitted, data=data, known=known[columns].to_numpy(),
                  known_data=known, sources=[source_entry(path) for path in sources])
    score = score_points(review, summary["point"])
    np.testing.assert_allclose([score["gp_mean"][0], score["gp_sd"][0]],
                               [summary["proposal"]["predicted_output"], summary["proposal"]["posterior_std"]],
                               rtol=1e-10, atol=1e-10)
    return review


def function_8_local_choice(review):
    """Rank local candidates by GP mean after screening with both fitted means."""
    summary, fitted = review["summary"], review["fitted"]
    settings, columns = summary["search_settings"], fitted["input_columns"]
    anchor = np.asarray(fitted["incumbent"])
    pool, _ = make_candidate_review(review["known"], anchor, input_columns=columns, search_settings=settings,
                                    minimum_distance=summary["minimum_query_distance"])
    if len(pool) != summary["candidate_count"]:
        raise ValueError("Function 8 candidate pool differs from its saved review")
    points = pool[columns].to_numpy()
    radius = settings["local_radius"]
    local = points[np.max(np.abs(points - anchor), axis=1) <= radius + .5 * 10 ** -settings["precision"]]
    scores = score_points(review, local)
    best = summary["best_observed_raw_output"]
    eligible = np.flatnonzero((scores["gp_mean"] > best) & (scores["neural_mean"] > best))
    if not len(eligible):
        raise ValueError("The agreed Function 8 screen has no eligible candidates")
    index = eligible[np.argmax(scores["gp_mean"][eligible])]
    return local[index], dict(pool_rows=len(pool), local_radius=radius, local_rows=len(local),
                              both_means_above_incumbent=len(eligible))


def build_submission(record, reviews=None):
    """Validate the decision record and return the exact eight-line submission."""
    choices = record["candidates"]
    if (record["week"] != 7 or record["feedback_through_round"] != 6
            or record["agreed_functions"] != list(range(1, 9))
            or [row["function"] for row in choices] != list(range(1, 9))
            or any(row["decision_status"] != "agreed" for row in choices)):
        raise ValueError("Require the eight agreed Week 7 choices in function order")
    reviews = [read_review(f) for f in range(1, 9)] if reviews is None else reviews
    earlier = sorted(STAGE.glob("submissions/Week_0[1-6]/submissions.txt"))
    if len(earlier) != 6 or [source_entry(path) for path in earlier] != record["earlier_submissions"]:
        raise ValueError("Earlier submission file evidence differs")
    if source_entry(FEEDBACK_PATH) != record["feedback_source"]:
        raise ValueError("Round 6 feedback evidence differs")
    feedback = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))["results"]
    text, checks = validate_lines([row["point"] for row in choices], reviews, earlier)
    raw = text.encode("ascii")
    expected_file = dict(path=SUBMISSION_PATH.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(raw).hexdigest())
    if record["submission_file"] != expected_file or text.splitlines() != [row["portal_input"] for row in choices]:
        raise ValueError("Prepared bytes differ from the agreed inputs")

    sources = [record["selection_source"]]
    for choice, review, checked, result in zip(choices, reviews, checks, feedback, strict=True):
        number = choice["function"]
        summary, data = review["summary"], review["data"]
        latest = data.loc[data.source_round.eq(6)].iloc[0]
        columns = review["fitted"]["input_columns"]
        if (result["function"] != number or len(data) != result["observations"]
                or latest.y != result["output"] or latest[columns].tolist() != result["point"]):
            raise ValueError(f"Function {number} does not match the round 6 return")
        distance = checked["nearest_known_input_distance"]
        if distance < summary["minimum_query_distance"]:
            raise ValueError(f"Function {number} violates its minimum spacing")
        scored = point_record(review, choice["point"])
        if (choice["prediction_units"] != summary["target_units"]
                or choice["predictions"]["point"] != choice["point"]
                or choice["predictions"]["portal_input"] != choice["portal_input"]):
            raise ValueError(f"Function {number} forecast metadata differs")
        for key in ("gp_mean", "gp_latent_sd", "gp_ucb_1_96", "neural_mean", "neural_seed_sd", "neural_seed_predictions"):
            np.testing.assert_allclose(choice["predictions"][key], scored[key], rtol=1e-10, atol=1e-10)
        sources.extend(choice["sources"])
    for source in sources:
        if fingerprint(ROOT / source["path"]) != source["sha256"]:
            raise ValueError(f"Recorded source changed: {source['path']}")

    if sum(len(review["data"]) for review in reviews) != record["official_observations"]:
        raise ValueError("Official observation total differs")
    point, local_check = function_8_local_choice(reviews[7])
    if point.tolist() != choices[7]["point"] or local_check != choices[7]["evidence"]["local_screen"]:
        raise ValueError("Function 8 local screen does not reproduce the agreed point")
    return raw, checks


def prepare_submission():
    """Create a missing input file, or verify an existing byte-identical file."""
    record = json.loads((FOLDER / "proposals.json").read_text(encoding="utf-8"))
    raw, _ = build_submission(record)
    if not SUBMISSION_PATH.exists():
        with SUBMISSION_PATH.open("xb") as output:
            output.write(raw)
    if SUBMISSION_PATH.read_bytes() != raw:
        raise ValueError("The existing Week 7 file differs from the agreed submission")
    print(f"Verified eight Week 7 inputs. Status: {record['status']}.")


if __name__ == "__main__":
    prepare_submission()

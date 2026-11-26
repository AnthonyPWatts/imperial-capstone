
"""Compare local TabPFN-2 and the official-data GP on identical BBO folds."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from benchmark_plots import plot_surrogates
from candidate_ranking import rank_candidates


STAGE = Path(__file__).resolve().parents[2]
RUNTIME = STAGE.parent / ".runtime/tabpfn-surrogate"
CHECKPOINT = RUNTIME / "tabpfn-v2-regressor.ckpt"
CHECKPOINT_SHA = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
QUANTILES = [.05, .5, .95]
NORMAL_90 = 1.64485362695147

sys.path[:0] = [
    str(STAGE / "notebooks/Shared"),
    str(STAGE / "notebooks/Function_1/FunctionSpecificShared"),
]
from cohort_advisory_bo import SETTINGS, fit_owner, owner_posterior  # noqa: E402
from plot_helpers import _sampling_gap_surface  # noqa: E402
from weekly_peak_review import PORTAL_MAX  # noqa: E402


def load_official(function):
    path = STAGE / f"data/accumulated/function_{function}/observations.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    axes = [f"x{i + 1}" for i in range(len(SETTINGS[function][0]))]
    x, y = frame[axes].to_numpy(), frame.y.to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((x < 0) | (x >= 1)):
        raise ValueError(f"Function {function} has invalid official observations")
    rounds = frame["source_round"].to_numpy()
    later = sorted(rounds[rounds > 0].tolist())
    if not later or later != list(range(1, len(later) + 1)):
        raise ValueError(f"Function {function} must have exactly one feedback row for every completed round")
    for r in later:
        feedback = json.loads((STAGE / f"submissions/round-{r:02d}-results.json").read_bytes())
        record = next(row for row in feedback["results"] if row["function"] == function)
        i = np.flatnonzero(rounds == r)[0]
        if (r != feedback["round"] or not np.array_equal(x[i], record["point"])
                or y[i] != record["output"]):
            raise ValueError(f"Function {function}, round {r} disagrees with the official feedback")
    return frame, axes, sha256(path.read_bytes()).hexdigest()


def exclude_inputs(function, last_round):
    folder = STAGE / "submissions"
    paths = [folder / f"Week_{r:02d}/submissions.txt" for r in range(1, last_round + 1)]
    next_folder = folder / f"Week_{last_round + 1:02d}"
    paths.extend(path for path in (next_folder / "submissions.txt",
                                   next_folder / "official-data-submissions.txt") if path.is_file())
    return np.array([[float(value) for value in path.read_text(encoding="utf-8-sig").splitlines()
                      [function - 1].split("-")] for path in paths])


def disable_network():
    # Explicit local checkpoint plus a fail-closed guard against Python sockets.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TABPFN_NO_BROWSER"] = "1"

    def reject_network(event, _):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.sendto"}:
            raise RuntimeError(f"Network access is disabled for this benchmark ({event})")

    sys.addaudithook(reject_network)


def fit_tabpfn(x, y, args):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, model_path=args.checkpoint, device="cpu", fit_mode="fit_preprocessors",
        n_estimators=args.estimators, random_state=args.seed,
    ).fit(x, y)


def gp_observation_prediction(model, points):
    mean, latent_sd = owner_posterior(model, points)
    gp, _, scale, _ = model[0]
    noise_sd = scale * math.sqrt(gp.kernel_.k2.noise_level)
    return mean, np.hypot(latent_sd, noise_sd)


def tabpfn_prediction(model, points, batch_size):
    return np.concatenate([model.predict(points[start:start + batch_size], output_type="mean")
                           for start in range(0, len(points), batch_size)])


def fold_predictions(function, frame, columns, args):
    x, y = frame[columns].to_numpy(), frame.y.to_numpy()
    rounds = frame.source_round.to_numpy()
    keys = x.round(6)
    rows = []
    for protocol, held in (
        ("Leave-one-coordinate-out", np.arange(len(y))),
        ("Sequential replay", np.flatnonzero(rounds > 0)),
    ):
        for i in held:
            # The replay uses all starters and only rounds completed before this return.
            train = np.any(keys != keys[i], axis=1) if protocol.startswith("Leave") else rounds < rounds[i]
            query = x[i:i + 1]
            gp = fit_owner(function, x[train], y[train])
            pf = fit_tabpfn(x[train], y[train], args)
            mean, sd = gp_observation_prediction(gp, query)
            full = pf.predict(query, output_type="full", quantiles=QUANTILES)
            q = np.asarray(full["quantiles"], float)[:, 0]
            if not np.isfinite(np.r_[mean, sd, q, full["mean"]]).all() or np.any(np.diff(q) < 0):
                raise ValueError(f"Invalid predictions on {protocol}, function {function}, row {i}")
            rows.append({
                "function": function, "protocol": protocol, "observation_id": frame.iloc[i]["observation_id"],
                "source_round": int(rounds[i]), "training_rows": int(train.sum()), "observed": y[i],
                "gp_mean": mean[0], "gp_q05_normal": mean[0] - NORMAL_90 * sd[0],
                "gp_q95_normal": mean[0] + NORMAL_90 * sd[0],
                "tabpfn_mean": float(full["mean"][0]), "tabpfn_q05": q[0], "tabpfn_q50": q[1], "tabpfn_q95": q[2],
            })
    return pd.DataFrame(rows)


def metrics(actual, predicted, low, high):
    k = math.ceil(len(actual) / 4)
    realised_top = np.argsort(-actual, kind="stable")[:k]
    predicted_top = np.argsort(-predicted, kind="stable")[:k]
    residual = predicted - actual
    rank = spearmanr(actual, predicted).statistic if np.ptp(predicted) and np.ptp(actual) else None
    return {
        "n": len(actual), "top_k": k,
        "rmse": float(np.sqrt(np.mean(residual**2))), "mae": float(np.mean(np.abs(residual))),
        "spearman": float(rank) if rank is not None else None,
        "top_quartile_recall": float(np.intersect1d(realised_top, predicted_top).size / k),
        "interval90_coverage": float(np.mean((actual >= low) & (actual <= high))),
        "interval90_mean_width": float(np.mean(high - low)),
    }


def validation_metrics(predictions):
    report = {}
    for protocol, fold in predictions.groupby("protocol", sort=False):
        y = fold.observed.to_numpy()
        report[protocol] = {
            "GP": metrics(y, *[fold[column].to_numpy() for column in ("gp_mean", "gp_q05_normal", "gp_q95_normal")]),
            "TabPFN": metrics(y, *[fold[column].to_numpy() for column in ("tabpfn_mean", "tabpfn_q05", "tabpfn_q95")]),
        }
    return report


def plot_grids(function, frame, axes, gp, tab, args):
    x, y = frame[axes].to_numpy(), frame.y.to_numpy()
    anchor = x[y.argmax()]
    grids = []
    if function == 1:
        _, edges, _ = _sampling_gap_surface(frame, axes)
        centres = (edges[:-1] + edges[1:]) / 2
        grids.append(((0, 1), centres, centres, "coverage-grid"))
        zoom = [np.linspace(max(0, v - .007), min(PORTAL_MAX, v + .007), 61) for v in anchor]
        grids.append(((0, 1), *zoom, "peak-zoom"))
    else:
        values = np.linspace(0, PORTAL_MAX, 41)
        for i, j in SETTINGS[function][4]:
            grids.append(((i, j), values, values, f"x{i + 1}-x{j + 1}"))

    paths = []
    for pair, xs, ys, name in grids:
        a, b = np.meshgrid(xs, ys)
        points = np.broadcast_to(anchor, (a.size, len(anchor))).copy()
        points[:, pair] = np.column_stack([a.ravel(), b.ravel()])
        gp_mean, _ = owner_posterior(gp, points)
        tab_mean = tabpfn_prediction(tab, points, args.batch_size)
        destination = args.out_dir / f"f{function}-{name}.png"
        plot_surrogates(gp_mean, tab_mean, pair, points, x, y, function, destination, anchor=anchor,
                        score_label="Raw BBO score")
        paths.append(destination.name)
    return paths


def benchmark(function, args):
    start = perf_counter()
    frame, axes, fingerprint = load_official(function)
    x, y = frame[axes].to_numpy(), frame.y.to_numpy()
    last_round = int(frame.source_round.max())
    folds = fold_predictions(function, frame, axes, args)
    validation = validation_metrics(folds)
    for protocol, summary in validation.items():
        print(f"F{function} {protocol}: GP RMSE {summary['GP']['rmse']:.6g} · "
              f"TabPFN RMSE {summary['TabPFN']['rmse']:.6g}", flush=True)
    folds.to_csv(args.out_dir / f"f{function}-validation.csv", index=False)

    owner = fit_owner(function, x, y)
    tab = fit_tabpfn(x, y, args)
    candidates = rank_candidates(
        function, x, y, lambda points: gp_observation_prediction(owner, points),
        lambda points: tabpfn_prediction(tab, points, args.batch_size),
        exclude_inputs(function, last_round), seed=args.seed, pool_power=args.pool_power,
        minimum_distance=SETTINGS[function][3], batch_size=args.batch_size,
    )
    candidates.to_csv(args.out_dir / f"f{function}-candidates.csv", index=False)
    first = {criterion: candidates.loc[candidates[criterion].idxmax()]
             for criterion in ("GP EI", "TabPFN mean", "Nearest official distance",
                               "Absolute mean disagreement", "Combined diagnostic")}
    candidate_summary = {name: {
        "input": row["Portal input"], "gp_mean": float(row["GP mean"]),
        "tabpfn_mean": float(row["TabPFN mean"]), "gp_ei": float(row["GP EI"]),
        "minimum_distance": float(row["Nearest official distance"]),
        "absolute_disagreement": float(row["Absolute mean disagreement"]),
    } for name, row in first.items()}
    plot_paths = plot_grids(function, frame, axes, owner, tab, args)
    report = {
        "function": function, "snapshot_sha256": fingerprint, "observations": len(frame),
        "last_round": last_round, "incumbent": float(np.max(y)), "validation": validation,
        "candidate_count": len(candidates), "candidate_leaders": candidate_summary,
        "candidate_weights": candidates.attrs.get("combined_diagnostic_weights"), "figures": plot_paths,
        "elapsed_seconds": round(perf_counter() - start, 2),
    }
    (args.out_dir / f"f{function}-summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"F{function} completed with {len(candidates)} eligible candidates and {len(plot_paths)} figures "
          f"in {report['elapsed_seconds']:.1f}s.", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functions", type=int, nargs="+", choices=range(1, 9), default=list(range(1, 9)))
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=RUNTIME / "results")
    parser.add_argument("--seed", type=int, default=907)
    parser.add_argument("--estimators", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--pool-power", type=int, default=10, choices=range(8, 15))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.estimators, args.batch_size, args.threads) < 1:
        parser.error("estimators, batch-size and threads must be positive")
    args.checkpoint = args.checkpoint.resolve()
    args.out_dir = args.out_dir.resolve()
    if not args.checkpoint.is_file() or sha256(args.checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA:
        parser.error("the checkpoint must match the pinned public TabPFN-2 regressor hash")
    return args


def main():
    args = parse_args()
    disable_network()
    import torch

    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "python": sys.version.split()[0],
        "packages": {name: version(name) for name in ("tabpfn", "torch", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib")},
        "checkpoint_sha256": CHECKPOINT_SHA, "device": "cpu", "options": {
            name: getattr(args, name) for name in ("seed", "estimators", "batch_size", "pool_power", "threads")
        },
    }
    with threadpool_limits(limits=1, user_api="blas"):
        results = [benchmark(function, args) for function in dict.fromkeys(args.functions)]
    # Scope is explicit so an F1 trial and an all-function run can coexist when
    # they use different output directories. Per-function files allow readback
    # after an interrupted run.
    (args.out_dir / "summary.json").write_text(json.dumps({
        "runtime": metadata, "functions": results,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

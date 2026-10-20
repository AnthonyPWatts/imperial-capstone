"""Run and cache geometric coverage searches for the weekly notebooks."""

import hashlib
import inspect
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import warnings
from zipfile import BadZipFile

import numpy as np
import scipy
from scipy.spatial.distance import cdist

import coverage_helpers


__all__ = [
    "cached_coverage_search",
    "compare_coverage_batches",
    "refine_coverage_batch",
]


def _array_fingerprint(values):
    digest = hashlib.sha256()
    digest.update(json.dumps(values.shape).encode("utf-8"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _source_fingerprint(function):
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _assess_coverage(existing, proposed, evaluation_space):
    design = np.vstack([existing, proposed])
    distances = cdist(evaluation_space, design).min(axis=1)
    return float(distances.max()), float(np.percentile(distances, 95))


def cached_coverage_search(
    existing, q, optimization_space, evaluation_space, *, cache_dir, seed=42
):
    """Load a matching search or optimise, assess and save it immediately.

    Returns q, proposed coordinates, max_distance, p95_distance and from_cache.
    The cache depends on input locations, both assessment spaces, q, seed,
    bounds, optimiser/assessment code and NumPy/SciPy versions. Output values
    y are not used. The optimiser's settings and return values are unchanged.

    Each completed batch is written atomically to its own .npz file. An
    interrupted search leaves earlier batches available for the next run.
    Invalid cache files produce a warning and are recomputed.
    """
    if not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError("q must be a non-negative integer.")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("A non-negative integer seed is required for caching.")
    q, seed = int(q), int(seed)

    # Canonical float64 inputs also allow equivalent DataFrames/arrays to match.
    existing, optimization_space, evaluation_space = (
        np.asarray(values, dtype="<f8")
        for values in (existing, optimization_space, evaluation_space)
    )
    arrays = (existing, optimization_space, evaluation_space)
    if any(values.ndim != 2 or 0 in values.shape for values in arrays):
        raise ValueError("Inputs and assessment spaces must be non-empty 2D arrays.")
    n_dimensions = existing.shape[1]
    if any(values.shape[1] != n_dimensions for values in arrays):
        raise ValueError("Inputs and assessment spaces must have matching dimensions.")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Inputs and assessment spaces must contain finite values.")

    metadata = {
        "cache_version": 1,
        "existing": _array_fingerprint(existing),
        "optimization_space": _array_fingerprint(optimization_space),
        "evaluation_space": _array_fingerprint(evaluation_space),
        "q": q,
        "seed": seed,
        "max_value": coverage_helpers.MAX_VALUE,
        "optimiser": _source_fingerprint(coverage_helpers.optimise_joint_minimax),
        "assessment": _source_fingerprint(_assess_coverage),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    metadata_json = json.dumps(metadata, sort_keys=True)
    key = hashlib.sha256(metadata_json.encode("utf-8")).hexdigest()
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"q-{q:02d}-seed-{seed}-{key}.npz"

    if cache_file.is_file():
        try:
            with np.load(cache_file, allow_pickle=False) as saved:
                if saved["metadata"].item() != metadata_json:
                    raise ValueError("Cache metadata does not match this search.")
                proposed = saved["proposed"]
                max_distance = float(saved["max_distance"].item())
                p95_distance = float(saved["p95_distance"].item())
                if (
                    proposed.shape != (q, n_dimensions)
                    or not np.isfinite(proposed).all()
                    or np.any(proposed < 0.0)
                    or np.any(proposed > coverage_helpers.MAX_VALUE)
                    or not np.isfinite([max_distance, p95_distance]).all()
                    or not 0.0 <= p95_distance <= max_distance
                ):
                    raise ValueError("Cache contains invalid coordinates or scores.")
            return {
                "q": q,
                "proposed": proposed,
                "max_distance": max_distance,
                "p95_distance": p95_distance,
                "from_cache": True,
            }
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
            warnings.warn(
                f"Cannot reuse {cache_file.name}: {error}. Recomputing this batch.",
                RuntimeWarning,
                stacklevel=2,
            )

    if q == 0:
        proposed = np.empty((0, n_dimensions))
    else:
        proposed = coverage_helpers.optimise_joint_minimax(
            existing=existing,
            q=q,
            coverage_space=optimization_space,
            seed=seed,
        )
    max_distance, p95_distance = _assess_coverage(
        existing, proposed, evaluation_space
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=cache_dir, suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(
                temporary,
                metadata=np.array(metadata_json),
                proposed=proposed,
                max_distance=max_distance,
                p95_distance=p95_distance,
            )
        temporary_path.replace(cache_file)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return {
        "q": q,
        "proposed": proposed,
        "max_distance": max_distance,
        "p95_distance": p95_distance,
        "from_cache": False,
    }


def compare_coverage_batches(
    existing,
    *,
    cache_dir,
    max_queries,
    optimization_power=12,
    evaluation_power=15,
    optimization_seed=42,
    evaluation_seed=43,
    comparison_seed=42,
    verbose=True,
):
    """Compare batches of zero through max_queries additional input locations.

    Generate a Sobol optimisation space and a larger, separately seeded
    evaluation space. Defaults match the weekly exploration notebook:
    4,096 optimisation points and 32,768 evaluation points, plus corners.
    The input dimension comes from existing; output values are not required.

    Every completed batch is saved by cached_coverage_search, so matching
    reruns reuse it and interrupted sweeps can resume. Print progress unless
    verbose=False. This function does not change the optimiser's settings.

    Return a dictionary containing results (score rows), proposed_batches
    (coordinates indexed by q), both assessment spaces and comparison_seed.
    These can be reused for elbow selection, plots and batch refinement.
    """
    if not isinstance(max_queries, (int, np.integer)) or max_queries < 0:
        raise ValueError("max_queries must be a non-negative integer.")
    existing = np.asarray(existing, dtype="<f8")
    if existing.ndim != 2 or 0 in existing.shape:
        raise ValueError("existing must be a non-empty 2D array of input locations.")

    optimization_space = coverage_helpers.make_coverage_space(
        n_dimensions=existing.shape[1],
        power=optimization_power,
        seed=optimization_seed,
    )
    coverage_evaluation_space = coverage_helpers.make_coverage_space(
        n_dimensions=existing.shape[1],
        power=evaluation_power,
        seed=evaluation_seed,
    )

    results = []
    proposed_batches = {}
    for q in range(max_queries + 1):
        batch = cached_coverage_search(
            existing=existing,
            q=q,
            optimization_space=optimization_space,
            evaluation_space=coverage_evaluation_space,
            cache_dir=cache_dir,
            seed=comparison_seed,
        )
        proposed_batches[q] = batch["proposed"]
        results.append({
            "q": q,
            "max_distance": batch["max_distance"],
            "p95_distance": batch["p95_distance"],
        })
        if verbose:
            status = "loaded from cache" if batch["from_cache"] else "computed and saved"
            print(f"q={q}: {status}", flush=True)

    return {
        "results": results,
        "proposed_batches": proposed_batches,
        "optimization_space": optimization_space,
        "coverage_evaluation_space": coverage_evaluation_space,
        "comparison_seed": comparison_seed,
    }


def refine_coverage_batch(
    existing, selected_q, comparison, *, cache_dir, seeds, verbose=True
):
    """Refine a selected batch using explicit seeds and the comparison's spaces.

    Pass the same existing inputs used by compare_coverage_batches and its
    returned dictionary. Retain its selected candidate unless a refinement
    achieves a strictly smaller maximum assessment distance. Each refinement
    reuses or saves its own cache entry. With selected_q=0, skip all searches.

    Return q, score, proposed coordinates and the winning seed. Print progress
    and the final choice unless verbose=False. Inputs are not modified.
    """
    if not isinstance(selected_q, (int, np.integer)) or selected_q < 0:
        raise ValueError("selected_q must be a non-negative integer.")
    selected_q = int(selected_q)
    selected_rows = [row for row in comparison["results"] if row["q"] == selected_q]
    if len(selected_rows) != 1:
        raise ValueError("The comparison must contain one result for selected_q.")
    best_batch = {
        "q": selected_q,
        "score": float(selected_rows[0]["max_distance"]),
        "proposed": comparison["proposed_batches"][selected_q],
        "seed": comparison["comparison_seed"],
    }

    if selected_q > 0:
        for seed in seeds:
            batch = cached_coverage_search(
                existing=existing,
                q=selected_q,
                optimization_space=comparison["optimization_space"],
                evaluation_space=comparison["coverage_evaluation_space"],
                cache_dir=cache_dir,
                seed=seed,
            )
            if verbose:
                status = "loaded from cache" if batch["from_cache"] else "computed and saved"
                print(f"q={selected_q}, seed={seed}: {status}", flush=True)

            if batch["max_distance"] < best_batch["score"]:
                best_batch = {
                    "q": selected_q,
                    "score": batch["max_distance"],
                    "proposed": batch["proposed"],
                    "seed": seed,
                }

    if verbose:
        print(
            f"Best batch: {best_batch['q']} queries, seed {best_batch['seed']}, "
            f"maximum assessment distance {best_batch['score']:.4f}."
        )
    return best_batch

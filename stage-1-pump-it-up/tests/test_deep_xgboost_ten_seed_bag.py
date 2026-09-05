"""Tests for the locked equal ten-seed deep-XGBoost variance test."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
SCRIPT_DIR = STAGE_DIR / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_partitioning import make_cross_validation
from data_partitioning import PartitionedData
from deep_xgboost_seed_average import FOLD_WIN_GATE
from deep_xgboost_seed_average import MEAN_ACCURACY_GAIN_GATE
from deep_xgboost_seed_average import REPAIR_RECALL_DELTA_GATE
from deep_xgboost_seed_average import WORST_FOLD_DELTA_GATE
from deep_xgboost_ten_seed_bag import confirm_equal_ten_seed_bag
from deep_xgboost_ten_seed_bag import FIT_SEEDS
from deep_xgboost_ten_seed_bag import LOCKED_SEEDS
from deep_xgboost_ten_seed_bag import passes_ten_seed_gate
from model_evaluation import build_candidate_evaluation
from run_deep_xgboost_ten_seed_bag import FIT_TIME_TWO_SEED_SHA256
from run_deep_xgboost_ten_seed_bag import INITIAL_FITTED_FOLDS
from run_deep_xgboost_ten_seed_bag import PINNED_EVIDENCE_SHA256
from run_deep_xgboost_ten_seed_bag import (
    _fit_time_baseline_evidence_hashes,
)
from run_deep_xgboost_ten_seed_bag import _write_csv


class DeepXgboostTenSeedBagTests(unittest.TestCase):
    def test_exact_equal_ten_seed_average_is_the_only_candidate(self) -> None:
        partitioned = _partitioned()
        cross_validation = make_cross_validation(partitioned)
        seeds = {
            seed: _evaluation(
                partitioned,
                cross_validation,
                f"seed {seed}",
                _probabilities(partitioned.y_development, seed),
            )
            for seed in LOCKED_SEEDS
        }
        two_values = 0.5 * (
            seeds[20260824].out_of_fold_probabilities.to_numpy()
            + seeds[20260905].out_of_fold_probabilities.to_numpy()
        )
        two_bag = _evaluation(
            partitioned,
            cross_validation,
            "two bag",
            two_values,
        )

        result = confirm_equal_ten_seed_bag(
            partitioned,
            cross_validation,
            seeds,
            two_bag,
        )

        expected = np.mean(
            [
                seeds[seed].out_of_fold_probabilities.to_numpy()
                for seed in LOCKED_SEEDS
            ],
            axis=0,
        )
        np.testing.assert_allclose(
            result.ten_seed_bag.out_of_fold_probabilities,
            expected,
        )
        self.assertEqual(
            set(result.summary.index),
            {
                "ten_seed_bag_vs_seed_20260824",
                "ten_seed_bag_vs_fixed_two_seed_bag",
            },
        )

    def test_seed_set_and_cached_two_bag_are_strict(self) -> None:
        partitioned = _partitioned()
        cross_validation = make_cross_validation(partitioned)
        values = _probabilities(partitioned.y_development, 0)
        evaluation = _evaluation(
            partitioned,
            cross_validation,
            "seed",
            values,
        )
        seeds = {seed: evaluation for seed in LOCKED_SEEDS}
        with self.assertRaisesRegex(ValueError, "evidence set"):
            confirm_equal_ten_seed_bag(
                partitioned,
                cross_validation,
                {seed: evaluation for seed in LOCKED_SEEDS[:-1]},
                evaluation,
            )

        wrong_two_bag = _evaluation(
            partitioned,
            cross_validation,
            "wrong",
            np.roll(values, 1, axis=1),
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            confirm_equal_ten_seed_bag(
                partitioned,
                cross_validation,
                seeds,
                wrong_two_bag,
            )

    def test_gate_boundaries_and_each_failure(self) -> None:
        passing = {
            "mean_accuracy_delta": MEAN_ACCURACY_GAIN_GATE,
            "fold_wins": FOLD_WIN_GATE,
            "worst_fold_delta": WORST_FOLD_DELTA_GATE,
            "repair_recall_delta": REPAIR_RECALL_DELTA_GATE,
        }
        self.assertTrue(passes_ten_seed_gate(**passing))
        failures = (
            {"mean_accuracy_delta": MEAN_ACCURACY_GAIN_GATE - 1e-5},
            {"fold_wins": FOLD_WIN_GATE - 1},
            {"worst_fold_delta": WORST_FOLD_DELTA_GATE - 1e-5},
            {"repair_recall_delta": REPAIR_RECALL_DELTA_GATE - 1e-5},
        )
        for failure in failures:
            with self.subTest(failure=failure):
                self.assertFalse(
                    passes_ten_seed_gate(**{**passing, **failure})
                )

    def test_fit_time_and_current_two_seed_provenance_remain_distinct(self) -> None:
        current = {
            "seed_20260824": {"result.json": "a" * 64},
            "seed_20260905": {"result.json": "b" * 64},
            "fixed_two_seed_bag": {
                "result.json": PINNED_EVIDENCE_SHA256["two_seed_result"],
                "seed-average-evaluation.joblib": PINNED_EVIDENCE_SHA256[
                    "two_seed_aggregate"
                ],
            },
        }

        fit_time = _fit_time_baseline_evidence_hashes(current)

        self.assertEqual(
            fit_time["fixed_two_seed_bag"],
            FIT_TIME_TWO_SEED_SHA256,
        )
        self.assertNotEqual(
            fit_time["fixed_two_seed_bag"],
            current["fixed_two_seed_bag"],
        )
        self.assertEqual(INITIAL_FITTED_FOLDS, len(FIT_SEEDS) * 5)
        self.assertEqual(fit_time["seed_20260824"], current["seed_20260824"])
        self.assertEqual(fit_time["seed_20260905"], current["seed_20260905"])

        changed = {
            **current,
            "fixed_two_seed_bag": {
                **current["fixed_two_seed_bag"],
                "result.json": "c" * 64,
            },
        }
        with self.assertRaisesRegex(ValueError, "current two-seed"):
            _fit_time_baseline_evidence_hashes(changed)

    def test_csv_replay_accepts_windows_newline_translation_only(self) -> None:
        frame = pd.DataFrame({"metric": [0.1, 0.2]}, index=["a", "b"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.csv"
            path.write_bytes(
                frame.to_csv(lineterminator="\n")
                .replace("\n", "\r\r\n")
                .encode()
            )

            _write_csv(path, frame)

            pd.DataFrame({"metric": [0.1, 0.3]}, index=["a", "b"]).to_csv(
                path
            )
            with self.assertRaisesRegex(FileExistsError, "different CSV"):
                _write_csv(path, frame)


def _partitioned() -> PartitionedData:
    labels = (
        "functional",
        "functional needs repair",
        "non functional",
    )
    target = pd.Series(labels * 10, name="status_group")
    rows = len(target)
    ids = pd.Series(range(rows), name="id")
    X = pd.DataFrame({"feature": range(rows)})
    return PartitionedData(
        development_ids=ids,
        X_development=X,
        y_development=target,
        local_test_ids=ids.iloc[0:0].copy(),
        X_local_test=X.iloc[0:0].copy(),
        y_local_test=target.iloc[0:0].copy(),
        validation_folds=pd.Series(
            np.repeat(np.arange(1, 6), 6),
            name="validation_fold",
        ),
        development_fingerprint="all labels",
        local_test_fingerprint="not reconstructed",
        cross_validation_fingerprint="fresh locked folds",
    )


def _probabilities(target: pd.Series, seed: int) -> np.ndarray:
    labels = (
        "functional",
        "functional needs repair",
        "non functional",
    )
    positions = {label: index for index, label in enumerate(labels)}
    probabilities = np.full((len(target), 3), 0.05, dtype="float64")
    probabilities[
        np.arange(len(target)),
        [positions[label] for label in target],
    ] = 0.90
    offset = (seed % 7) * 1e-4
    probabilities[:, 0] += offset
    probabilities[:, 1] -= offset
    return probabilities


def _evaluation(partitioned, cross_validation, name, probabilities):
    return build_candidate_evaluation(
        model_name=name,
        partitioned_data=partitioned,
        cross_validation=cross_validation,
        probability_values=probabilities,
        diagnostic_rows=[
            {"validation_fold": fold}
            for fold in range(1, 6)
        ],
    )


if __name__ == "__main__":
    unittest.main()

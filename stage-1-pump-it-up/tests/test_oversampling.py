"""Focused tests for leakage-safe minority-class oversampling."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from sklearn.model_selection import PredefinedSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_partitioning import PartitionedData
from oversampling import CLASS_LABELS
from oversampling import MINORITY_CLASS
from oversampling import STRATEGY_NAMES
from oversampling import evaluate_oversampling_strategies
from oversampling import random_oversample_minority
from oversampling import write_oversampled_fold_artifacts


class OversamplingTests(unittest.TestCase):
    def test_default_raises_minority_to_second_largest_class(self) -> None:
        X = pd.DataFrame(
            {
                "integer": pd.Series(range(9), dtype="int64"),
                "category": pd.Series(list("aaabbbccc"), dtype="string"),
            }
        )
        y = pd.Series(
            ["functional"] * 4
            + ["non functional"] * 3
            + [MINORITY_CLASS] * 2,
            name="status_group",
        )
        ids = pd.Series(range(100, 109), name="id")

        result = random_oversample_minority(X, y, ids, seed=17)

        self.assertEqual(result.counts_after[MINORITY_CLASS], 3)
        self.assertEqual(len(result.X), 10)
        self.assertEqual(result.lineage["is_replica"].sum(), 1)
        self.assertEqual(
            result.y.loc[result.lineage["is_replica"]].unique().tolist(),
            [MINORITY_CLASS],
        )
        self.assertEqual(result.X.dtypes.tolist(), X.dtypes.tolist())
        pd.testing.assert_frame_equal(result.X.iloc[:9], X)
        self.assertTrue(result.lineage["oversampled_row_id"].is_unique)

    def test_same_seed_reproduces_values_and_lineage(self) -> None:
        X, y, ids = self._small_training_data()

        first = random_oversample_minority(X, y, ids, target_count=8, seed=29)
        second = random_oversample_minority(X, y, ids, target_count=8, seed=29)

        pd.testing.assert_frame_equal(first.X, second.X)
        pd.testing.assert_series_equal(first.y, second.y)
        pd.testing.assert_frame_equal(first.lineage, second.lineage)

    def test_duplicate_source_ids_are_rejected(self) -> None:
        X, y, ids = self._small_training_data()
        ids.iloc[-1] = ids.iloc[0]

        with self.assertRaisesRegex(ValueError, "unique"):
            random_oversample_minority(X, y, ids)

    def test_all_strategies_use_unchanged_validation_folds(self) -> None:
        partitioned, cross_validation = self._partitioned_data()

        evaluation = evaluate_oversampling_strategies(
            partitioned,
            cross_validation,
            pipeline_factory=self._small_pipeline,
            seed=101,
        )

        self.assertEqual(
            len(evaluation.fold_metrics),
            len(STRATEGY_NAMES) * 3,
        )
        for strategy in STRATEGY_NAMES:
            self.assertEqual(
                int(evaluation.confusion_counts[strategy].to_numpy().sum()),
                len(partitioned.y_development),
            )
        validation_rows = evaluation.diagnostics["validation_rows"]
        self.assertTrue(validation_rows.eq(10).all())
        baseline_rows = evaluation.diagnostics.loc[
            "unbalanced baseline", "fit_rows"
        ]
        oversampled_rows = evaluation.diagnostics.loc[
            "random oversampling", "fit_rows"
        ]
        self.assertTrue((oversampled_rows > baseline_rows).all())
        self.assertFalse(evaluation.out_of_fold_predictions.isna().any().any())

    def test_writer_materialises_compressed_lineage_and_proves_no_test_overlap(
        self,
    ) -> None:
        partitioned, cross_validation = self._partitioned_data()

        with TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory)
            manifest = write_oversampled_fold_artifacts(
                partitioned,
                cross_validation,
                destination,
                seed=303,
            )
            manifest_from_disk = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            combined_path = (
                destination
                / "development-training-oversampled-with-lineage.csv.gz"
            )
            combined = pd.read_csv(combined_path)

            self.assertTrue(combined_path.is_file())
            self.assertEqual(manifest["local_test_id_overlap"], 0)
            self.assertEqual(manifest_from_disk["local_test_id_overlap"], 0)
            self.assertTrue(combined["oversampled_row_id"].is_unique)
            self.assertTrue(
                {
                    "oversampled_row_id",
                    "source_id",
                    "status_group",
                    "is_replica",
                    "replica_number_for_source",
                }.issubset(combined.columns)
            )
            self.assertFalse(
                set(combined["source_id"]).intersection(
                    set(partitioned.local_test_ids)
                )
            )
            self.assertTrue(
                (
                    destination
                    / "fold-01"
                    / "training-values-oversampled.csv.gz"
                ).is_file()
            )
            tracked_paths = [
                manifest_from_disk["final_refit_development_version"]["file"][
                    "path"
                ],
                *[
                    file_metadata["path"]
                    for fold in manifest_from_disk["folds"]
                    for file_metadata in fold["files"].values()
                ],
            ]
            self.assertTrue(all("\\" not in path for path in tracked_paths))
            self.assertTrue(
                all((destination / path).is_file() for path in tracked_paths)
            )

    @staticmethod
    def _small_training_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        X = pd.DataFrame({"feature": range(12)})
        y = pd.Series(
            ["functional"] * 6
            + ["non functional"] * 4
            + [MINORITY_CLASS] * 2,
            name="status_group",
        )
        ids = pd.Series(range(12), name="id")
        return X, y, ids

    @staticmethod
    def _small_pipeline() -> Pipeline:
        return Pipeline(
            [
                (
                    "classifier",
                    DecisionTreeClassifier(max_depth=3, random_state=11),
                )
            ]
        )

    @staticmethod
    def _partitioned_data() -> tuple[PartitionedData, PredefinedSplit]:
        y = pd.Series(
            ["functional"] * 14
            + ["non functional"] * 10
            + [MINORITY_CLASS] * 6,
            name="status_group",
        )
        X = pd.DataFrame(
            {
                "feature": np.arange(len(y)),
                "feature_2": np.arange(len(y)) % 4,
            }
        )
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
        fold_assignments = np.zeros(len(y), dtype="int64")
        for fold_number, (_, validation_positions) in enumerate(
            splitter.split(X, y),
        ):
            fold_assignments[validation_positions] = fold_number
        predefined = PredefinedSplit(fold_assignments)
        partitioned = PartitionedData(
            development_ids=pd.Series(range(1_000, 1_030), name="id"),
            X_development=X,
            y_development=y,
            local_test_ids=pd.Series(range(2_000, 2_006), name="id"),
            X_local_test=pd.DataFrame({"feature": range(6)}),
            y_local_test=pd.Series(
                [CLASS_LABELS[0]] * 6,
                name="status_group",
            ),
            validation_folds=pd.Series(fold_assignments + 1),
            development_fingerprint="development",
            local_test_fingerprint="local-test",
            cross_validation_fingerprint="cross-validation",
        )
        return partitioned, predefined


if __name__ == "__main__":
    unittest.main()

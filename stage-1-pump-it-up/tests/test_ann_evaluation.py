"""Focused tests for the bounded ANN evaluation plumbing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ann_evaluation import ANN_CANDIDATES
from ann_evaluation import ANN_EARLY_STOPPING_PATIENCE
from ann_evaluation import ANN_MAX_ITERATIONS
from ann_evaluation import ANN_SEED
from ann_evaluation import ANN_VALIDATION_FRACTION
from ann_evaluation import evaluate_ann_blends
from ann_evaluation import make_ann_pipeline


class AnnEvaluationTests(unittest.TestCase):
    def test_candidate_keys_names_and_shapes_are_unique(self) -> None:
        specs = list(ANN_CANDIDATES.values())

        self.assertEqual(len({spec.key for spec in specs}), len(specs))
        self.assertEqual(len({spec.model_name for spec in specs}), len(specs))
        self.assertEqual(
            {spec.hidden_layer_sizes for spec in specs},
            {(64,), (128,), (128, 64)},
        )
        self.assertEqual(
            {spec.feature_policy for spec in specs},
            {
                "baseline",
                "funder_frequency",
                "funder_rare20_frequency",
                "both_rare20_frequency",
            },
        )

    def test_pipeline_uses_scaled_sparse_features_and_early_stopping(self) -> None:
        spec = ANN_CANDIDATES["relu_128_64_regularised"]

        pipeline = make_ann_pipeline(spec)
        classifier = pipeline.named_steps["classifier"]
        preprocessing = pipeline.named_steps["preprocessing"]
        numeric = preprocessing.named_steps["column_preprocessing"].transformers[0][1]

        self.assertEqual(classifier.hidden_layer_sizes, (128, 64))
        self.assertEqual(classifier.alpha, 0.01)
        self.assertEqual(classifier.random_state, ANN_SEED)
        self.assertEqual(classifier.max_iter, ANN_MAX_ITERATIONS)
        self.assertTrue(classifier.early_stopping)
        self.assertEqual(
            classifier.validation_fraction,
            ANN_VALIDATION_FRACTION,
        )
        self.assertEqual(
            classifier.n_iter_no_change,
            ANN_EARLY_STOPPING_PATIENCE,
        )
        self.assertEqual(numeric.steps[-1][0], "standardisation")

    def test_high_cardinality_candidate_uses_audited_feature_policy(self) -> None:
        spec = ANN_CANDIDATES["relu_128_64_both_identity_frequency"]

        pipeline = make_ann_pipeline(spec)
        feature_engineer = pipeline.named_steps["preprocessing"].named_steps[
            "feature_engineering"
        ]

        self.assertEqual(
            feature_engineer.policy.key,
            "both_rare20_frequency",
        )

    def test_invalid_blend_weight_is_rejected_before_evaluation(self) -> None:
        placeholder = SimpleNamespace()

        with self.assertRaisesRegex(ValueError, "strictly between"):
            evaluate_ann_blends(
                placeholder,
                placeholder,
                placeholder,
                {},
                ann_weights=(0.0,),
            )


if __name__ == "__main__":
    unittest.main()

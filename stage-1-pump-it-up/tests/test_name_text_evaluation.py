"""Tests for the deferred-name character n-gram representation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from name_text_evaluation import NAME_TEXT_FIELDS
from name_text_evaluation import NameTextExtractor
from name_text_evaluation import make_name_text_pipeline


class NameTextEvaluationTests(unittest.TestCase):
    def test_extractor_preserves_field_tags_and_missing_states(self) -> None:
        frame = pd.DataFrame(
            {
                field: [" Pump  Name ", None]
                for field in NAME_TEXT_FIELDS
            }
        )

        documents = NameTextExtractor().fit_transform(frame)

        self.assertIn("funder_pump_name", documents[0])
        self.assertIn("wpt_name_pump_name", documents[0])
        self.assertIn("scheme_name___missing__", documents[1])
        self.assertEqual(len(documents), 2)

    def test_pipeline_freezes_vectoriser_and_regularisation_policy(self) -> None:
        pipeline = make_name_text_pipeline()
        features = dict(pipeline.named_steps["features"].transformer_list)
        vectorizer = features["name_text"].named_steps["tfidf"]
        classifier = pipeline.named_steps["classifier"]

        self.assertEqual(vectorizer.ngram_range, (3, 5))
        self.assertEqual(vectorizer.min_df, 3)
        self.assertEqual(vectorizer.max_features, 60_000)
        self.assertEqual(classifier.solver, "saga")
        self.assertEqual(classifier.C, 1.0)


if __name__ == "__main__":
    unittest.main()

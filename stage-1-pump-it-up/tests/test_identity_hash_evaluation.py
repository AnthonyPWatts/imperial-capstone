"""Tests for target-free deferred-identity hashing."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


STAGE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = STAGE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_engineering import EXPECTED_SOURCE_FEATURES
from identity_hash_evaluation import IDENTITY_HASH_FEATURES
from identity_hash_evaluation import IdentityTokenExtractor
from identity_hash_evaluation import make_identity_hash_preprocessor
from identity_hash_evaluation import make_identity_hash_transformer


class IdentityHashEvaluationTests(unittest.TestCase):
    def test_tokens_are_normalised_field_prefixed_and_target_free(self) -> None:
        frame = _source_frame()

        tokens = IdentityTokenExtractor().fit_transform(frame)

        self.assertEqual(len(tokens), 2)
        self.assertEqual(len(tokens[0]), 6)
        self.assertIn("funder=government", tokens[0])
        self.assertIn("wpt_name=pump a", tokens[0])
        self.assertIn("funder=__missing__", tokens[1])

    def test_hash_is_deterministic_non_negative_and_bounded(self) -> None:
        frame = _source_frame()
        transformer = make_identity_hash_transformer()

        first = transformer.fit_transform(frame)
        second = transformer.transform(frame)

        self.assertEqual(first.shape, (2, IDENTITY_HASH_FEATURES))
        self.assertEqual((first != second).nnz, 0)
        self.assertTrue(np.all(first.data >= 0))
        self.assertTrue(np.all(first.getnnz(axis=1) <= 6))

    def test_hash_appends_to_accepted_sparse_preprocessing(self) -> None:
        transformed = make_identity_hash_preprocessor().fit_transform(
            _source_frame()
        )

        self.assertEqual(transformed.shape[0], 2)
        self.assertGreater(transformed.shape[1], IDENTITY_HASH_FEATURES)
        self.assertTrue(np.isfinite(transformed.data).all())


def _source_frame() -> pd.DataFrame:
    defaults: dict[str, object] = {
        column: "value" for column in EXPECTED_SOURCE_FEATURES
    }
    defaults.update(
        {
            "amount_tsh": 10.0,
            "date_recorded": "2013-01-02",
            "funder": " Government ",
            "installer": "DWE",
            "wpt_name": "Pump   A",
            "gps_height": 100.0,
            "longitude": 31.0,
            "latitude": -6.0,
            "num_private": 0.0,
            "region_code": 11,
            "district_code": 4,
            "population": 100.0,
            "construction_year": 2000,
        }
    )
    second = defaults.copy()
    second.update({"funder": None, "wpt_name": ""})
    return pd.DataFrame([defaults, second], columns=EXPECTED_SOURCE_FEATURES)


if __name__ == "__main__":
    unittest.main()

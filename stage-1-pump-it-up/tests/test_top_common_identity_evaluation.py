"""Tests for fold-fitted top-common identity indicators."""

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

from feature_engineering import DEFERRED_HIGH_CARDINALITY_FEATURES
from feature_engineering import EXPECTED_SOURCE_FEATURES
from top_common_identity_evaluation import TopCommonIdentityEncoder
from top_common_identity_evaluation import make_top_common_occurrence_preprocessor


class TopCommonIdentityEvaluationTests(unittest.TestCase):
    def test_encoder_selects_top_values_without_prediction_rows(self) -> None:
        training = _source_frame(5)
        prediction = _source_frame(2)
        for feature in DEFERRED_HIGH_CARDINALITY_FEATURES:
            training.loc[:, feature] = [
                "common",
                "common",
                "second",
                "second",
                "rare",
            ]
            prediction.loc[:, feature] = ["rare", "unseen"]
        encoder = TopCommonIdentityEncoder(top_k=2).fit(training)

        encoded = encoder.transform(prediction).toarray()

        self.assertEqual(encoded.shape, (2, 12))
        self.assertEqual(encoded[0].sum(), 0)
        self.assertEqual(encoded[1].sum(), 0)
        for feature in DEFERRED_HIGH_CARDINALITY_FEATURES:
            self.assertNotIn("unseen", encoder.selected_values_[feature])

    def test_each_selected_field_contributes_at_most_one_indicator(self) -> None:
        training = _source_frame(5)
        prediction = _source_frame(1)
        for feature in DEFERRED_HIGH_CARDINALITY_FEATURES:
            training.loc[:, feature] = [
                "common",
                "common",
                "second",
                "second",
                "rare",
            ]
            prediction.loc[:, feature] = "common"
        encoder = TopCommonIdentityEncoder(top_k=2).fit(training)

        encoded = encoder.transform(prediction)

        self.assertEqual(encoded.shape, (1, 12))
        self.assertEqual(encoded.nnz, len(DEFERRED_HIGH_CARDINALITY_FEATURES))
        self.assertTrue(np.equal(encoded.data, 1).all())

    def test_equal_counts_use_deterministic_lexical_tie_break(self) -> None:
        training = _source_frame(3)
        training.loc[:, "funder"] = ["z", "a", "m"]

        encoder = TopCommonIdentityEncoder(top_k=2).fit(training)

        self.assertEqual(encoder.selected_values_["funder"], ("a", "m"))

    def test_default_encoder_preserves_raw_case_and_whitespace(self) -> None:
        training = _source_frame(4)
        training.loc[:, "funder"] = ["Acme", " acme ", "ACME", "Acme"]

        encoder = TopCommonIdentityEncoder(top_k=3).fit(training)

        self.assertEqual(
            encoder.selected_values_["funder"],
            ("Acme", " acme ", "ACME"),
        )

    def test_normalised_encoder_collapses_case_and_whitespace_with_sentinels(self) -> None:
        training = _source_frame(6)
        prediction = _source_frame(3)
        training.loc[:, "funder"] = [
            "  Acme  Water ",
            "ACME\tWATER",
            "acme water",
            "   ",
            None,
            "Other",
        ]
        prediction.loc[:, "funder"] = [" AcMe   Water ", "", None]

        encoder = TopCommonIdentityEncoder(top_k=4, normalise=True).fit(training)
        encoded = encoder.transform(prediction).toarray()
        offset = encoder.feature_offsets_["funder"]
        selected = encoder.selected_values_["funder"]

        self.assertEqual(selected[0], "acme water")
        self.assertIn("__blank__", selected)
        self.assertIn("__missing__", selected)
        for row, value in enumerate(("acme water", "__blank__", "__missing__")):
            self.assertEqual(encoded[row, offset + selected.index(value)], 1.0)

    def test_combined_occurrence_preprocessor_is_finite(self) -> None:
        training = _source_frame(60)

        preprocessor = make_top_common_occurrence_preprocessor()
        transformed = preprocessor.fit_transform(training)

        self.assertEqual(transformed.shape[0], len(training))
        self.assertEqual(
            transformed.shape[1],
            len(preprocessor.get_feature_names_out()),
        )
        values = transformed.data if hasattr(transformed, "data") else transformed
        self.assertTrue(np.isfinite(values).all())


def _source_frame(rows: int) -> pd.DataFrame:
    values: dict[str, object] = {
        column: "value" for column in EXPECTED_SOURCE_FEATURES
    }
    values.update(
        {
            "amount_tsh": 10.0,
            "date_recorded": "2013-01-01",
            "gps_height": 100.0,
            "longitude": 31.774483,
            "latitude": -0.998916,
            "num_private": 0.0,
            "region_code": 11,
            "district_code": 4,
            "population": 100.0,
            "construction_year": 2000,
        }
    )
    return pd.DataFrame(
        [values.copy() for _ in range(rows)],
        columns=EXPECTED_SOURCE_FEATURES,
    )


if __name__ == "__main__":
    unittest.main()

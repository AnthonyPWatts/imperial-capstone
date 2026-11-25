"""Protect cohort validation grouping and reconciling repeated measurements."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks/Shared"))
from cohort_surrogate_study import grouped_official_folds, _median_repeats, _training_peers


class CohortSurrogateValidationTests(unittest.TestCase):
    def test_portal_repeats_stay_in_one_fold(self):
        x = np.array([[.1000001], [.4], [.1000002], [.6], [.8], [.9]])
        folds = list(grouped_official_folds(x, np.arange(len(x)), splits=3, seed=42))
        for train, test in folds:
            self.assertFalse(set(train) & set(test))
            self.assertEqual(0 in test, 2 in test)
        self.assertCountEqual(np.concatenate([test for _, test in folds]), np.arange(len(x)))

    def test_inner_training_excludes_peers_matching_every_omitted_official_row(self):
        matches = np.array([[True, False, False, False],
                            [False, True, False, False],
                            [False, False, False, True]])
        # Row 0 is outer-test and row 2 is inner-validation.
        np.testing.assert_array_equal(_training_peers(matches, [1]), [False, True, True, False])

    def test_duplicate_median_retains_an_original_coordinate(self):
        x = np.array([[.1000001], [.4], [.1000002], [.6]])
        y = np.array([10., 2., 20., 3.])
        xx, yy = _median_repeats(x, y)
        np.testing.assert_array_equal(xx, x[[0, 1, 3]])
        np.testing.assert_array_equal(yy, [15., 2., 3.])


if __name__ == "__main__":
    unittest.main()

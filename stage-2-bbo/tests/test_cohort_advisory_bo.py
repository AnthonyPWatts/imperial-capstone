
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
sys.path[:0] = [str(NOTEBOOKS / "Shared"), str(NOTEBOOKS / "Function_1/FunctionSpecificShared")]

from cohort_advisory_bo import WEIGHTS, advisory_groups, rerank, residuals


class CohortAdvisoryTests(unittest.TestCase):
    def test_external_views_exclude_all_official_coordinates_and_each_other(self):
        owner = pd.DataFrame({"x1": [.1, .2], "y": [8., 9.]})
        current = pd.DataFrame({"x1": [.1000001, .3, .3], "y": [50., 2., 6.]})
        archive = pd.DataFrame({"x1": [.2, .3, .4], "y": [99., -100., 7.]})
        current_used, archive_used = advisory_groups(owner, current, archive, ["x1"])
        self.assertEqual(current_used.x1.tolist(), [.3])
        self.assertEqual(current_used.y.tolist(), [4.])
        self.assertEqual(current_used.repeat_variance.tolist(), [8.])
        self.assertEqual(current_used.evaluations.tolist(), [2])
        self.assertEqual(archive_used.x1.tolist(), [.4])

    def test_advisory_limits_keep_weak_owner_candidate_from_leapfrogging(self):
        ei = np.array([1., .98, .49])
        changes = np.array([[-100., -100.], [100., 100.], [1e12, 1e12]])
        advisory = rerank(ei, changes, WEIGHTS["10% each"])
        np.testing.assert_allclose(advisory, [.6, .984, .592])
        self.assertEqual(np.argmax(advisory), 1)
        np.testing.assert_allclose(rerank(ei, changes, WEIGHTS["Official only"]), ei)

    def test_unsupported_layers_supply_zero_residual(self):
        points = np.array([[.1, .2], [.3, .4]])
        np.testing.assert_array_equal(residuals([None, None], points), np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()

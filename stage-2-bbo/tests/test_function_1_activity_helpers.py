"""Guard the signed/activity distinction and the shared-candidate counterfactual."""

from pathlib import Path
import sys
import unittest

import numpy as np

BASE = Path(__file__).resolve().parents[1] / "notebooks"
sys.path.insert(0, str(BASE / "Shared"))
sys.path.insert(0, str(BASE / "Function_1/FunctionSpecificShared"))
from function_1_activity_helpers import activity_targets, common_activity_pools, proposal_geometry


class ActivityHypothesisTests(unittest.TestCase):
    def test_targets_preserve_tiny_nonzero_values_and_make_the_changed_objective_explicit(self):
        values = np.array([-.005, -1e-100, 1e-200, 1e-12])
        before = values.copy();targets = activity_targets(values, scale=1e-200)
        np.testing.assert_array_equal(values, before)
        np.testing.assert_array_equal(targets["raw_magnitude"], np.abs(before))
        np.testing.assert_array_equal(targets["magnitude"], np.abs(targets["signed"]))
        self.assertEqual(targets["signed"].argmax(), 3)
        self.assertEqual(targets["magnitude"].argmax(), 0)
        self.assertNotEqual(targets["raw_magnitude"][2], 0.)

    def test_movement_is_measured_against_the_declared_geometric_centre(self):
        reference = np.array([.6, .85]);point = np.array([.52, .54])
        result = proposal_geometry(point, reference)
        self.assertAlmostEqual(result["distance_to_centre"], np.sqrt(.02**2+.04**2))
        self.assertLess(result["change_in_centre_distance"], 0)
        self.assertGreater(result["percent_closer_to_centre"], 80)
        self.assertIsNone(proposal_geometry(point, [.5,.5])["percent_closer_to_centre"])

    def test_symmetric_pool_preserves_baseline_and_excludes_known_points_including_peers(self):
        inputs = np.array([[.2,.2],[.8,.8],[.4,.6]])
        known = np.vstack([inputs, [.5,.5]])
        pools = common_activity_pools(inputs, known, [-.1,.01,-.2], input_columns=["x1","x2"],
            search_settings=dict(broad_power=3, local_power=2, precision=6),
            prepared_point=[.7,.9], activity_anchor=[.4,.6])
        base = set(map(tuple,pools["baseline"][["x1","x2"]].to_numpy()))
        expanded = set(map(tuple,pools["symmetric"][["x1","x2"]].to_numpy()))
        self.assertTrue(base.issubset(expanded))
        self.assertIn((.7,.9), expanded)
        self.assertFalse(expanded.intersection(set(map(tuple,known))))


if __name__ == "__main__":
    unittest.main()

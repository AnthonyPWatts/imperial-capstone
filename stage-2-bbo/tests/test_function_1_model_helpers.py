"""Checks for Function 1's shared target transform."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" /
                       "Function_1" / "FunctionSpecificShared"))
from function_1_model_helpers import signed_log


class Function1ModelTests(unittest.TestCase):
    def test_signed_log_keeps_zero_sign_order_and_does_not_change_values(self):
        values = np.array([-0.01, -1e-100, -1e-200, 0, 1e-200, 1e-100, 1e-15])
        original = values.copy()
        heights = signed_log(values)
        np.testing.assert_array_equal(values, original)
        np.testing.assert_array_equal(np.sign(heights), np.sign(values))
        self.assertTrue(np.all(np.diff(heights) > 0))
        self.assertEqual(heights[3], 0)
        self.assertAlmostEqual(heights[4], np.log10(2))
        for scale in (0, -1, np.inf):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                signed_log(values, scale)



if __name__ == "__main__":
    unittest.main()

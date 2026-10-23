"""Check snapshot validation and fixed-plan handling across input dimensions."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks" / "Shared"))
from exploration_review_helpers import load_exploration_review, remaining_exploration_queries


class ExplorationSnapshotTests(unittest.TestCase):
    def observations(self, dimensions):
        columns = [f"input_{number}" for number in range(dimensions)]
        data = pd.DataFrame({"observation_id": ["a", "b", "c", "d"],
                             "source_round": [0, 0, 1, 2]})
        for number, name in enumerate(columns):
            data[name] = np.linspace(.1, .8, 4) + number * .01
        data["score"] = [.25, -.1, .25, .9]
        return data, columns

    def load(self, data, columns, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "observations.csv"
            data.to_csv(path, index=False)
            original_bytes = path.read_bytes()
            review = load_exploration_review(
                path, input_columns=columns, output_column="score", **kwargs,
            )
            self.assertEqual(path.read_bytes(), original_bytes)
        return review

    def test_two_and_five_inputs_keep_precision_tied_ranks_and_latest_mask(self):
        for dimensions in (2, 5):
            with self.subTest(dimensions=dimensions):
                data, columns = self.observations(dimensions)
                data.loc[0, columns[0]] = np.nextafter(.1, 1.)
                review = self.load(data, columns, expected_round_counts={0: 2, 1: 1, 2: 1})
                pd.testing.assert_frame_equal(review["inputs"], data[columns])
                pd.testing.assert_series_equal(review["outputs"], data["score"])
                np.testing.assert_array_equal(review["plot_data"]["rank"], [2, 4, 2, 1])
                np.testing.assert_array_equal(review["latest_mask"], [False, False, False, True])
                self.assertNotIn("rank", review["data"])
                review["plot_data"].loc[0, columns[0]] = .99
                self.assertEqual(review["data"].loc[0, columns[0]], data.loc[0, columns[0]])

    def test_invalid_snapshots_fail_before_they_can_be_plotted(self):
        data, columns = self.observations(3)
        cases = {
            "duplicate ID": data.assign(observation_id=["a", "a", "c", "d"]),
            "missing value": data.assign(score=[.25, np.nan, .25, .9]),
            "non-finite output": data.assign(score=[.25, np.inf, .25, .9]),
            "outside bounds": data.assign(input_0=[.1, .3, .5, 1.]),
            "fractional round": data.assign(source_round=[0, 0, 1, 2.5]),
            "wrong column order": data[data.columns[::-1]],
        }
        for label, invalid in cases.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                self.load(invalid, columns)
        with self.assertRaisesRegex(ValueError, "round history"):
            self.load(data, columns, expected_round_counts={0: 2, 1: 2})


class RemainingExplorationPlanTests(unittest.TestCase):
    def test_five_input_plan_preserves_coordinates_and_formats_remaining_weeks(self):
        plan = pd.DataFrame(np.linspace(.1, .9, 15).reshape(3, 5),
                            columns=[f"input_{i}" for i in range(5)])
        original = plan.copy(deep=True)
        remaining = remaining_exploration_queries(plan, current_week=4, first_plan_week=3)
        self.assertEqual(remaining.index.tolist(), [4, 5])
        self.assertEqual(remaining.index.name, "Week")
        np.testing.assert_array_equal(remaining[plan.columns], plan.iloc[1:])
        self.assertEqual(remaining.loc[4, "Portal input"],
                         "0.385714-0.442857-0.500000-0.557143-0.614286")
        pd.testing.assert_frame_equal(plan, original)

    def test_completed_plan_returns_empty_inputs_instead_of_reusing_a_query(self):
        plan = pd.DataFrame({"a": [.2, .8], "b": [.7, .1]})
        for week in (4, 5):
            with self.subTest(week=week):
                remaining = remaining_exploration_queries(plan, current_week=week)
                self.assertTrue(remaining.empty)
                self.assertEqual(list(remaining), ["a", "b", "Portal input"])
        future = remaining_exploration_queries(plan, current_week=1)
        self.assertEqual(future.index.tolist(), [2, 3])

    def test_unsafe_or_unrepresentable_plan_inputs_are_rejected(self):
        for value in (np.nan, np.inf, -.1, 1., .9999999):
            with self.subTest(value=value), self.assertRaises(ValueError):
                remaining_exploration_queries(pd.DataFrame({"a": [value]}), current_week=2)


if __name__ == "__main__":
    unittest.main()

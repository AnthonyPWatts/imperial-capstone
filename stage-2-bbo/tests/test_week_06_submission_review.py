"""Protect scarce weekly queries against malformed or repeated prepared inputs."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from review_week_06_queries import DIMENSIONS, apply_submission_confirmation, review_f1, validate_lines


class SubmissionReviewTests(unittest.TestCase):
    def setUp(self):
        self.points = [np.full(dimensions, .2) for dimensions in DIMENSIONS]
        self.reviews = [{"known": np.full((1, dimensions), .1)} for dimensions in DIMENSIONS]

    def test_format_is_exactly_eight_function_ordered_lines_with_portal_precision(self):
        text, checked = validate_lines(self.points, self.reviews, [])
        self.assertEqual(len(text.splitlines()), 8)
        self.assertTrue(text.endswith("\n"))
        self.assertEqual([len(line.split("-")) for line in text.splitlines()], list(DIMENSIONS))
        self.assertTrue(all(token == "0.200000" for line in text.splitlines() for token in line.split("-")))
        self.assertEqual([r["function"] for r in checked], list(range(1, 9)))

    def test_function_1_keeps_the_agreed_hunch_without_claiming_a_clear_signal(self):
        review = dict(summary={"weight_sensitivity": [{"Weight": 0., "x1": .537942, "x2": .522217}],
                               "point": [.608515, .846200]}, fitted={"input_columns": ["x1", "x2"]})
        # The decision contract can be checked without the ignored course data or fitted models.
        with patch("review_week_06_queries.point_record", side_effect=lambda _, point: {"point": np.asarray(point).tolist()}):
            point, decision = review_f1(review)
        np.testing.assert_array_equal(point, [.537942, .522217])
        self.assertEqual(decision["effective_ucb_weight"], 0.)
        self.assertFalse(decision["outputs_regarded_as_clearly_nonzero"])
        self.assertFalse(decision["numerical_outputs_modified"])
        self.assertNotEqual(point.tolist(), review["summary"]["point"])

    def test_missing_function_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly eight"):
            validate_lines(self.points[:-1], self.reviews, [])

    def test_invalid_last_function_is_rejected_before_any_artifact_write(self):
        for invalid in (1., -.01, np.nan, .1234567):
            with self.subTest(invalid=invalid):
                self.points[-1] = np.full(8, invalid)
                with self.assertRaisesRegex(ValueError, "Function 8"):
                    validate_lines(self.points, self.reviews, [])

    def test_wrong_input_dimension_is_rejected(self):
        self.points[2] = np.full(2, .2)
        with self.assertRaisesRegex(ValueError, "Function 3"):
            validate_lines(self.points, self.reviews, [])

    def test_peer_or_official_duplicate_is_rejected_at_six_decimal_precision(self):
        self.reviews[5]["known"] = np.array([[.20000001] * 5])
        with self.assertRaisesRegex(ValueError, "known official/peer"):
            validate_lines(self.points, self.reviews, [])

    def test_prior_submission_is_checked_even_if_not_in_the_current_known_inputs(self):
        text, _ = validate_lines(self.points, self.reviews, [])
        with TemporaryDirectory() as temporary:
            earlier = Path(temporary) / "submissions.txt"
            earlier.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "earlier official submission"):
                validate_lines(self.points, self.reviews, [earlier])
            self.assertEqual(earlier.read_text(), text)


class SubmissionConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(week=6, status="prepared_not_submitted", feedback_through_round=5,
                           official_observations=215, new_course_evaluations=0,
                           future_opportunities_after_week_06_if_submitted=7,
                           submission_file=dict(path="Week_06/submissions.txt", sha256="retained-input-hash"))
        self.confirmation = dict(week=6, status="submitted_awaiting_results", confirmed_on="2026-10-31",
            source="project_owner_confirmation", results_status="pending", exact_portal_timestamp=None,
            latest_confirmed_feedback_round=5, official_observations=215,
            remaining_queries_per_function_after_submission=7, submission_file=self.record["submission_file"].copy())

    def test_owner_confirmation_preserves_submission_without_inventing_results(self):
        updated = apply_submission_confirmation(self.record, self.confirmation)
        self.assertEqual(updated["status"], "submitted_awaiting_results")
        self.assertEqual(updated["remaining_queries_per_function_after_submission"], 7)
        self.assertEqual(updated["official_observations"], 215)
        self.assertEqual(updated["new_course_evaluations"], 0)
        self.assertTrue(updated["portal_submission_performed"])
        self.assertFalse(updated["portal_submission_performed_by_this_workflow"])
        self.assertIsNone(updated["submission_confirmation"]["exact_portal_timestamp"])

    def test_confirmation_for_different_input_bytes_is_rejected_before_status_changes(self):
        changed = {**self.confirmation, "submission_file": {**self.confirmation["submission_file"], "sha256": "different"}}
        with self.assertRaisesRegex(ValueError, "retained inputs"):
            apply_submission_confirmation(self.record, changed)
        self.assertEqual(self.record["status"], "prepared_not_submitted")

    def test_submission_cannot_add_results_or_change_the_round_or_remaining_budget(self):
        for field, value in (("week", 5), ("latest_confirmed_feedback_round", 6),
                             ("official_observations", 223), ("remaining_queries_per_function_after_submission", 8)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                apply_submission_confirmation(self.record, {**self.confirmation, field: value})
        self.assertEqual(self.record["status"], "prepared_not_submitted")


if __name__ == "__main__":
    unittest.main()

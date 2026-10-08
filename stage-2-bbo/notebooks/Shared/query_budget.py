"""Remaining weekly query opportunities for each capstone function."""

from operator import index
from pathlib import Path
import re


# One query per function in each of 13 rounds, corresponding to Modules 12–24.
TOTAL_QUERY_WEEKS = 13

__all__ = ["remaining_queries"]


def remaining_queries(week_dir, *, total_weeks=TOTAL_QUERY_WEEKS):
    """Return the per-function query budget from a Week_01/Week_1 folder name.

    Include the current week's query, assuming it has not yet been submitted.
    Week_02 therefore has 12 queries left; Week_13 has one; later weeks have
    none. Use the folder's course week, not the calendar date or CSV row count.
    No files are read and no directories need to exist.
    """
    try:
        total_weeks = index(total_weeks)
    except TypeError as error:
        raise ValueError("total_weeks must be a positive integer.") from error
    if total_weeks < 1:
        raise ValueError("total_weeks must be a positive integer.")

    week_name = Path(week_dir).name
    match = re.fullmatch(r"Week_(\d+)", week_name, flags=re.IGNORECASE)
    if match is None or int(match[1]) < 1:
        raise ValueError(f"Expected a week folder such as Week_02; got {week_name!r}.")
    return max(0, total_weeks - int(match[1]) + 1)

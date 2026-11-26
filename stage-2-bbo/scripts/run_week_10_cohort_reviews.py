"""Execute the Week 10 full-cohort exploration and model notebooks."""

from run_week_07_reviews import main


def cohort_sources(stage):
    snapshot = stage / "data/cohort/2026-11-25"
    yield snapshot / "dataset-audit.json"
    for function in range(1, 9):
        yield snapshot / f"function_{function}/observations.csv"
        week = stage / "notebooks" / f"Function_{function}/Week_10"
        official = week / "official" if function <= 4 else week
        yield official / "model-review.json"


if __name__ == "__main__":
    main(week=10, forecast_sources=cohort_sources, full_cohort=True)

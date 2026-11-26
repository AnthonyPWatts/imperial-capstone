"""Execute the Week 10 official exploration and model notebooks."""

from run_week_07_reviews import main


def week_09_sources(stage):
    yield stage / "submissions/round-09-results.json"
    yield stage / "submissions/Week_09/submission-confirmation.json"
    for function in range(1, 9):
        base = stage / "notebooks" / f"Function_{function}"
        previous = base / "Week_09"
        folder = previous / "official" if function <= 4 else previous
        for name in ("observations.csv", "model-review.json", "figures/neural-network-explorer.html"):
            yield folder / name
        if function <= 4:
            yield previous / "cohort/data-audit.json"
            yield base / "Week_10/cohort/data-audit.json"


def decision_references(stage):
    decisions = stage / "submissions/Week_10"
    for name in ("initial-proposal.md", "advisory-submissions.txt", "submissions.txt",
                 "official-data-submissions.txt"):
        yield decisions / name
    cohort = stage / "data/cohort/2026-11-25"
    yield from cohort.glob("dataset-audit.json")
    yield from cohort.glob("function_*/observations.csv")
    archive = stage / "data/historical-cohort/2024"
    yield from archive.glob("intake-audit.json")
    yield from archive.glob("function_*.csv")


if __name__ == "__main__":
    main(week=10, forecast_sources=week_09_sources, decision_reference_sources=decision_references)

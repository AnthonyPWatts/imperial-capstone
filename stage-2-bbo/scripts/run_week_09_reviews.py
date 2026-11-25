"""Execute the Week 9 exploration and model notebooks with the Capstone kernel."""

from run_week_07_reviews import main


def week_08_sources(stage):
    yield stage / "submissions/round-08-results.json"
    yield stage / "submissions/Week_08/submission-confirmation.json"
    for function in range(1, 9):
        base = stage / "notebooks" / f"Function_{function}"
        previous = base / "Week_08"
        folder = previous / "official" if function <= 4 else previous
        for name in ("observations.csv", "model-review.json", "figures/neural-network-explorer.html"):
            yield folder / name
        if function <= 4:
            yield previous / "cohort/data-audit.json"
            yield base / "Week_09/cohort/data-audit.json"


if __name__ == "__main__":
    main(week=9, forecast_sources=week_08_sources)

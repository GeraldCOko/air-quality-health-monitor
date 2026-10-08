"""
Check how trustworthy the estimate for a place without a monitor (Antigonish) is.

Two checks, both using only real measurements:

1. For each estimated place, show which monitors it is built from, their distances and weights, and how closely
   those monitors agree with each other. If the two monitors either side of Antigonish rise and fall together,
   a point between them very likely does too.
2. Leave-one-out: hide each real monitor in turn, estimate it from its neighbours in exactly the same way, and
   compare with what it really measured. Those neighbours are further away than Antigonish's, so this is a
   deliberately tough test.

Writes ``models/estimate_check.csv`` for the dashboard.

Usage:  python -m scripts.check_estimate
"""

import pandas as pd

from aqmonitor import config, places
from scripts import cli


def _agreement(readings: pd.DataFrame, first: int, second: int) -> tuple[float, float, float]:
    wide = readings.pivot(index="time", columns="station_id", values="value")[[first, second]].dropna()
    daily = wide.resample("D").mean().dropna()
    gap = float((wide[first] - wide[second]).abs().mean())
    return float(wide[first].corr(wide[second])), float(daily[first].corr(daily[second])), gap


def main() -> None:
    """Print and save the checks on the estimating method."""
    stations, table = cli.load_stations(), cli.load_table()
    readings = table.rename(columns={config.PARAMETER: "value"})[["station_id", "time", "value"]]
    for place in places.estimates(stations).itertuples():
        chosen = places.sources(place.latitude, place.longitude, places.monitors(stations))
        cli.say(f"{place.name} is estimated from:")
        for row in chosen.itertuples():
            cli.say(f"  {row.name:<20} {row.distance_km:5.1f} km away   weight {row.weight:.0%}")
        if len(chosen) == 2:
            hourly, daily, gap = _agreement(readings, int(chosen["station_id"][0]), int(chosen["station_id"][1]))
            cli.say(
                f"  These two agree with a correlation of {hourly:.2f} hour by hour and {daily:.2f} day by day; "
                f"on average they differ by {gap:.1f} µg/m³."
            )
    scores = places.leave_one_out(readings, stations)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(config.ESTIMATE_CHECK_FILE, index=False)
    cli.say("\nLeave-one-out test: each real monitor estimated from its two nearest neighbours\n")
    headings = ("neighbours", "hourly miss", "hourly corr", "daily miss", "daily corr")
    cli.say(f"  {'monitor':<20} " + " ".join(f"{heading:>11}" for heading in headings))
    for row in scores.itertuples():
        cli.say(
            f"  {row.name:<20} {row.source_distance_km:>8.0f} km {row.mae_hourly:>11.2f} "
            f"{row.correlation_hourly:>11.2f} {row.mae_daily:>11.2f} {row.correlation_daily:>11.2f}"
        )
    cli.say("\n'miss' is the average error in µg/m³; correlation runs from 0 (unrelated) to 1 (move together).")


if __name__ == "__main__":
    main()

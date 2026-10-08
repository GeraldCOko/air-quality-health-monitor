"""
Step 3: clean the raw downloads and join air quality with weather into one hourly table.

Writes ``data/processed/hourly.csv`` and prints how complete each station's record is.

Usage:  python -m scripts.build_dataset
"""

import pandas as pd

from aqmonitor import config, dataset, places
from scripts import cli


def main() -> None:
    """Build the hourly training table from everything in ``data/raw``."""
    stations = cli.load_stations()
    files = sorted(config.RAW_OPENAQ_DIR.glob("*/*.csv.gz"))
    if not files:
        cli.stop("no archive files in data/raw/openaq. Run  python -m scripts.download_data  first.")
    cli.say(f"Reading {len(files)} daily archive files ...")
    readings = dataset.read_archive(files, config.PARAMETER)
    readings = readings[readings["time"] >= pd.Timestamp(config.HISTORY_START)]
    measured = dataset.clean_hourly(readings)
    air = pd.concat([places.estimate_readings(measured, stations), measured], ignore_index=True)
    weather_frames: dict[int, pd.DataFrame] = {}
    for station_id in stations["station_id"]:
        path = config.RAW_WEATHER_DIR / f"station_{station_id}.csv"
        if path.exists():
            weather_frames[int(station_id)] = pd.read_csv(path, parse_dates=["time"])
    table = dataset.join_weather(air, weather_frames, config.PARAMETER)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.DATASET_FILE, index=False)
    summary = dataset.coverage(table, config.PARAMETER).merge(stations[["station_id", "name"]], on="station_id")
    cli.say(f"\nWrote {len(table):,} hourly rows to {config.DATASET_FILE.relative_to(config.PROJECT_ROOT)}\n")
    for row in summary.itertuples():
        cli.say(
            f"  {row.name:<28} {row.first:%Y-%m-%d} to {row.last:%Y-%m-%d}   "
            f"{row.share_with_reading:5.1%} of hours have a reading   "
            f"mean {row.mean:4.1f}  max {row.maximum:5.1f} µg/m³"
        )
    cli.say("\nDone. Next:  python -m scripts.train")


if __name__ == "__main__":
    main()

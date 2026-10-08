"""
Step 2: download the training data - air-quality history from the OpenAQ archive and weather from Open-Meteo.

Neither source needs a key. Files already on disk are skipped, so the script can be re-run at any time to top up
with newer days, and it can be interrupted and restarted without losing work.

Usage:  python -m scripts.download_data [--start 2022-01-01]
"""

import argparse

import pandas as pd

from aqmonitor import config, openaq, places, weather
from scripts import cli


def _download_air(stations: pd.DataFrame, start: pd.Timestamp, today: pd.Timestamp) -> None:
    for station in places.monitors(stations).itertuples():
        keys: list[str] = []
        for year in range(start.year, today.year + 1):
            keys.extend(openaq.archive_keys(int(station.station_id), year))
        wanted = [key for key in keys if key.rsplit("-", 1)[-1][:8] >= start.strftime("%Y%m%d")]
        fetched = openaq.download_archive(wanted, config.RAW_OPENAQ_DIR)
        cli.say(f"  {station.name}: {len(wanted)} daily files in the archive, {fetched} newly downloaded")


def _download_weather(stations: pd.DataFrame, start: pd.Timestamp, today: pd.Timestamp) -> None:
    config.RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    for station in stations.itertuples():
        frame = weather.fetch_history(
            station.latitude, station.longitude, start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        )
        frame.to_csv(config.RAW_WEATHER_DIR / f"station_{station.station_id}.csv", index=False)
        cli.say(f"  {station.name}: {len(frame)} hours of weather")


def main() -> None:
    """Download air-quality and weather history for every station in the station list."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=config.HISTORY_START, help="first day to download, YYYY-MM-DD")
    arguments = parser.parse_args()
    stations = cli.load_stations()
    start, today = pd.Timestamp(arguments.start), pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    cli.say(f"Downloading air-quality history from the OpenAQ archive since {start:%Y-%m-%d} ...")
    _download_air(stations, start, today)
    cli.say("Downloading weather history from Open-Meteo ...")
    _download_weather(stations, start, today)
    cli.say("Done. Next:  python -m scripts.build_dataset")


if __name__ == "__main__":
    main()

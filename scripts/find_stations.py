"""
Step 1: ask OpenAQ which monitoring stations are inside the region and save the list.

Needs the OpenAQ API key in ``.env``. Writes ``data/stations.csv``, which every later step reads. The file is
plain CSV: delete a row to drop a station, then re-run the later steps.

Three rules decide which stations are kept:

* Stations silent for more than ``--active-days`` days are left out, because a monitor that no longer reports
  cannot be mapped or forecast.
* Only reference-grade (government-standard) monitors are kept unless ``--include-low-cost`` is given. Low-cost
  community sensors are less accurate and would blur the comparison between stations.
* A station already in the list is kept even when OpenAQ's search no longer returns it, as long as its sensor
  is still reporting. OpenAQ's search drops some live stations (Pictou in October 2026), and losing Pictou would
  quietly remove one of the two monitors the Antigonish estimate is built from.

Usage:  python -m scripts.find_stations [--active-days 30] [--include-low-cost]
"""

import argparse

import pandas as pd

from aqmonitor import config, openaq, region
from scripts import cli


def _search(client: openaq.OpenAQClient, include_low_cost: bool) -> pd.DataFrame:
    found = openaq.stations_table(client.locations(region.bounding_box(), config.PARAMETER_ID), config.PARAMETER_ID)
    inside = [region.contains(row.longitude, row.latitude) for row in found.itertuples()]
    found = found[inside]
    return found if include_low_cost else found[found["reference_grade"]]


def _still_reporting(client: openaq.OpenAQClient, found: pd.DataFrame, newest_allowed: pd.Timestamp) -> pd.DataFrame:
    if not config.STATIONS_FILE.exists():
        return found.iloc[:0]
    known = pd.read_csv(config.STATIONS_FILE)
    kept: list[pd.DataFrame] = []
    for index, row in known[~known["station_id"].isin(found["station_id"])].iterrows():
        last_seen = client.sensor_last_seen(int(row["sensor_id"]))
        if last_seen is not None and last_seen >= newest_allowed:
            kept.append(known.loc[[index]].assign(last_utc=last_seen.strftime("%Y-%m-%dT%H:%M:%SZ")))
    return pd.concat(kept) if kept else found.iloc[:0]


def main() -> None:
    """Find active PM2.5 stations in the configured region and write the station list."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--active-days", type=int, default=30, help="drop stations silent for longer than this")
    parser.add_argument("--include-low-cost", action="store_true", help="also keep low-cost community sensors")
    arguments = parser.parse_args()
    client = cli.client_or_none()
    if client is None:
        cli.stop("no OpenAQ API key found. Paste your key into the .env file (see README, Setup).")
    cli.say(f"Asking OpenAQ for {config.PARAMETER} stations in {config.REGION_NAME} ...")
    newest_allowed = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=arguments.active_days)
    found = _search(client, arguments.include_low_cost)
    last_seen = pd.to_datetime(found["last_utc"], utc=True).dt.tz_localize(None)
    active = found[last_seen >= newest_allowed]
    rescued = _still_reporting(client, found, newest_allowed)
    stations = pd.concat([active, rescued], ignore_index=True).reindex(columns=openaq.STATION_COLUMNS)
    stations["reference_grade"] = stations["reference_grade"].fillna(True).astype(bool)
    stations = stations.sort_values("name").reset_index(drop=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    stations.to_csv(config.STATIONS_FILE, index=False)
    cli.say(f"OpenAQ's search returned {len(found)} stations; {len(active)} of them reported recently.")
    for row in rescued.itertuples():
        cli.say(f"Kept {row.name}: missing from OpenAQ's search, but its sensor is still reporting.")
    cli.say(f"\nStation list ({len(stations)}):")
    for row in stations.itertuples():
        cli.say(f"  {row.station_id:>8}  {row.name:<32} {row.latitude:8.4f} {row.longitude:9.4f}  {row.provider}")
    cli.say(f"\nSaved to {config.STATIONS_FILE.relative_to(config.PROJECT_ROOT)}")
    cli.say("Next:  python -m scripts.download_data")


if __name__ == "__main__":
    main()

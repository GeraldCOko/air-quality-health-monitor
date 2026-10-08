"""
Clean the raw downloads and join them into one tidy hourly table the model can learn from.

The raw OpenAQ archive is thousands of small daily files with one row per reading. This module:

1. reads them and keeps one pollutant (PM2.5);
2. converts every timestamp to UTC, so stations and weather line up without daylight-saving surprises;
3. removes impossible readings. Monitors report a sentinel such as -999 when they have no value, and a healthy
   PM2.5 instrument can read slightly below zero in very clean air. Readings below ``_MIN_VALID`` or above
   ``_MAX_VALID`` are treated as missing; the small negatives that remain are set to zero;
4. lays each station on a complete hourly grid and fills gaps of up to ``_MAX_GAP_HOURS`` hours by drawing a
   straight line between the readings on either side. Longer gaps stay empty: inventing a day of data would
   teach the model something that never happened;
5. attaches the weather for the same station and hour.

The result has one row per station per hour: ``station_id, time, pm25`` and the weather columns.
"""

from pathlib import Path

import pandas as pd

_MIN_VALID: float = -10.0
_MAX_VALID: float = 2000.0
_MAX_GAP_HOURS: int = 3

_ARCHIVE_COLUMNS: list[str] = ["location_id", "datetime", "parameter", "value"]


def read_archive(files: list[Path], parameter: str) -> pd.DataFrame:
    """Read daily archive files and return ``station_id, time, value`` for one parameter, time in UTC."""
    pieces = [pd.read_csv(file, usecols=_ARCHIVE_COLUMNS) for file in files]
    if not pieces:
        return pd.DataFrame(columns=["station_id", "time", "value"])
    raw = pd.concat(pieces, ignore_index=True)
    raw = raw[raw["parameter"] == parameter]
    return pd.DataFrame(
        {
            "station_id": raw["location_id"].astype(int).to_numpy(),
            "time": pd.to_datetime(raw["datetime"], utc=True).dt.tz_localize(None).to_numpy(),
            "value": pd.to_numeric(raw["value"], errors="coerce").to_numpy(),
        }
    )


def _fill_short_gaps(series: pd.Series) -> pd.Series:
    missing = series.isna()
    gap_length = missing.groupby((~missing).cumsum()).transform("sum")
    return series.interpolate(limit_area="inside").where(~missing | (gap_length <= _MAX_GAP_HOURS))


def clean_hourly(readings: pd.DataFrame) -> pd.DataFrame:
    """
    Turn raw ``station_id, time, value`` readings into a gap-aware hourly series per station.

    Invalid readings are dropped, duplicates within an hour are averaged, each station is laid on a complete
    hourly grid from its first to its last reading, and short gaps are filled by linear interpolation.
    """
    valid = readings[readings["value"].between(_MIN_VALID, _MAX_VALID)].copy()
    valid["value"] = valid["value"].clip(lower=0.0)
    valid["time"] = pd.to_datetime(valid["time"]).dt.ceil("h")
    series: list[pd.DataFrame] = []
    for station_id, group in valid.groupby("station_id"):
        hourly = group.groupby("time")["value"].mean()
        grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", name="time")
        series.append(_fill_short_gaps(hourly.reindex(grid)).reset_index().assign(station_id=station_id))
    if not series:
        return pd.DataFrame(columns=["station_id", "time", "value"])
    return pd.concat(series, ignore_index=True)[["station_id", "time", "value"]]


def join_weather(air: pd.DataFrame, weather: dict[int, pd.DataFrame], parameter: str) -> pd.DataFrame:
    """Attach each station's weather to its hourly readings; the reading column is renamed to the parameter."""
    joined: list[pd.DataFrame] = []
    for station_id, group in air.groupby("station_id"):
        if station_id not in weather:
            continue
        joined.append(group.merge(weather[station_id], on="time", how="left"))
    if not joined:
        return pd.DataFrame(columns=["station_id", "time", parameter])
    table = pd.concat(joined, ignore_index=True).rename(columns={"value": parameter})
    return table.sort_values(["station_id", "time"]).reset_index(drop=True)


def coverage(table: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Summarise each station's record: first and last hour, hours on the grid, and the share with a reading."""
    summary = table.groupby("station_id").agg(
        first=("time", "min"),
        last=("time", "max"),
        hours=("time", "size"),
        with_reading=(parameter, "count"),
        mean=(parameter, "mean"),
        maximum=(parameter, "max"),
    )
    summary["share_with_reading"] = summary["with_reading"] / summary["hours"]
    return summary.reset_index()

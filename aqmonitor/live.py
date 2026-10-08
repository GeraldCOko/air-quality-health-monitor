"""
Bring the monitor up to date: newest readings, newest weather forecast, and a fresh 72-hour PM2.5 forecast.

One refresh does three things for every station:

1. **Readings.** The last ``LIVE_LOOKBACK_DAYS`` days come from the training table (built from the OpenAQ
   archive, which runs several days behind) and, when an API key is present, from the live OpenAQ API. Where both
   have the same hour the live value wins. Without a key the monitor still works but shows archive data, and says
   so.
2. **Weather.** Open-Meteo's forecast for the station covers the same past days and the days ahead.
3. **Forecast.** The station's saved model is anchored on its newest reading and run forward to
   ``FORECAST_HOURS`` hours past the current time. If a station has been silent for a while, the forecast still
   starts from now; the further its last reading lies in the past, the less it sways the forecast.

Results are written as plain CSV and JSON files in ``data/live/`` for the dashboard to read. A problem with one
station is recorded in the status file and does not stop the others.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aqmonitor import config, dataset, forecast, openaq, places, weather

_FORECAST_DAYS: int = 5
SOURCE_LIVE: str = "OpenAQ live API"
SOURCE_ARCHIVE: str = "OpenAQ archive only (no API key yet, so readings are several days old)"


def utc_now() -> pd.Timestamp:
    """The current time as a timezone-naive UTC timestamp, matching the rest of the project."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def model_path(station_id: int, folder: Path = config.MODELS_DIR) -> Path:
    """Where a station's fitted model is saved."""
    return folder / f"station_{station_id}.json"


def gather_readings(
    stations: pd.DataFrame,
    table: pd.DataFrame,
    client: openaq.OpenAQClient | None,
    now: pd.Timestamp,
    problems: list[str],
) -> pd.DataFrame:
    """
    Combine recent archive readings with live API readings into a clean hourly ``station_id, time, pm25``.

    Estimated places (see ``aqmonitor.places``) are rebuilt from the monitors' freshest readings, so an estimate
    never lags behind the monitors it is made from.
    """
    start = now - pd.Timedelta(days=config.LIVE_LOOKBACK_DAYS)
    recent = table.loc[table["time"] >= start, ["station_id", "time", config.PARAMETER]]
    pieces = [recent.rename(columns={config.PARAMETER: "value"}).dropna(subset=["value"])]
    if client is not None:
        for station in places.monitors(stations).itertuples():
            try:
                fresh = openaq.readings_table(client.sensor_readings(int(station.sensor_id), start, now))
                pieces.append(fresh.assign(station_id=int(station.station_id)))
            except (openaq.OpenAQError, OSError) as error:
                problems.append(f"{station.name}: could not fetch live readings ({error})")
    combined = pd.concat(pieces, ignore_index=True).drop_duplicates(["station_id", "time"], keep="last")
    combined = combined[
        (combined["time"] <= now) & combined["station_id"].isin(places.monitors(stations)["station_id"])
    ]
    measured = dataset.clean_hourly(combined)
    everything = pd.concat([measured, places.estimate_readings(measured, stations)], ignore_index=True)
    return everything.rename(columns={"value": config.PARAMETER})


def station_forecast(
    station_id: int, readings: pd.DataFrame, weather_frame: pd.DataFrame, now: pd.Timestamp
) -> pd.DataFrame:
    """Run one station's saved model from the current hour to ``FORECAST_HOURS`` ahead."""
    model = forecast.load(model_path(station_id))
    own = readings.loc[readings["station_id"] == station_id, ["time", config.PARAMETER]]
    frame = weather_frame.merge(own, on="time", how="left")
    this_hour = now.floor("h")
    history = frame[frame["time"] <= this_hour]
    ahead = frame[
        (frame["time"] > this_hour) & (frame["time"] <= this_hour + pd.Timedelta(hours=config.FORECAST_HOURS))
    ]
    result = forecast.forecast(model, history, ahead, config.PARAMETER)
    result.insert(0, "station_id", station_id)
    result["hours_ahead"] = ((result["time"] - this_hour) / pd.Timedelta(hours=1)).round().astype(int)
    return result


def refresh(stations: pd.DataFrame, table: pd.DataFrame, client: openaq.OpenAQClient | None) -> dict[str, Any]:
    """Refresh readings and forecasts for every station, write them to ``data/live/`` and return the status."""
    now = utc_now()
    problems: list[str] = []
    readings = gather_readings(stations, table, client, now, problems)
    forecasts: list[pd.DataFrame] = []
    for station in stations.itertuples():
        station_id = int(station.station_id)
        if not model_path(station_id).exists():
            problems.append(f"{station.name}: no trained model, so no forecast")
            continue
        try:
            outlook = weather.fetch_forecast(
                station.latitude, station.longitude, config.LIVE_LOOKBACK_DAYS, _FORECAST_DAYS
            )
            forecasts.append(station_forecast(station_id, readings, outlook, now))
        except (RuntimeError, OSError, ValueError) as error:
            problems.append(f"{station.name}: forecast failed ({error})")
    status = {
        "updated_utc": now.strftime("%Y-%m-%d %H:%M"),
        "source": SOURCE_LIVE if client is not None else SOURCE_ARCHIVE,
        "stations_with_forecast": len(forecasts),
        "problems": problems,
    }
    config.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    readings.to_csv(config.LIVE_OBSERVATIONS_FILE, index=False, float_format="%.2f")
    columns = ["station_id", "time", config.PARAMETER, "lower", "upper", "hours_ahead"]
    all_forecasts = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame(columns=columns)
    all_forecasts.to_csv(config.LIVE_FORECAST_FILE, index=False, float_format="%.2f")
    config.LIVE_STATUS_FILE.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status


def latest_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """The newest non-missing reading per station, as ``station_id, time, pm25``."""
    known = readings.dropna(subset=[config.PARAMETER]).sort_values("time")
    return known.groupby("station_id").tail(1).reset_index(drop=True)

"""
Fetch hourly weather for a point on the map from Open-Meteo (free, no key).

Weather is the "optional extra" that the forecast leans on: wind blows pollution away, rain washes it out, and
cold still nights trap it near the ground. Two Open-Meteo services are used:

* the Historical Forecast API, which keeps every past run of the weather models stitched into one continuous
  record from 2022 onward, supplies the training history;
* the Forecast API supplies the last few days plus the days ahead when a forecast is made.

Both come from the same weather models, so the numbers the forecast sees when it runs look like the numbers it
was trained on. All times are UTC. Temperature, humidity, wind and pressure are values at the stated instant;
precipitation is the total over the hour ending then, the same hour-ending convention as the OpenAQ readings.

References
----------
Open-Meteo documentation: https://open-meteo.com/en/docs
Zippenfenig, P. (2023). Open-Meteo.com Weather API. Zenodo. https://doi.org/10.5281/zenodo.7970649
"""

import time
from typing import Any

import pandas as pd
import requests

from aqmonitor import config

HISTORY_URL: str = "https://historical-forecast-api.open-meteo.com/v1/forecast"
FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"

_TIMEOUT_SECONDS: int = 120
_MAX_ATTEMPTS: int = 5


def _get(url: str, params: dict[str, Any], session: requests.Session | None) -> dict[str, Any]:
    http = session or requests.Session()
    for attempt in range(_MAX_ATTEMPTS):
        response = http.get(url, params=params, timeout=_TIMEOUT_SECONDS)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(20.0 * (attempt + 1))
            continue
        raise RuntimeError(f"Open-Meteo answered {response.status_code}: {response.text[:200]}")
    raise RuntimeError(f"Open-Meteo kept failing after {_MAX_ATTEMPTS} attempts.")


def to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Turn an Open-Meteo JSON answer into a frame with a UTC ``time`` column and one column per variable."""
    frame = pd.DataFrame(payload["hourly"])
    frame["time"] = pd.to_datetime(frame["time"])
    return frame


def _base_params(latitude: float, longitude: float) -> dict[str, Any]:
    return {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "hourly": ",".join(config.WEATHER_VARIABLES),
        "timezone": "UTC",
    }


def fetch_history(
    latitude: float, longitude: float, start: str, end: str, session: requests.Session | None = None
) -> pd.DataFrame:
    """Hourly weather at a point between two dates (inclusive, ``YYYY-MM-DD``), fetched one year at a time."""
    first, last = pd.Timestamp(start), pd.Timestamp(end)
    pieces: list[pd.DataFrame] = []
    for year in range(first.year, last.year + 1):
        chunk_start = max(first, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(last, pd.Timestamp(year=year, month=12, day=31))
        params = _base_params(latitude, longitude) | {
            "start_date": chunk_start.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
        }
        pieces.append(to_frame(_get(HISTORY_URL, params, session)))
    return pd.concat(pieces, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_forecast(
    latitude: float, longitude: float, past_days: int, forecast_days: int, session: requests.Session | None = None
) -> pd.DataFrame:
    """Hourly weather at a point for the last ``past_days`` days and the next ``forecast_days`` days."""
    params = _base_params(latitude, longitude) | {"past_days": past_days, "forecast_days": forecast_days}
    return to_frame(_get(FORECAST_URL, params, session))

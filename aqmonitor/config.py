"""
Project settings: where files live, which region is monitored, and how the forecast is set up.

Everything a newcomer is likely to change sits in this one module. To monitor a different place, change
``REGION_NAME`` here and the outline in ``aqmonitor/region.py``, then re-run the workflow from
``python -m scripts.find_stations``.

The OpenAQ API key is never written in code. It is read from the environment variable ``OPENAQ_API_KEY``, which
is loaded from the ``.env`` file in the project root when present.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_OPENAQ_DIR: Path = DATA_DIR / "raw" / "openaq"
RAW_WEATHER_DIR: Path = DATA_DIR / "raw" / "weather"
PROCESSED_DIR: Path = DATA_DIR / "processed"
LIVE_DIR: Path = DATA_DIR / "live"
MODELS_DIR: Path = PROJECT_ROOT / "models"

STATIONS_FILE: Path = DATA_DIR / "stations.csv"
DATASET_FILE: Path = PROCESSED_DIR / "hourly.csv"
METRICS_FILE: Path = MODELS_DIR / "metrics.csv"
ESTIMATE_CHECK_FILE: Path = MODELS_DIR / "estimate_check.csv"
LIVE_OBSERVATIONS_FILE: Path = LIVE_DIR / "observations.csv"
LIVE_FORECAST_FILE: Path = LIVE_DIR / "forecast.csv"
LIVE_STATUS_FILE: Path = LIVE_DIR / "status.json"

REGION_NAME: str = "Nova Scotia"
"""Display name only. The region's shape is the outline in ``aqmonitor/region.py``."""

PARAMETER: str = "pm25"
PARAMETER_ID: int = 2
"""OpenAQ's id for PM2.5 in µg/m³."""

HISTORY_START: str = "2022-01-01"
"""First day of training history. Open-Meteo's archived forecasts also begin in 2022."""

FORECAST_HOURS: int = 72
BACKTEST_DAYS: int = 90
LIVE_LOOKBACK_DAYS: int = 10

WEATHER_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
)


def api_key() -> str | None:
    """Return the OpenAQ API key from ``.env`` or the environment, or None when it has not been set."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("OPENAQ_API_KEY", "").strip()
    return key or None

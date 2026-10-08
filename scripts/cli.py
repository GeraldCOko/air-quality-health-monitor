"""Helpers shared by the command-line scripts: loading the project's files with friendly errors, and printing."""

import sys
from pathlib import Path

import pandas as pd

from aqmonitor import config, openaq, places


def say(message: str) -> None:
    """Print a progress line straight away, even when output is being piped."""
    print(message, flush=True)


def stop(message: str) -> None:
    """Print an explanation and end the script with a failure code."""
    say(f"\nCannot continue: {message}")
    sys.exit(1)


def _require(path: Path, made_by: str) -> None:
    if not path.exists():
        stop(f"{path.relative_to(config.PROJECT_ROOT)} does not exist yet. Run  python -m scripts.{made_by}  first.")


def load_stations() -> pd.DataFrame:
    """Read the station list written by ``find_stations`` and add the estimated places such as Antigonish."""
    _require(config.STATIONS_FILE, "find_stations")
    return places.with_places(pd.read_csv(config.STATIONS_FILE))


def load_table() -> pd.DataFrame:
    """Read the hourly training table written by ``build_dataset``."""
    _require(config.DATASET_FILE, "build_dataset")
    return pd.read_csv(config.DATASET_FILE, parse_dates=["time"])


def client_or_none() -> openaq.OpenAQClient | None:
    """An OpenAQ API client when a key is configured, otherwise None."""
    key = config.api_key()
    return openaq.OpenAQClient(key) if key else None

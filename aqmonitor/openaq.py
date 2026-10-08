"""
Talk to the two public faces of OpenAQ: the live API (needs a free key) and the open data archive (needs none).

OpenAQ gathers readings from government and research air-quality monitors around the world and republishes
them in one format. The Nova Scotia stations reach it through the US EPA's AirNow feed.

The live API, ``https://api.openaq.org/v3``, answers four questions used here:

* ``/locations?bbox=...``          which stations sit inside a box on the map,
* ``/locations/{id}/latest``       the newest reading from each sensor at a station,
* ``/sensors/{id}``                one sensor's details, including when it last reported,
* ``/sensors/{id}/measurements``   the readings from one sensor between two times.

Deviation from the obvious choice: OpenAQ also offers ``/sensors/{id}/hours``, a ready-made hourly summary. It
is not used because it returns nothing for some sensors that are reporting normally (Pictou's PM2.5 sensor in
October 2026, for one), whereas ``/measurements`` returns their readings. The same station is also missing from
the ``/locations`` search although its sensor is live, which is why ``find_stations`` keeps a station it already
knows about for as long as ``/sensors/{id}`` shows it reporting.

Every request carries the key in an ``X-API-Key`` header. The free tier allows 60 requests a minute; when the
server answers 429 (too many requests) the client waits for the time the server asks for and tries again.

The archive, ``https://openaq-data-archive.s3.amazonaws.com``, holds one small gzipped CSV per station per day
under ``records/csv.gz/locationid={id}/year={yyyy}/month={mm}/``. It trails real time by several days, which is
why training history comes from the archive and the last few days come from the API.

A timestamp in either source marks the END of the hour the reading covers.

References
----------
OpenAQ API v3 documentation: https://docs.openaq.org
OpenAQ data archive on AWS:  https://docs.openaq.org/aws/about
"""

import threading
import time
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API_URL: str = "https://api.openaq.org/v3"
ARCHIVE_URL: str = "https://openaq-data-archive.s3.amazonaws.com"

_PAGE_SIZE: int = 1000
_TIMEOUT_SECONDS: int = 60
_MAX_ATTEMPTS: int = 4
_S3_NAMESPACE: dict[str, str] = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
_THREAD: threading.local = threading.local()
"""Each download thread keeps its own connection open, which is far faster than reconnecting per file."""

STATION_COLUMNS: list[str] = [
    "station_id",
    "name",
    "locality",
    "latitude",
    "longitude",
    "timezone",
    "provider",
    "sensor_id",
    "reference_grade",
    "first_utc",
    "last_utc",
]


class OpenAQError(RuntimeError):
    """Raised when the OpenAQ API refuses a request or keeps failing."""


class OpenAQClient:
    """A small client for the OpenAQ v3 API that pages through results and respects the rate limit."""

    def __init__(self, key: str, session: requests.Session | None = None) -> None:
        if not key:
            raise OpenAQError("No OpenAQ API key. Paste yours into the .env file (see README, Setup).")
        self._session = session or requests.Session()
        self._headers = {"X-API-Key": key, "Accept": "application/json"}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in range(_MAX_ATTEMPTS):
            response = self._session.get(
                f"{API_URL}{path}", params=params, headers=self._headers, timeout=_TIMEOUT_SECONDS
            )
            if response.status_code == 429:
                time.sleep(float(response.headers.get("x-ratelimit-reset", 60)) + 1.0)
                continue
            if response.status_code in (401, 403):
                raise OpenAQError("OpenAQ rejected the API key. Check the key in your .env file.")
            if response.status_code >= 500:
                time.sleep(2.0 * (attempt + 1))
                continue
            if response.status_code != 200:
                raise OpenAQError(f"OpenAQ answered {response.status_code} for {path}: {response.text[:200]}")
            return response.json()
        raise OpenAQError(f"OpenAQ kept failing for {path} after {_MAX_ATTEMPTS} attempts.")

    def _get_all(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        page = 1
        while True:
            batch = self._get(path, {**params, "limit": _PAGE_SIZE, "page": page}).get("results", [])
            results.extend(batch)
            if len(batch) < _PAGE_SIZE:
                return results
            page += 1

    def locations(self, bbox: tuple[float, float, float, float], parameter_id: int) -> list[dict[str, Any]]:
        """Return every station inside ``bbox`` (west, south, east, north) that measures the parameter."""
        box = ",".join(f"{edge:.4f}" for edge in bbox)
        return self._get_all("/locations", {"bbox": box, "parameters_id": parameter_id})

    def latest(self, station_id: int) -> list[dict[str, Any]]:
        """Return the newest reading from each sensor at a station."""
        return self._get(f"/locations/{station_id}/latest").get("results", [])

    def sensor_readings(self, sensor_id: int, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
        """Return the readings from one sensor between two UTC times."""
        window = {"datetime_from": _iso(start), "datetime_to": _iso(end)}
        return self._get_all(f"/sensors/{sensor_id}/measurements", window)

    def sensor_last_seen(self, sensor_id: int) -> pd.Timestamp | None:
        """Return the UTC time of a sensor's newest reading, or None when the sensor is unknown or empty."""
        try:
            results = self._get(f"/sensors/{sensor_id}").get("results", [])
        except OpenAQError:
            return None
        newest = _nested(results[0], "datetimeLast", "utc") if results else None
        return pd.Timestamp(newest).tz_localize(None) if newest else None


def _iso(moment: pd.Timestamp) -> str:
    return pd.Timestamp(moment).strftime("%Y-%m-%dT%H:%M:%SZ")


def _nested(record: dict[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def stations_table(locations: list[dict[str, Any]], parameter_id: int) -> pd.DataFrame:
    """
    Flatten the API's nested station records into one row per fixed station.

    Mobile stations are skipped, as are stations with no sensor for the parameter. Where a station has more than
    one such sensor the first is used. ``reference_grade`` is True for government-standard monitors and False for
    low-cost community sensors, which are less accurate.
    """
    rows: list[dict[str, Any]] = []
    for location in locations:
        sensors = [s for s in location.get("sensors", []) if _nested(s, "parameter", "id") == parameter_id]
        if location.get("isMobile") or not sensors:
            continue
        rows.append(
            {
                "station_id": location["id"],
                "name": (location.get("name") or f"Station {location['id']}").title(),
                "locality": location.get("locality") or "",
                "latitude": _nested(location, "coordinates", "latitude"),
                "longitude": _nested(location, "coordinates", "longitude"),
                "timezone": location.get("timezone") or "UTC",
                "provider": _nested(location, "provider", "name") or "",
                "sensor_id": sensors[0]["id"],
                "reference_grade": bool(location.get("isMonitor")),
                "first_utc": _nested(location, "datetimeFirst", "utc"),
                "last_utc": _nested(location, "datetimeLast", "utc"),
            }
        )
    return pd.DataFrame(rows, columns=STATION_COLUMNS).sort_values("name").reset_index(drop=True)


def readings_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn ``/sensors/{id}/measurements`` records into a frame of period-ending UTC ``time`` and ``value``."""
    rows = [
        {"time": _nested(record, "period", "datetimeTo", "utc"), "value": record.get("value")} for record in records
    ]
    frame = pd.DataFrame(rows, columns=["time", "value"])
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def archive_keys(station_id: int, year: int, session: requests.Session | None = None) -> list[str]:
    """List the archive's file names (S3 keys) for one station and one year."""
    http = session or requests.Session()
    prefix = f"records/csv.gz/locationid={station_id}/year={year}/"
    keys: list[str] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        response = http.get(ARCHIVE_URL, params=params, timeout=_TIMEOUT_SECONDS)
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        keys.extend(node.text for node in root.findall("s3:Contents/s3:Key", _S3_NAMESPACE) if node.text)
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=_S3_NAMESPACE)
        if not token:
            return sorted(keys)


def archive_local_path(key: str, folder: Path) -> Path:
    """Where an archive file is saved on disk: ``<folder>/<station id>/<file name>``."""
    name = key.rsplit("/", 1)[-1]
    station_id = name.split("-")[1]
    return folder / station_id / name


def _thread_session() -> requests.Session:
    if not hasattr(_THREAD, "session"):
        _THREAD.session = requests.Session()
    return _THREAD.session


def _download_one(key: str, folder: Path) -> bool:
    target = archive_local_path(key, folder)
    if target.exists() and target.stat().st_size > 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = _thread_session().get(f"{ARCHIVE_URL}/{key}", timeout=_TIMEOUT_SECONDS)
            response.raise_for_status()
            partial = target.with_suffix(".part")
            partial.write_bytes(response.content)
            partial.replace(target)
            return True
        except requests.RequestException:
            time.sleep(1.5 * (attempt + 1))
    raise OpenAQError(f"Could not download {key} from the OpenAQ archive.")


def download_archive(keys: list[str], folder: Path, workers: int = 16) -> int:
    """Download archive files that are not on disk yet and return how many new files were fetched."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(lambda key: _download_one(key, folder), keys))

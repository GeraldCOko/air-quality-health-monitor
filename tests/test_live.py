from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from aqmonitor import config, live, openaq, places

_NOW = pd.Timestamp("2026-10-07 12:00")


class _Client:
    def __init__(self, hours: list[dict[str, Any]] | None = None, fail: bool = False) -> None:
        self.hours = hours or []
        self.fail = fail

    def sensor_readings(self, sensor_id: int, start: pd.Timestamp, end: pd.Timestamp) -> list[dict[str, Any]]:
        if self.fail:
            raise openaq.OpenAQError("boom")
        return self.hours


def _record(time: str, value: float) -> dict[str, Any]:
    return {"value": value, "period": {"datetimeTo": {"utc": time}}}


def _setup() -> tuple[pd.DataFrame, pd.DataFrame]:
    stations = pd.DataFrame({"station_id": [774], "name": ["Lake Major"], "sensor_id": [25148]})
    times = pd.date_range("2026-10-01 01:00", "2026-10-03 00:00", freq="h")
    table = pd.DataFrame({"station_id": 774, "time": times, config.PARAMETER: 5.0})
    return stations, table


def test_model_path_is_named_after_station() -> None:
    assert live.model_path(774, Path("models")) == Path("models") / "station_774.json"


def test_gather_readings_without_client_uses_archive_only() -> None:
    stations, table = _setup()
    readings = live.gather_readings(stations, table, None, _NOW, [])
    assert readings["time"].max() == pd.Timestamp("2026-10-03 00:00")


def test_gather_readings_adds_newer_live_hours() -> None:
    stations, table = _setup()
    client = _Client([_record("2026-10-07T11:00:00Z", 7.0)])
    readings = live.gather_readings(stations, table, client, _NOW, [])
    assert readings["time"].max() == pd.Timestamp("2026-10-07 11:00")


def test_gather_readings_live_value_wins_over_archive_for_same_hour() -> None:
    stations, table = _setup()
    client = _Client([_record("2026-10-02T12:00:00Z", 99.0)])
    readings = live.gather_readings(stations, table, client, _NOW, [])
    assert readings.loc[readings["time"] == pd.Timestamp("2026-10-02 12:00"), config.PARAMETER].iloc[0] == 99.0


def test_gather_readings_drops_archive_rows_older_than_lookback() -> None:
    stations, table = _setup()
    old = table.assign(time=table["time"] - pd.Timedelta(days=60))
    assert live.gather_readings(stations, old, None, _NOW, []).empty


def test_gather_readings_ignores_hours_in_the_future() -> None:
    stations, table = _setup()
    client = _Client([_record("2026-10-09T00:00:00Z", 7.0)])
    assert live.gather_readings(stations, table, client, _NOW, [])["time"].max() <= _NOW


def test_gather_readings_api_failure_is_recorded_not_raised() -> None:
    stations, table = _setup()
    problems: list[str] = []
    live.gather_readings(stations, table, _Client(fail=True), _NOW, problems)
    assert problems == ["Lake Major: could not fetch live readings (boom)"]


def test_gather_readings_adds_estimate_for_place_without_monitor() -> None:
    stations, table = _setup()
    listed = places.with_places(stations.assign(latitude=44.72, longitude=-63.48, timezone="America/Halifax"))
    readings = live.gather_readings(listed, table, None, _NOW, [])
    estimate = readings[readings["station_id"] == places.PLACES[0].station_id]
    assert estimate[config.PARAMETER].dropna().unique().tolist() == [5.0]


def test_gather_readings_rebuilds_estimate_instead_of_reusing_stale_table_rows() -> None:
    stations, table = _setup()
    listed = places.with_places(stations.assign(latitude=44.72, longitude=-63.48, timezone="America/Halifax"))
    stale = table.assign(station_id=places.PLACES[0].station_id, **{config.PARAMETER: 400.0})
    readings = live.gather_readings(listed, pd.concat([table, stale]), None, _NOW, [])
    assert readings[config.PARAMETER].max() == 5.0


def test_gather_readings_does_not_ask_the_api_about_estimated_places() -> None:
    stations, table = _setup()
    listed = places.with_places(stations.assign(latitude=44.72, longitude=-63.48, timezone="America/Halifax"))
    problems: list[str] = []
    live.gather_readings(listed, table, _Client(fail=True), _NOW, problems)
    assert len(problems) == 1


def test_latest_readings_picks_newest_non_missing_per_station() -> None:
    readings = pd.DataFrame(
        {
            "station_id": [774, 774, 774, 776],
            "time": pd.to_datetime(["2026-10-07 09:00", "2026-10-07 10:00", "2026-10-07 11:00", "2026-10-07 08:00"]),
            config.PARAMETER: [3.0, 4.0, None, 9.0],
        }
    )
    latest = live.latest_readings(readings).set_index("station_id")[config.PARAMETER].to_dict()
    assert latest == {774: 4.0, 776: 9.0}


def test_refresh_station_without_model_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stations, table = _setup()
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(live, "model_path", lambda station_id: tmp_path / "models" / f"station_{station_id}.json")
    monkeypatch.setattr(config, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setattr(config, "LIVE_OBSERVATIONS_FILE", tmp_path / "live" / "observations.csv")
    monkeypatch.setattr(config, "LIVE_FORECAST_FILE", tmp_path / "live" / "forecast.csv")
    monkeypatch.setattr(config, "LIVE_STATUS_FILE", tmp_path / "live" / "status.json")
    status = live.refresh(stations, table, None)
    assert status["problems"] == ["Lake Major: no trained model, so no forecast"]

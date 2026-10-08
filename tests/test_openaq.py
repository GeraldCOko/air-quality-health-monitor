from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from aqmonitor import openaq

_PM25 = 2


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None, content: bytes = b"") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.content = content
        self.text = str(payload)
        self.headers: dict[str, str] = {"x-ratelimit-reset": "0"}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def _location(station_id: int, name: str, mobile: bool = False, parameter_id: int = _PM25) -> dict[str, Any]:
    return {
        "id": station_id,
        "name": name,
        "locality": None,
        "timezone": "America/Halifax",
        "isMobile": mobile,
        "provider": {"id": 119, "name": "AirNow"},
        "coordinates": {"latitude": 44.7, "longitude": -63.5},
        "sensors": [{"id": station_id * 10, "parameter": {"id": parameter_id, "name": "x"}}],
        "datetimeFirst": {"utc": "2016-03-06T19:00:00Z"},
        "datetimeLast": {"utc": "2026-10-07T20:00:00Z"},
    }


def test_client_empty_key_raises() -> None:
    with pytest.raises(openaq.OpenAQError):
        openaq.OpenAQClient("")


def test_client_sends_key_in_header() -> None:
    session = _Session([_Response(200, {"results": []})])
    openaq.OpenAQClient("secret", session).latest(774)
    assert session.calls[0]["headers"]["X-API-Key"] == "secret"


def test_client_rejected_key_raises_with_helpful_message() -> None:
    session = _Session([_Response(401)])
    with pytest.raises(openaq.OpenAQError, match="rejected the API key"):
        openaq.OpenAQClient("bad", session).latest(774)


def test_client_rate_limited_request_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openaq.time, "sleep", lambda seconds: None)
    session = _Session([_Response(429), _Response(200, {"results": [{"value": 5.0}]})])
    assert openaq.OpenAQClient("key", session).latest(774) == [{"value": 5.0}]


def test_client_full_page_requests_next_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(openaq, "_PAGE_SIZE", 2)
    session = _Session([_Response(200, {"results": [{}, {}]}), _Response(200, {"results": [{}]})])
    records = openaq.OpenAQClient("key", session).sensor_readings(
        1, pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-02")
    )
    assert (len(records), session.calls[1]["params"]["page"]) == (3, 2)


def test_client_locations_formats_bounding_box() -> None:
    session = _Session([_Response(200, {"results": []})])
    openaq.OpenAQClient("key", session).locations((-66.5, 43.3, -59.6, 47.1), _PM25)
    assert session.calls[0]["params"]["bbox"] == "-66.5000,43.3000,-59.6000,47.1000"


def test_client_sensor_readings_uses_measurements_endpoint() -> None:
    session = _Session([_Response(200, {"results": []})])
    openaq.OpenAQClient("key", session).sensor_readings(25335, pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-02"))
    assert session.calls[0]["url"].endswith("/sensors/25335/measurements")


def test_client_sensor_last_seen_returns_newest_reading_time() -> None:
    session = _Session([_Response(200, {"results": [{"datetimeLast": {"utc": "2026-10-07T23:00:00Z"}}]})])
    assert openaq.OpenAQClient("key", session).sensor_last_seen(25335) == pd.Timestamp("2026-10-07 23:00")


def test_client_sensor_last_seen_unknown_sensor_is_none() -> None:
    assert openaq.OpenAQClient("key", _Session([_Response(404)])).sensor_last_seen(1) is None


def test_stations_table_flags_low_cost_sensor_as_not_reference_grade() -> None:
    record = _location(3286476, "Treehouse Village") | {"isMonitor": False}
    assert not openaq.stations_table([record], _PM25).loc[0, "reference_grade"]


def test_stations_table_flags_government_monitor_as_reference_grade() -> None:
    record = _location(774, "Lake Major") | {"isMonitor": True}
    assert openaq.stations_table([record], _PM25).loc[0, "reference_grade"]


def test_stations_table_keeps_fixed_station_with_parameter() -> None:
    table = openaq.stations_table([_location(774, "Lake Major")], _PM25)
    assert table.loc[0, ["station_id", "sensor_id", "provider"]].tolist() == [774, 7740, "AirNow"]


def test_stations_table_title_cases_shouting_names() -> None:
    assert openaq.stations_table([_location(776, "PICTOU")], _PM25).loc[0, "name"] == "Pictou"


def test_stations_table_skips_mobile_station() -> None:
    assert openaq.stations_table([_location(1, "Van", mobile=True)], _PM25).empty


def test_stations_table_skips_station_without_parameter() -> None:
    assert openaq.stations_table([_location(1, "Ozone only", parameter_id=10)], _PM25).empty


def test_readings_table_uses_hour_ending_time_without_timezone() -> None:
    records = [{"value": 4.2, "period": {"datetimeTo": {"utc": "2026-10-07T13:00:00Z"}}}]
    assert openaq.readings_table(records).loc[0, "time"] == pd.Timestamp("2026-10-07 13:00")


def test_readings_table_sorts_by_time() -> None:
    records = [
        {"value": 2.0, "period": {"datetimeTo": {"utc": "2026-10-07T14:00:00Z"}}},
        {"value": 1.0, "period": {"datetimeTo": {"utc": "2026-10-07T13:00:00Z"}}},
    ]
    assert openaq.readings_table(records)["value"].tolist() == [1.0, 2.0]


def test_readings_table_no_records_gives_empty_frame_with_columns() -> None:
    assert list(openaq.readings_table([]).columns) == ["time", "value"]


def test_archive_keys_follows_continuation_token() -> None:
    namespace = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'
    first = f"<ListBucketResult {namespace}><Contents><Key>a/location-774-20260102.csv.gz</Key></Contents>"
    first += "<NextContinuationToken>more</NextContinuationToken></ListBucketResult>"
    second = f"<ListBucketResult {namespace}><Contents><Key>a/location-774-20260101.csv.gz</Key></Contents>"
    second += "</ListBucketResult>"
    session = _Session([_Response(200, content=first.encode()), _Response(200, content=second.encode())])
    keys = openaq.archive_keys(774, 2026, session)
    assert keys == ["a/location-774-20260101.csv.gz", "a/location-774-20260102.csv.gz"]


def test_archive_local_path_groups_files_by_station() -> None:
    key = "records/csv.gz/locationid=774/year=2026/month=09/location-774-20260915.csv.gz"
    assert openaq.archive_local_path(key, Path("raw")) == Path("raw") / "774" / "location-774-20260915.csv.gz"


def test_download_archive_skips_files_already_on_disk(tmp_path: Path) -> None:
    key = "records/csv.gz/locationid=774/year=2026/month=09/location-774-20260915.csv.gz"
    target = openaq.archive_local_path(key, tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"already here")
    assert openaq.download_archive([key], tmp_path) == 0

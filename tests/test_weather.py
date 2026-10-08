from typing import Any

import pandas as pd
import pytest

from aqmonitor import weather


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = "error text"

    def json(self) -> dict[str, Any]:
        return self._payload


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def _payload(times: list[str]) -> dict[str, Any]:
    return {"hourly": {"time": times, "temperature_2m": [1.0] * len(times), "wind_speed_10m": [9.0] * len(times)}}


def test_to_frame_parses_time_column() -> None:
    frame = weather.to_frame(_payload(["2026-10-07T00:00", "2026-10-07T01:00"]))
    assert frame["time"].tolist() == [pd.Timestamp("2026-10-07 00:00"), pd.Timestamp("2026-10-07 01:00")]


def test_fetch_history_makes_one_request_per_year() -> None:
    session = _Session([_Response(200, _payload(["2024-12-31T23:00"])), _Response(200, _payload(["2025-01-01T00:00"]))])
    weather.fetch_history(44.7, -63.5, "2024-06-01", "2025-03-01", session)
    assert len(session.calls) == 2


def test_fetch_history_clips_first_and_last_year_to_requested_dates() -> None:
    session = _Session([_Response(200, _payload(["2024-12-31T23:00"])), _Response(200, _payload(["2025-01-01T00:00"]))])
    weather.fetch_history(44.7, -63.5, "2024-06-01", "2025-03-01", session)
    spans = [(call["params"]["start_date"], call["params"]["end_date"]) for call in session.calls]
    assert spans == [("2024-06-01", "2024-12-31"), ("2025-01-01", "2025-03-01")]


def test_fetch_history_joins_years_in_time_order() -> None:
    session = _Session([_Response(200, _payload(["2024-12-31T23:00"])), _Response(200, _payload(["2025-01-01T00:00"]))])
    frame = weather.fetch_history(44.7, -63.5, "2024-06-01", "2025-03-01", session)
    assert frame["time"].is_monotonic_increasing and len(frame) == 2


def test_fetch_history_asks_for_utc() -> None:
    session = _Session([_Response(200, _payload(["2025-01-01T00:00"]))])
    weather.fetch_history(44.7, -63.5, "2025-01-01", "2025-01-02", session)
    assert session.calls[0]["params"]["timezone"] == "UTC"


def test_fetch_forecast_passes_past_and_future_days() -> None:
    session = _Session([_Response(200, _payload(["2026-10-07T00:00"]))])
    weather.fetch_forecast(44.7, -63.5, 10, 5, session)
    assert (session.calls[0]["params"]["past_days"], session.calls[0]["params"]["forecast_days"]) == (10, 5)


def test_fetch_forecast_server_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(weather.time, "sleep", lambda seconds: None)
    session = _Session([_Response(502), _Response(200, _payload(["2026-10-07T00:00"]))])
    assert len(weather.fetch_forecast(44.7, -63.5, 1, 1, session)) == 1


def test_fetch_forecast_bad_request_raises() -> None:
    with pytest.raises(RuntimeError, match="400"):
        weather.fetch_forecast(44.7, -63.5, 1, 1, _Session([_Response(400)]))

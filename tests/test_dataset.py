import gzip
from pathlib import Path

import pandas as pd

from aqmonitor import dataset

_HEADER = '"location_id","sensors_id","location","datetime","lat","lon","parameter","units","value"\n'


def _archive_file(folder: Path, rows: list[str]) -> Path:
    path = folder / "location-774-20260915.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(_HEADER + "\n".join(rows) + "\n")
    return path


def _readings(values: list[float | None], start: str = "2026-01-01 01:00") -> pd.DataFrame:
    times = pd.date_range(start, periods=len(values), freq="h")
    frame = pd.DataFrame({"station_id": 774, "time": times, "value": values})
    return frame.dropna(subset=["value"])


def test_read_archive_converts_local_time_to_utc(tmp_path: Path) -> None:
    row = '774,25148,"Lake Major-774","2026-09-15T01:00:00-03:00","44.7","-63.5","pm25","µg/m³","6.0"'
    frame = dataset.read_archive([_archive_file(tmp_path, [row])], "pm25")
    assert frame.loc[0, "time"] == pd.Timestamp("2026-09-15 04:00")


def test_read_archive_keeps_only_requested_parameter(tmp_path: Path) -> None:
    rows = [
        '774,25148,"Lake Major-774","2026-09-15T01:00:00-03:00","44.7","-63.5","pm25","µg/m³","6.0"',
        '774,1330,"Lake Major-774","2026-09-15T01:00:00-03:00","44.7","-63.5","o3","ppm","0.03"',
    ]
    assert dataset.read_archive([_archive_file(tmp_path, rows)], "pm25")["value"].tolist() == [6.0]


def test_read_archive_no_files_gives_empty_frame_with_columns() -> None:
    assert list(dataset.read_archive([], "pm25").columns) == ["station_id", "time", "value"]


def test_clean_hourly_drops_missing_value_sentinel() -> None:
    cleaned = dataset.clean_hourly(_readings([5.0, -999.0, 5.0]))
    assert cleaned["value"].min() >= 0.0


def test_clean_hourly_sets_small_negative_reading_to_zero() -> None:
    assert dataset.clean_hourly(_readings([5.0, -2.0, 5.0]))["value"].tolist() == [5.0, 0.0, 5.0]


def test_clean_hourly_fills_short_gap_with_straight_line() -> None:
    assert dataset.clean_hourly(_readings([2.0, None, None, 8.0]))["value"].tolist() == [2.0, 4.0, 6.0, 8.0]


def test_clean_hourly_leaves_long_gap_empty() -> None:
    cleaned = dataset.clean_hourly(_readings([2.0] + [None] * 10 + [8.0]))
    assert cleaned["value"].isna().sum() == 10


def test_clean_hourly_lays_station_on_complete_hourly_grid() -> None:
    cleaned = dataset.clean_hourly(_readings([2.0] + [None] * 10 + [8.0]))
    assert cleaned["time"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_clean_hourly_averages_duplicate_hours() -> None:
    doubled = pd.concat([_readings([4.0]), _readings([6.0])])
    assert dataset.clean_hourly(doubled)["value"].tolist() == [5.0]


def test_clean_hourly_keeps_stations_separate() -> None:
    other = _readings([1.0, 1.0]).assign(station_id=776)
    cleaned = dataset.clean_hourly(pd.concat([_readings([9.0, 9.0]), other]))
    assert cleaned.groupby("station_id")["value"].mean().to_dict() == {774: 9.0, 776: 1.0}


def test_join_weather_attaches_weather_for_matching_hour() -> None:
    air = dataset.clean_hourly(_readings([3.0, 4.0]))
    weather = pd.DataFrame({"time": air["time"], "wind_speed_10m": [10.0, 20.0]})
    assert dataset.join_weather(air, {774: weather}, "pm25")["wind_speed_10m"].tolist() == [10.0, 20.0]


def test_join_weather_renames_value_to_parameter() -> None:
    air = dataset.clean_hourly(_readings([3.0, 4.0]))
    weather = pd.DataFrame({"time": air["time"], "wind_speed_10m": [10.0, 20.0]})
    assert dataset.join_weather(air, {774: weather}, "pm25")["pm25"].tolist() == [3.0, 4.0]


def test_join_weather_drops_station_without_weather() -> None:
    air = dataset.clean_hourly(_readings([3.0, 4.0]))
    assert dataset.join_weather(air, {}, "pm25").empty


def test_coverage_reports_share_of_hours_with_reading() -> None:
    table = dataset.clean_hourly(_readings([2.0] + [None] * 8 + [8.0])).rename(columns={"value": "pm25"})
    assert dataset.coverage(table, "pm25").loc[0, "share_with_reading"] == 0.2

import numpy as np
import pandas as pd

from aqmonitor import insights


def _forecast(values: list[float]) -> pd.DataFrame:
    times = pd.date_range("2026-10-07 01:00", periods=len(values), freq="h")
    return pd.DataFrame({"time": times, "hours_ahead": range(1, len(values) + 1), "pm25": values})


def _history(**columns: list[float]) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_outlook_seventy_two_hours_gives_three_blocks() -> None:
    assert len(insights.outlook(_forecast([5.0] * 72), "pm25")) == 3


def test_outlook_names_blocks_by_distance_from_now() -> None:
    periods = insights.outlook(_forecast([5.0] * 72), "pm25")["period"].tolist()
    assert periods == ["Next 24 hours", "24-48 hours from now", "48-72 hours from now"]


def test_outlook_category_follows_block_average_not_peak() -> None:
    values = [2.0] * 23 + [40.0] + [2.0] * 48
    assert insights.outlook(_forecast(values), "pm25").loc[0, "category"] == "Good"


def test_outlook_reports_worst_hour_and_when() -> None:
    values = [2.0] * 23 + [40.0] + [2.0] * 48
    first = insights.outlook(_forecast(values), "pm25").iloc[0]
    assert (first["peak"], first["peak_time"]) == (40.0, pd.Timestamp("2026-10-08 00:00"))


def test_outlook_dirty_second_day_is_flagged() -> None:
    values = [2.0] * 24 + [60.0] * 24 + [2.0] * 24
    assert insights.outlook(_forecast(values), "pm25")["category"].tolist() == ["Good", "Unhealthy", "Good"]


def test_by_wind_speed_averages_within_band() -> None:
    frame = _history(wind_speed_10m=[1.0, 4.0, 25.0], pm25=[10.0, 20.0, 3.0])
    summary = insights.by_wind_speed(frame, "pm25").set_index("wind_speed")
    assert (summary.loc["0-5", "average"], summary.loc["20-30", "average"]) == (15.0, 3.0)


def test_by_wind_speed_counts_hours_per_band() -> None:
    frame = _history(wind_speed_10m=[1.0, 4.0, 25.0], pm25=[10.0, 20.0, 3.0])
    assert insights.by_wind_speed(frame, "pm25").set_index("wind_speed")["hours"].sum() == 3


def test_by_wind_direction_wraps_north_across_zero_degrees() -> None:
    frame = _history(wind_direction_10m=[350.0, 10.0], pm25=[4.0, 6.0])
    summary = insights.by_wind_direction(frame, "pm25").set_index("wind_from")
    assert (summary.loc["N", "average"], summary.loc["N", "hours"]) == (5.0, 2)


def test_by_wind_direction_lists_all_eight_compass_points() -> None:
    frame = _history(wind_direction_10m=[90.0], pm25=[4.0])
    assert insights.by_wind_direction(frame, "pm25")["wind_from"].tolist() == [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]


def test_by_wind_direction_skips_hours_without_direction() -> None:
    frame = _history(wind_direction_10m=[90.0, np.nan], pm25=[4.0, 100.0])
    assert insights.by_wind_direction(frame, "pm25")["hours"].sum() == 1


def test_by_rain_separates_dry_and_wet_hours() -> None:
    frame = _history(precipitation=[0.0, 0.0, 2.0], pm25=[8.0, 10.0, 3.0])
    summary = insights.by_rain(frame, "pm25").set_index("weather")["average"].to_dict()
    assert summary == {"Dry hours": 9.0, "Wet hours": 3.0}


def test_by_rain_ignores_hours_without_a_reading() -> None:
    frame = _history(precipitation=[0.0, 0.0], pm25=[8.0, np.nan])
    assert insights.by_rain(frame, "pm25").set_index("weather").loc["Dry hours", "hours"] == 1

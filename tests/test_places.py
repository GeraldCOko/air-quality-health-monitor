import numpy as np
import pandas as pd
import pytest

from aqmonitor import places

_PICTOU, _HAWKESBURY, _SYDNEY, _LAKE_MAJOR = 776, 747, 2037, 774
_ANTIGONISH = places.PLACES[0]


def _stations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": [_PICTOU, _HAWKESBURY, _SYDNEY, _LAKE_MAJOR],
            "name": ["Pictou", "Port Hawkesbury", "Sydney", "Lake Major"],
            "latitude": [45.681744, 45.614028, 46.142317, 44.7198],
            "longitude": [-62.696575, -61.362122, -60.172575, -63.48076],
            "timezone": "America/Halifax",
        }
    )


def _readings(values: dict[int, list[float | None]]) -> pd.DataFrame:
    frames = [
        pd.DataFrame(
            {
                "station_id": station,
                "time": pd.date_range("2026-01-01 01:00", periods=len(series), freq="h"),
                "value": series,
            }
        )
        for station, series in values.items()
    ]
    return pd.concat(frames, ignore_index=True)


def test_distance_km_same_point_is_zero() -> None:
    assert places.distance_km(45.0, -62.0, 45.0, -62.0) == 0.0


def test_distance_km_one_degree_of_latitude_is_about_111_km() -> None:
    assert places.distance_km(45.0, -62.0, 46.0, -62.0) == pytest.approx(111.2, abs=0.2)


def test_distance_km_is_symmetric() -> None:
    there = places.distance_km(45.62, -61.99, 45.68, -62.70)
    assert places.distance_km(45.68, -62.70, 45.62, -61.99) == pytest.approx(there)


def test_distance_km_antigonish_to_pictou_is_about_55_km() -> None:
    assert places.distance_km(_ANTIGONISH.latitude, _ANTIGONISH.longitude, 45.681744, -62.696575) == pytest.approx(
        55.0, abs=1.5
    )


def test_with_places_lists_antigonish_first() -> None:
    assert places.with_places(_stations()).loc[0, "name"] == "Antigonish (estimate)"


def test_with_places_marks_real_stations_as_monitors() -> None:
    combined = places.with_places(_stations())
    assert combined.loc[combined["station_id"] == _PICTOU, "kind"].item() == places.KIND_MONITOR


def test_monitors_list_without_kind_column_is_all_monitors() -> None:
    assert len(places.monitors(_stations())) == 4


def test_estimates_list_without_kind_column_is_empty() -> None:
    assert places.estimates(_stations()).empty


def test_estimates_returns_only_place_rows() -> None:
    assert places.estimates(places.with_places(_stations()))["station_id"].tolist() == [_ANTIGONISH.station_id]


def test_sources_antigonish_uses_pictou_and_port_hawkesbury() -> None:
    chosen = places.sources(_ANTIGONISH.latitude, _ANTIGONISH.longitude, _stations())
    assert sorted(chosen["station_id"]) == sorted([_PICTOU, _HAWKESBURY])


def test_sources_weights_sum_to_one() -> None:
    chosen = places.sources(_ANTIGONISH.latitude, _ANTIGONISH.longitude, _stations())
    assert chosen["weight"].sum() == pytest.approx(1.0)


def test_sources_nearer_monitor_gets_larger_weight() -> None:
    chosen = places.sources(_ANTIGONISH.latitude, _ANTIGONISH.longitude, _stations())
    assert (
        chosen.loc[0, "weight"] > chosen.loc[1, "weight"]
        and chosen.loc[0, "distance_km"] < chosen.loc[1, "distance_km"]
    )


def test_sources_weights_follow_inverse_square_of_distance() -> None:
    chosen = places.sources(_ANTIGONISH.latitude, _ANTIGONISH.longitude, _stations())
    ratio = (chosen.loc[1, "distance_km"] / chosen.loc[0, "distance_km"]) ** 2
    assert chosen.loc[0, "weight"] / chosen.loc[1, "weight"] == pytest.approx(ratio)


def test_sources_leaves_out_low_cost_sensors() -> None:
    stations = _stations().assign(reference_grade=[True, False, True, True])
    chosen = places.sources(_ANTIGONISH.latitude, _ANTIGONISH.longitude, stations)
    assert _HAWKESBURY not in chosen["station_id"].tolist()


def test_blend_is_weighted_average() -> None:
    result = places.blend(_readings({_PICTOU: [10.0], _HAWKESBURY: [20.0]}), {_PICTOU: 0.25, _HAWKESBURY: 0.75})
    assert result["value"].tolist() == [17.5]


def test_blend_equal_readings_give_same_value() -> None:
    result = places.blend(_readings({_PICTOU: [6.0, 6.0], _HAWKESBURY: [6.0, 6.0]}), {_PICTOU: 0.44, _HAWKESBURY: 0.56})
    assert result["value"].tolist() == pytest.approx([6.0, 6.0])


def test_blend_one_station_missing_uses_the_other_alone() -> None:
    result = places.blend(
        _readings({_PICTOU: [10.0, None], _HAWKESBURY: [20.0, 30.0]}), {_PICTOU: 0.5, _HAWKESBURY: 0.5}
    )
    assert result["value"].tolist() == [15.0, 30.0]


def test_blend_all_stations_missing_stays_missing() -> None:
    result = places.blend(
        _readings({_PICTOU: [10.0, None], _HAWKESBURY: [20.0, None]}), {_PICTOU: 0.5, _HAWKESBURY: 0.5}
    )
    assert np.isnan(result["value"].iloc[1])


def test_blend_ignores_stations_without_a_weight() -> None:
    result = places.blend(_readings({_PICTOU: [10.0], _SYDNEY: [99.0]}), {_PICTOU: 1.0})
    assert result["value"].tolist() == [10.0]


def test_blend_no_matching_readings_gives_empty_frame() -> None:
    assert places.blend(_readings({_SYDNEY: [1.0]}), {_PICTOU: 1.0}).empty


def test_blend_result_is_on_complete_hourly_grid() -> None:
    readings = _readings({_PICTOU: [1.0, 2.0, 3.0, 4.0]})
    result = places.blend(readings[readings["value"] != 2.0], {_PICTOU: 1.0})
    assert len(result) == 4


def test_estimate_readings_gives_rows_under_the_place_id() -> None:
    readings = _readings({_PICTOU: [10.0], _HAWKESBURY: [20.0], _SYDNEY: [99.0], _LAKE_MAJOR: [99.0]})
    result = places.estimate_readings(readings, places.with_places(_stations()))
    assert result["station_id"].tolist() == [_ANTIGONISH.station_id]


def test_estimate_readings_value_lies_between_its_two_sources() -> None:
    readings = _readings({_PICTOU: [10.0], _HAWKESBURY: [20.0], _SYDNEY: [99.0], _LAKE_MAJOR: [99.0]})
    result = places.estimate_readings(readings, places.with_places(_stations()))
    assert 15.0 < result["value"].iloc[0] < 20.0


def test_estimate_readings_without_places_is_empty() -> None:
    assert places.estimate_readings(_readings({_PICTOU: [10.0]}), _stations()).empty


def test_leave_one_out_identical_monitors_have_zero_miss() -> None:
    series = [3.0, 5.0, 4.0, 8.0] * 12
    readings = _readings({_PICTOU: series, _HAWKESBURY: series, _SYDNEY: series, _LAKE_MAJOR: series})
    assert places.leave_one_out(readings, _stations())["mae_hourly"].max() == pytest.approx(0.0)


def test_leave_one_out_gives_one_row_per_monitor() -> None:
    series = [3.0, 5.0, 4.0, 8.0] * 12
    readings = _readings({_PICTOU: series, _HAWKESBURY: series, _SYDNEY: series, _LAKE_MAJOR: series})
    assert sorted(places.leave_one_out(readings, _stations())["name"]) == [
        "Lake Major",
        "Pictou",
        "Port Hawkesbury",
        "Sydney",
    ]


def test_leave_one_out_never_uses_the_hidden_monitor_itself() -> None:
    quiet, loud = [2.0, 3.0] * 24, [50.0, 60.0] * 24
    readings = _readings({_PICTOU: loud, _HAWKESBURY: quiet, _SYDNEY: quiet, _LAKE_MAJOR: quiet})
    scores = places.leave_one_out(readings, _stations()).set_index("name")
    assert scores.loc["Pictou", "mae_hourly"] == pytest.approx(52.5)

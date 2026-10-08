import pandas as pd

from aqmonitor import aqi, mapping

_NOW = pd.Timestamp("2026-10-07 12:00")


def _setup() -> tuple[pd.DataFrame, pd.DataFrame]:
    stations = pd.DataFrame(
        {
            "station_id": [774, 776],
            "name": ["Lake Major", "Pictou <b>"],
            "latitude": [44.72, 45.68],
            "longitude": [-63.48, -62.70],
        }
    )
    latest = pd.DataFrame({"station_id": [774], "time": [_NOW - pd.Timedelta(hours=1)], "pm25": [42.0]})
    return stations, latest


def test_marker_details_fresh_reading_uses_its_category() -> None:
    assert mapping.marker_details(42.0, 1.0)[0].name == "Unhealthy for Sensitive Groups"


def test_marker_details_fresh_reading_text_includes_aqi() -> None:
    assert mapping.marker_details(9.0, 1.0)[1] == "9.0 µg/m³ (AQI 50)"


def test_marker_details_stale_reading_is_shown_as_no_data() -> None:
    assert mapping.marker_details(42.0, mapping.STALE_HOURS + 1.0)[0] is aqi.UNKNOWN


def test_marker_details_missing_reading_is_shown_as_no_data() -> None:
    assert mapping.marker_details(None, None)[0] is aqi.UNKNOWN


def test_legend_html_names_every_category() -> None:
    legend = mapping.legend_html()
    assert all(band.name in legend for band in aqi.CATEGORIES)


def test_build_map_draws_station_in_its_category_colour() -> None:
    stations, latest = _setup()
    rendered = mapping.build_map(stations, latest, _NOW, "pm25").get_root().render()
    assert aqi.category(42.0).colour in rendered


def test_build_map_station_without_reading_is_grey() -> None:
    stations, latest = _setup()
    rendered = mapping.build_map(stations, latest, _NOW, "pm25").get_root().render()
    assert aqi.UNKNOWN.colour in rendered


def test_build_map_estimated_place_gets_dashed_ring() -> None:
    stations, latest = _setup()
    rendered = (
        mapping.build_map(stations.assign(kind=["monitor", "estimate"]), latest, _NOW, "pm25").get_root().render()
    )
    assert rendered.count('"dashArray": "5 5"') == 1


def test_build_map_without_kind_column_draws_no_dashed_ring() -> None:
    stations, latest = _setup()
    rendered = mapping.build_map(stations, latest, _NOW, "pm25").get_root().render()
    assert '"dashArray": "5 5"' not in rendered


def test_legend_html_explains_dashed_ring() -> None:
    assert "estimate" in mapping.legend_html()


def test_build_map_escapes_html_in_station_names() -> None:
    stations, latest = _setup()
    rendered = mapping.build_map(stations, latest, _NOW, "pm25").get_root().render()
    assert "Pictou <b>" not in rendered

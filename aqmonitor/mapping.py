"""
Draw the stations on an interactive map, coloured by how clean or dirty the air is.

Each station becomes a circle whose colour is its AQI category (see ``aqmonitor.aqi``). Hovering shows the station
name and reading; clicking opens a small card with the AQI number, category and the time of the reading. Stations
whose newest reading is older than ``STALE_HOURS`` are drawn grey: an old number in a bright colour would look
like current information.

A place with no monitor of its own (see ``aqmonitor.places``) is drawn with a thick dashed ring, so an estimate
can never be mistaken for a measurement.
"""

import html

import folium
import pandas as pd

from aqmonitor import aqi, places

STALE_HOURS: int = 6

_LEGEND_STYLE: str = "display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 0.85rem; margin: 2px 0 12px 0;"
_DOT_STYLE: str = "display:inline-block;width:11px;height:11px;border-radius:50%;margin-right:6px"


def legend_html() -> str:
    """
    An HTML legend listing every AQI category with its colour and PM2.5 range in µg/m³.

    It is a wrapping row meant to sit under the map, so it can never hide a station.
    """
    rows = [
        f'<span><span style="{_DOT_STYLE};background:{band.colour}"></span>{html.escape(band.name)} '
        f'<span style="opacity:0.65">({band.conc_low:g}-{band.conc_high:g})</span></span>'
        for band in aqi.CATEGORIES
    ]
    rows.append(f'<span><span style="{_DOT_STYLE};background:{aqi.UNKNOWN.colour}"></span>No recent reading</span>')
    rows.append(f'<span><span style="{_DOT_STYLE};border:2px dashed currentColor"></span>Dashed ring = estimate</span>')
    return f'<div style="{_LEGEND_STYLE}">{"".join(rows)}</div>'


def marker_details(value: float | None, age_hours: float | None) -> tuple[aqi.Category, str]:
    """Pick the category and reading text for a marker; a missing or stale reading is shown as no data."""
    if value is None or pd.isna(value) or age_hours is None or age_hours > STALE_HOURS:
        return aqi.UNKNOWN, "no recent reading"
    return aqi.category(value), f"{value:.1f} µg/m³ (AQI {aqi.aqi(value)})"


def build_map(stations: pd.DataFrame, latest: pd.DataFrame, now: pd.Timestamp, parameter: str) -> folium.Map:
    """
    Build the station map.

    ``stations`` needs ``station_id, name, latitude, longitude``; ``latest`` holds each station's newest reading
    as ``station_id, time, <parameter>`` with UTC times; ``now`` is the current UTC time used to judge staleness.
    """
    merged = stations.merge(latest, on="station_id", how="left")
    centre = [merged["latitude"].mean(), merged["longitude"].mean()]
    chart = folium.Map(location=centre, zoom_start=7, tiles="OpenStreetMap", scrollWheelZoom=False)
    for row in merged.itertuples():
        value = getattr(row, parameter)
        age = None if pd.isna(row.time) else (now - row.time) / pd.Timedelta(hours=1)
        band, reading = marker_details(value, age)
        name = html.escape(str(row.name))
        estimated = getattr(row, "kind", places.KIND_MONITOR) == places.KIND_ESTIMATE
        when = "" if pd.isna(row.time) else f"<br>{'Estimate' if estimated else 'Reading'} for the hour ending "
        when += "" if pd.isna(row.time) else f"{row.time:%d %b %H:%M} UTC"
        when += "<br>No monitor here: blended from the nearest monitors." if estimated else ""
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=11,
            color="#111111" if estimated else "#333333",
            weight=3 if estimated else 1,
            dash_array="5 5" if estimated else None,
            fill=True,
            fill_color=band.colour,
            fill_opacity=0.9,
            tooltip=f"{name}: {reading}",
            popup=folium.Popup(f"<b>{name}</b><br>{html.escape(band.name)}<br>{reading}{when}", max_width=260),
        ).add_to(chart)
    if len(merged) > 1:
        south_west = [merged["latitude"].min(), merged["longitude"].min()]
        north_east = [merged["latitude"].max(), merged["longitude"].max()]
        chart.fit_bounds([south_west, north_east], padding=(30, 30))
    return chart

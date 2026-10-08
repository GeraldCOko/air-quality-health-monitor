"""
The dashboard. Start it from the project folder with:  python -m streamlit run app.py

It only reads files the scripts have already made (station list, hourly table, saved models, live forecast), so
it opens quickly. The "Refresh now" button re-runs the live update: newest readings, newest weather forecast and
a fresh 72-hour forecast.
"""

import json

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from aqmonitor import aqi, charts, config, insights, live, mapping, openaq, places

_HISTORY_DAYS_SHOWN: int = 7

st.set_page_config(page_title="Air Quality & Health Monitor", page_icon="🌬️", layout="wide")


@st.cache_data(show_spinner=False)
def _read_csv(path: str, modified: float, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=list(dates))


def _load(path, dates: tuple[str, ...] = ()) -> pd.DataFrame | None:
    return _read_csv(str(path), path.stat().st_mtime, dates) if path.exists() else None


def _local(times: pd.Series, zone: str) -> pd.Series:
    return times.dt.tz_localize("UTC").dt.tz_convert(zone).dt.tz_localize(None)


def _chip(band: aqi.Category) -> str:
    return (
        f'<span style="display:inline-block;width:0.8em;height:0.8em;border-radius:50%;'
        f'background:{band.colour};margin-right:0.4em"></span><b>{band.name}</b>'
    )


def _refresh(stations: pd.DataFrame, table: pd.DataFrame) -> None:
    key = config.api_key()
    with st.spinner("Fetching the newest readings and weather, then forecasting ..."):
        live.refresh(stations, table, openaq.OpenAQClient(key) if key else None)
    st.rerun()


def _setup_needed(missing: str) -> None:
    st.title("Air Quality & Health Monitor")
    st.warning(f"The monitor is not set up yet: {missing}")
    st.markdown(
        "Run these in a terminal opened in the project folder, in order:\n\n"
        "1. `python -m scripts.find_stations`\n"
        "2. `python -m scripts.download_data`\n"
        "3. `python -m scripts.build_dataset`\n"
        "4. `python -m scripts.train`\n"
        "5. `python -m scripts.update_live`\n\n"
        "The README explains each step."
    )
    st.stop()


def _header(status: dict, stations: pd.DataFrame, table: pd.DataFrame) -> None:
    left, right = st.columns([5, 1], vertical_alignment="bottom")
    left.title(f"Air Quality & Health Monitor: {config.REGION_NAME}")
    left.caption(f"Fine particles (PM2.5). Last updated {status['updated_utc']} UTC from {status['source']}.")
    if right.button("Refresh now", type="primary", width="stretch"):
        _refresh(stations, table)
    if config.api_key() is None:
        st.info(
            "No OpenAQ API key yet, so the readings below come from OpenAQ's archive and are several days old. "
            "Paste your free key into the `.env` file in the project folder, then press **Refresh now**."
        )
    if status.get("problems"):
        with st.expander(f"{len(status['problems'])} note(s) from the last update"):
            for problem in status["problems"]:
                st.write(problem)


def _tiles(stations: pd.DataFrame, latest: pd.DataFrame, predicted: pd.DataFrame, now: pd.Timestamp) -> None:
    real = places.monitors(stations)
    fresh = latest[(now - latest["time"]) <= pd.Timedelta(hours=mapping.STALE_HOURS)]
    fresh = fresh.merge(real[["station_id", "name"]], on="station_id")
    first, second, third = st.columns(3)
    first.metric("Monitors reporting now", f"{len(fresh)} of {len(real)}")
    if len(fresh):
        worst = fresh.loc[fresh[config.PARAMETER].idxmax()]
        second.metric(
            "Highest reading right now",
            f"{worst[config.PARAMETER]:.1f} µg/m³",
            f"{worst['name']} · {aqi.category(worst[config.PARAMETER]).name}",
            delta_color="off",
            delta_arrow="off",
        )
    else:
        second.metric("Highest reading right now", "no recent data")
    if len(predicted):
        peak = predicted.loc[predicted[config.PARAMETER].idxmax()]
        name = stations.loc[stations["station_id"] == peak["station_id"], "name"].iloc[0]
        third.metric(
            "Highest forecast, next 72 hours",
            f"{peak[config.PARAMETER]:.1f} µg/m³",
            f"{name} · {aqi.category(peak[config.PARAMETER]).name}",
            delta_color="off",
            delta_arrow="off",
        )


def _health(station: pd.Series, latest: pd.DataFrame, predicted: pd.DataFrame, now: pd.Timestamp) -> None:
    own = latest[latest["station_id"] == station["station_id"]]
    st.subheader("Health outlook")
    if len(own) and (now - own["time"].iloc[0]) <= pd.Timedelta(hours=mapping.STALE_HOURS):
        value = float(own[config.PARAMETER].iloc[0])
        band = aqi.category(value)
        label = "Right now (estimated)" if station["kind"] == places.KIND_ESTIMATE else "Right now"
        st.markdown(f"**{label}:** {_chip(band)}, {value:.1f} µg/m³ (AQI {aqi.aqi(value)})", unsafe_allow_html=True)
        st.write(band.advice)
    elif len(own):
        last = _local(own["time"], station["timezone"]).iloc[0]
        st.markdown(f"**Right now:** no recent reading. The last one was on {last:%d %b at %H:%M} local time.")
    if not len(predicted):
        st.write("No forecast for this station yet.")
        return
    for row in insights.outlook(predicted, config.PARAMETER).itertuples():
        band = aqi.category(row.average)
        st.markdown(
            f"**{row.period}:** {_chip(band)}, average {row.average:.1f} µg/m³"
            f"<br><span style='opacity:0.75'>Worst hour about {row.peak:.1f} µg/m³ "
            f"around {row.peak_time:%a %H:%M}. {row.advice}</span>",
            unsafe_allow_html=True,
        )


def _forecast_tab(station: pd.Series, readings: pd.DataFrame, predicted: pd.DataFrame, now: pd.Timestamp) -> None:
    zone = station["timezone"]
    shown = readings[readings["time"] >= now - pd.Timedelta(days=_HISTORY_DAYS_SHOWN)].copy()
    ahead = predicted.copy()
    shown["time"], ahead["time"] = _local(shown["time"], zone), _local(ahead["time"], zone)
    chart_column, health_column = st.columns([3, 2])
    with chart_column:
        st.subheader(f"Last {_HISTORY_DAYS_SHOWN} days and next {config.FORECAST_HOURS} hours")
        label = "Estimated" if station["kind"] == places.KIND_ESTIMATE else "Measured"
        st.plotly_chart(charts.forecast_figure(shown, ahead, config.PARAMETER, label), width="stretch")
        st.caption(f"Times are local ({zone}). The shaded band is where the model expects 8 readings in 10 to fall.")
    with health_column:
        _health(station, live.latest_readings(readings), ahead, now)
    with st.expander("Show the forecast as a table"):
        table = ahead[["time", config.PARAMETER, "lower", "upper"]].round(1)
        table["category"] = [aqi.category(value).name for value in table[config.PARAMETER]]
        st.dataframe(table, hide_index=True, width="stretch")


def _weather_tab(station: pd.Series, table: pd.DataFrame) -> None:
    own = table[table["station_id"] == station["station_id"]]
    st.subheader("How weather relates to PM2.5 at this station")
    st.caption(
        f"Averages over {own[config.PARAMETER].count():,} hours of data since {own['time'].min():%B %Y}. "
        "These show what tends to go together, not proof of cause."
    )
    first, second, third = st.columns(3)
    with first:
        st.markdown("**Wind speed**")
        figure = charts.bar_figure(insights.by_wind_speed(own, config.PARAMETER), "wind_speed", "Wind speed (km/h)")
        st.plotly_chart(figure, width="stretch")
    with second:
        st.markdown("**Wind direction**")
        figure = charts.bar_figure(insights.by_wind_direction(own, config.PARAMETER), "wind_from", "Wind blowing from")
        st.plotly_chart(figure, width="stretch")
    with third:
        st.markdown("**Rain or snow**")
        st.plotly_chart(charts.bar_figure(insights.by_rain(own, config.PARAMETER), "weather", ""), width="stretch")


def _estimate_note(station: pd.Series, stations: pd.DataFrame, latest: pd.DataFrame, now: pd.Timestamp) -> None:
    chosen = places.sources(station["latitude"], station["longitude"], places.monitors(stations))
    chosen = chosen.merge(latest[["station_id", "time"]], on="station_id", how="left")
    parts = []
    for row in chosen.itertuples():
        age = None if pd.isna(row.time) else (now - row.time) / pd.Timedelta(hours=1)
        state = "reporting now" if age is not None and age <= mapping.STALE_HOURS else "no recent reading"
        parts.append(f"**{row.name}** ({row.distance_km:.0f} km away, weight {row.weight:.0%}, {state})")
    st.info(
        f"There is no air monitor at this location. The numbers for {station['name']} are an estimate blended "
        f"from the nearest monitors: {' and '.join(parts)}. Nearer monitors count for more. When one of them has "
        "no reading, the estimate rests on the other alone. The weather used is for this exact location."
    )


def _estimate_accuracy(checks: pd.DataFrame | None) -> None:
    st.warning(
        "The test above compares the forecast with the *estimate*, because nothing is measured here. It shows how "
        "well the model predicts the blended number, not how close that number is to the real air."
    )
    if checks is None or not len(checks):
        st.write("Run `python -m scripts.check_estimate` to test the estimating method itself.")
        return
    st.markdown(
        "**How good is the estimating method?** Each real monitor was hidden in turn and estimated from its two "
        "nearest neighbours in the same way, then compared with what it really measured. Those neighbours are "
        "further apart than the ones used here, so this is a tough version of the test."
    )
    shown = checks[["name", "source_distance_km", "mae_hourly", "correlation_hourly", "mae_daily", "correlation_daily"]]
    shown = shown.round({"source_distance_km": 0, "mae_hourly": 2, "correlation_hourly": 2, "mae_daily": 2})
    shown = shown.round({"correlation_daily": 2})
    shown.columns = [
        "Hidden monitor",
        "Distance to neighbours (km)",
        "Hourly miss (µg/m³)",
        "Hourly match (0-1)",
        "Daily-average miss (µg/m³)",
        "Daily match (0-1)",
    ]
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("Match is the correlation: 1 means the estimate and the truth rise and fall perfectly together.")


def _accuracy_tab(station: pd.Series, metrics: pd.DataFrame | None) -> None:
    st.subheader("How good is the forecast?")
    own = None if metrics is None else metrics[metrics["station_id"] == station["station_id"]]
    if own is None or not len(own):
        st.write("No test results for this station. Run `python -m scripts.train` to make them.")
        return
    st.markdown(
        f"Before the final model was trained, the last {config.BACKTEST_DAYS} days were hidden from it. It then "
        "forecast those days and was marked against what really happened. Shorter bars are better. The honest "
        "yardstick is the simple guess that the air will stay exactly as it is now."
    )
    st.plotly_chart(charts.accuracy_figure(own), width="stretch")
    shown = own[["lead", "mae_model", "mae_persistence", "mae_prophet", "skill", "coverage", "forecasts"]].copy()
    shown["skill"] = (shown["skill"] * 100).round(0)
    shown["coverage"] = (shown["coverage"] * 100).round(0)
    shown.columns = [
        "How far ahead",
        "Model error (µg/m³)",
        "Stays-as-now error",
        "Prophet-alone error",
        "Better than stays-as-now (%)",
        "Truth inside band (%)",
        "Forecasts tested",
    ]
    st.dataframe(shown.round(2), hide_index=True, width="stretch")
    st.caption(
        "The test used archived weather-model data, which is closer to what happened than a real 3-day weather "
        "forecast. Expect real errors two and three days ahead to be somewhat larger than shown."
    )


def main() -> None:
    """Lay out the page."""
    listed = _load(config.STATIONS_FILE)
    table = _load(config.DATASET_FILE, ("time",))
    if listed is None or table is None:
        _setup_needed("the station list or the training table is missing.")
    stations = places.with_places(listed)
    if not config.LIVE_STATUS_FILE.exists():
        _refresh(stations, table)
    status = json.loads(config.LIVE_STATUS_FILE.read_text(encoding="utf-8"))
    readings = _load(config.LIVE_OBSERVATIONS_FILE, ("time",))
    predicted = _load(config.LIVE_FORECAST_FILE, ("time",))
    now = live.utc_now()
    latest = live.latest_readings(readings)

    _header(status, stations, table)
    _tiles(stations, latest, predicted, now)
    st.subheader("Where the air is clean and where it is not")
    st_folium(
        mapping.build_map(stations, latest, now, config.PARAMETER),
        height=430,
        use_container_width=True,
        returned_objects=[],
    )
    st.markdown(mapping.legend_html(), unsafe_allow_html=True)
    st.caption("Dot colour = PM2.5 in µg/m³ for the newest reading. Hover over a dot for the number.")

    names = stations["name"].tolist()
    station = stations[stations["name"] == st.selectbox("Choose a place", names)].iloc[0]
    estimated = station["kind"] == places.KIND_ESTIMATE
    if estimated:
        _estimate_note(station, stations, latest, now)
    own_readings = readings[readings["station_id"] == station["station_id"]]
    own_forecast = predicted[predicted["station_id"] == station["station_id"]]
    forecast_tab, weather_tab, accuracy_tab = st.tabs(["Forecast and health", "Weather link", "Forecast accuracy"])
    with forecast_tab:
        _forecast_tab(station, own_readings, own_forecast, now)
    with weather_tab:
        _weather_tab(station, table)
    with accuracy_tab:
        _accuracy_tab(station, _load(config.METRICS_FILE))
        if estimated:
            _estimate_accuracy(_load(config.ESTIMATE_CHECK_FILE))
    st.caption(
        "Air-quality readings: OpenAQ (government monitors relayed through AirNow). Weather: Open-Meteo. "
        "Categories follow the US EPA AQI for PM2.5. This is a learning project, not medical advice."
    )


main()

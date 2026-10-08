"""
Small summaries the dashboard shows: the health outlook for the next three days and how weather relates to PM2.5.

**Health outlook.** The 72-hour forecast is cut into three 24-hour blocks. Each block gets the AQI category of
its *average* forecast, because the AQI is defined on a 24-hour average, alongside the single worst hour so a
short spike is not hidden by a clean rest-of-day.

**Weather link.** Three simple tables answer "does the weather matter here?" using the station's whole history:
average PM2.5 by wind-speed band, by the compass direction the wind blows from, and in dry versus wet hours.
These are averages of what happened, not proof of cause: wind direction, for example, also stands in for
"is the air arriving from the ocean or from the continent".
"""

import numpy as np
import pandas as pd

from aqmonitor import aqi

_BLOCK_HOURS: int = 24
_WIND_EDGES: list[float] = [0.0, 5.0, 10.0, 20.0, 30.0, np.inf]
_WIND_LABELS: list[str] = ["0-5", "5-10", "10-20", "20-30", "30+"]
_COMPASS: list[str] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
_WET_THRESHOLD_MM: float = 0.1


def outlook(prediction: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """
    Summarise a forecast with ``time, hours_ahead`` and the parameter column into one row per 24-hour block.

    Returns ``period, average, peak, peak_time, category, colour, advice``.
    """
    rows: list[dict[str, object]] = []
    blocks = (prediction["hours_ahead"] - 1) // _BLOCK_HOURS
    for block, group in prediction.groupby(blocks):
        worst = group.loc[group[parameter].idxmax()]
        band = aqi.category(float(group[parameter].mean()))
        first, last = int(block) * _BLOCK_HOURS, (int(block) + 1) * _BLOCK_HOURS
        rows.append(
            {
                "period": f"Next {last} hours" if block == 0 else f"{first}-{last} hours from now",
                "average": float(group[parameter].mean()),
                "peak": float(worst[parameter]),
                "peak_time": worst["time"],
                "category": band.name,
                "colour": band.colour,
                "advice": band.advice,
            }
        )
    return pd.DataFrame(rows)


def _average(frame: pd.DataFrame, groups: pd.Series, parameter: str, label: str) -> pd.DataFrame:
    known = frame.assign(_group=groups).dropna(subset=[parameter, "_group"])
    summary = known.groupby("_group", observed=False)[parameter].agg(average="mean", hours="count")
    return summary.reset_index().rename(columns={"_group": label})


def by_wind_speed(frame: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Average PM2.5 in each wind-speed band (km/h): ``wind_speed, average, hours``."""
    bands = pd.cut(frame["wind_speed_10m"], _WIND_EDGES, labels=_WIND_LABELS, right=False)
    return _average(frame, bands, parameter, "wind_speed")


def by_wind_direction(frame: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Average PM2.5 by the compass direction the wind blows from: ``wind_from, average, hours``."""
    sector = ((frame["wind_direction_10m"] + 22.5) % 360.0) // 45.0
    names = pd.Categorical.from_codes(sector.fillna(-1).astype(int), categories=_COMPASS)
    return _average(frame, pd.Series(names, index=frame.index), parameter, "wind_from")


def by_rain(frame: pd.DataFrame, parameter: str) -> pd.DataFrame:
    """Average PM2.5 in dry hours and in wet hours (0.1 mm of precipitation or more): ``weather, average, hours``."""
    wet = pd.Series(pd.NA, index=frame.index, dtype="object")
    wet[frame["precipitation"] < _WET_THRESHOLD_MM] = "Dry hours"
    wet[frame["precipitation"] >= _WET_THRESHOLD_MM] = "Wet hours"
    return _average(frame, wet, parameter, "weather")

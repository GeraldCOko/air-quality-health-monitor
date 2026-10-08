"""
Turn a PM2.5 concentration into an Air Quality Index number, a colour, and plain-language health advice.

The index used is the US EPA AQI for PM2.5 with the breakpoints that took effect in May 2024, when the top of the
"Good" band dropped from 12.0 to 9.0 µg/m³. Within a band the index is a straight line between the band's ends:

    AQI = (I_hi - I_lo) / (C_hi - C_lo) * (C - C_lo) + I_lo

where ``C`` is the concentration truncated to one decimal place, ``C_lo``/``C_hi`` are the band's concentration
limits and ``I_lo``/``I_hi`` its index limits.

Deviation from the official method: the EPA defines the AQI on a 24-hour average (or the NowCast weighted average
for current conditions). This project applies the same breakpoints to single hourly values so that the map and the
hour-by-hour forecast use one scale. An hourly reading is noisier than a daily mean, so treat the category for one
hour as a guide and the daily picture as the health-relevant one.

Canada officially reports the AQHI, which also needs ozone and nitrogen dioxide. Not every station here measures
all three, so the PM2.5 AQI is used instead.

References
----------
US EPA (2024). Technical Assistance Document for the Reporting of Daily Air Quality - the Air Quality Index (AQI).
EPA-454/B-24-002.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    """One AQI band: its name, display colour, concentration and index limits, and the advice that goes with it."""

    name: str
    colour: str
    conc_low: float
    conc_high: float
    index_low: int
    index_high: int
    advice: str


CATEGORIES: tuple[Category, ...] = (
    Category("Good", "#00a651", 0.0, 9.0, 0, 50, "Air quality is fine. Enjoy normal outdoor activities."),
    Category(
        "Moderate",
        "#f2c200",
        9.1,
        35.4,
        51,
        100,
        "Fine for most people. If you are unusually sensitive to smoke or dust, take it easier outdoors.",
    ),
    Category(
        "Unhealthy for Sensitive Groups",
        "#ff7e00",
        35.5,
        55.4,
        101,
        150,
        "Children, older adults, pregnant people and anyone with asthma, heart or lung disease should cut back on "
        "long or hard outdoor activity.",
    ),
    Category(
        "Unhealthy",
        "#e0001b",
        55.5,
        125.4,
        151,
        200,
        "Everyone should cut back on hard outdoor activity. Sensitive groups should stay indoors where possible.",
    ),
    Category(
        "Very Unhealthy",
        "#8f3f97",
        125.5,
        225.4,
        201,
        300,
        "Everyone should avoid outdoor exertion. Keep windows closed and run an air cleaner if you have one.",
    ),
    Category(
        "Hazardous",
        "#7e0023",
        225.5,
        325.4,
        301,
        500,
        "Health emergency. Stay indoors, keep indoor air clean and follow local public-health instructions.",
    ),
)

UNKNOWN: Category = Category("No data", "#8a8a8a", math.nan, math.nan, 0, 0, "No recent reading from this station.")

_MAX_INDEX: int = 500


def _truncate(concentration: float) -> float:
    return math.floor(concentration * 10.0 + 1e-9) / 10.0


def category(concentration: float | None) -> Category:
    """Return the AQI band a PM2.5 concentration (µg/m³) falls in, or ``UNKNOWN`` for a missing value."""
    if concentration is None or math.isnan(concentration):
        return UNKNOWN
    conc = _truncate(max(concentration, 0.0))
    for band in CATEGORIES:
        if conc <= band.conc_high:
            return band
    return CATEGORIES[-1]


def aqi(concentration: float | None) -> int | None:
    """Return the AQI number (0-500) for a PM2.5 concentration in µg/m³, or None for a missing value."""
    if concentration is None or math.isnan(concentration):
        return None
    conc = _truncate(max(concentration, 0.0))
    band = category(conc)
    if conc > band.conc_high:
        return _MAX_INDEX
    slope = (band.index_high - band.index_low) / (band.conc_high - band.conc_low)
    return int(round(slope * (conc - band.conc_low) + band.index_low))

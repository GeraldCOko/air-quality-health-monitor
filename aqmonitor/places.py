"""
Estimate air quality for a place that has no monitor, such as Antigonish, from the monitors around it.

Antigonish sits between two government monitors: Pictou, about 55 km to the west, and Port Hawkesbury, about
49 km to the east. Its PM2.5 for any hour is estimated as a weighted average of those two readings, with the
nearer monitor counting for more. This is inverse-distance weighting (Shepard, 1968):

    estimate = sum_i w_i * reading_i,    w_i = (1 / d_i^p) / sum_j (1 / d_j^p)

where ``d_i`` is the great-circle distance to monitor ``i`` and ``p`` is ``POWER``. With p = 2 the weights are
about 0.44 for Pictou and 0.56 for Port Hawkesbury. Only the ``NEAREST`` reference-grade monitors are used:
adding Sydney or Halifax, each roughly 150 km away, would give them weights of a few percent and mostly add noise.

When one of the monitors has no reading for an hour, the estimate for that hour rests on the other one alone
(the weights are rescaled to sum to 1). When none has a reading the estimate is missing.

Everything downstream treats the place like a station: it gets its own row in the hourly table, its own weather
(fetched for its real coordinates), its own Prophet model and its own forecast. The one thing to remember is
that the "readings" are estimates. A forecast for Antigonish is scored against the estimate, not against air
that was actually measured there. ``leave_one_out`` gives an honest feel for how good such estimates are: it
hides each real monitor in turn, estimates it from the others in the same way, and compares with the truth.

Why this works here: PM2.5 in Nova Scotia is mostly regional (ocean air, continental air, wildfire smoke) and
changes slowly across 50 km, so neighbouring monitors rise and fall together. It would work badly next to a
strong local source such as a mill, a busy road or wood-stove smoke on a still winter night, which a monitor
50 km away cannot see.

References
----------
Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. Proceedings of the
1968 ACM National Conference, 517-524.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

KIND_MONITOR: str = "monitor"
KIND_ESTIMATE: str = "estimate"

NEAREST: int = 2
POWER: float = 2.0
_EARTH_RADIUS_KM: float = 6371.0


@dataclass(frozen=True)
class Place:
    """A location with no monitor whose air quality is estimated from nearby monitors."""

    station_id: int
    name: str
    latitude: float
    longitude: float
    timezone: str


PLACES: tuple[Place, ...] = (Place(900001, "Antigonish (estimate)", 45.6220, -61.9933, "America/Halifax"),)
"""Ids from 900001 up are reserved for places, well clear of any real OpenAQ station id in the region."""


def distance_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    """Great-circle distance between two points in kilometres (haversine formula)."""
    phi_a, phi_b = np.radians(lat_a), np.radians(lat_b)
    half_dphi = (phi_b - phi_a) / 2.0
    half_dlambda = np.radians(lon_b - lon_a) / 2.0
    chord = np.sin(half_dphi) ** 2 + np.cos(phi_a) * np.cos(phi_b) * np.sin(half_dlambda) ** 2
    return float(2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(chord)))


def with_places(stations: pd.DataFrame) -> pd.DataFrame:
    """Return the station list with a ``kind`` column and one extra row, listed first, for every place."""
    monitors = stations.assign(kind=KIND_MONITOR)
    rows = [
        {
            "station_id": place.station_id,
            "name": place.name,
            "latitude": place.latitude,
            "longitude": place.longitude,
            "timezone": place.timezone,
            "provider": "Estimated from nearby monitors",
            "kind": KIND_ESTIMATE,
        }
        for place in PLACES
    ]
    return pd.concat([pd.DataFrame(rows), monitors], ignore_index=True)


def monitors(stations: pd.DataFrame) -> pd.DataFrame:
    """The rows that are real monitors; a list without a ``kind`` column is taken to be all monitors."""
    return stations[stations["kind"] == KIND_MONITOR] if "kind" in stations else stations


def estimates(stations: pd.DataFrame) -> pd.DataFrame:
    """The rows that are estimated places; a list without a ``kind`` column has none."""
    return stations[stations["kind"] == KIND_ESTIMATE] if "kind" in stations else stations.iloc[:0]


def sources(latitude: float, longitude: float, candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Pick the monitors an estimate is built from and give each its weight.

    ``candidates`` needs ``station_id, name, latitude, longitude``; if it has a ``reference_grade`` column,
    low-cost sensors are left out. Returns the ``NEAREST`` of them as ``station_id, name, distance_km, weight``,
    nearest first, with weights that sum to 1.
    """
    usable = candidates
    if "reference_grade" in candidates:
        usable = candidates[candidates["reference_grade"].fillna(True).astype(bool)]
    distances = [distance_km(latitude, longitude, row.latitude, row.longitude) for row in usable.itertuples()]
    chosen = usable.assign(distance_km=distances).nsmallest(NEAREST, "distance_km")
    closeness = 1.0 / chosen["distance_km"].clip(lower=0.1) ** POWER
    chosen = chosen.assign(weight=closeness / closeness.sum())
    return chosen[["station_id", "name", "distance_km", "weight"]].reset_index(drop=True)


def blend(readings: pd.DataFrame, weights: dict[int, float]) -> pd.DataFrame:
    """
    Weighted average of several stations' hourly readings.

    ``readings`` holds ``station_id, time, value`` with at most one row per station and hour. Hours where some
    stations are missing use the rest with their weights rescaled; hours where all are missing stay missing.
    Returns ``time, value`` on a complete hourly grid.
    """
    chosen = readings[readings["station_id"].isin(list(weights))]
    if chosen.empty:
        return pd.DataFrame(columns=["time", "value"])
    wide = chosen.pivot(index="time", columns="station_id", values="value").sort_index().asfreq("h")
    share = pd.Series(weights).reindex(wide.columns)
    available = wide.notna().mul(share, axis=1).sum(axis=1)
    total = wide.mul(share, axis=1).sum(axis=1, min_count=1)
    return pd.DataFrame({"time": wide.index, "value": (total / available).to_numpy()})


def estimate_readings(readings: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Build ``station_id, time, value`` rows for every estimated place in ``stations`` from monitor readings."""
    pieces: list[pd.DataFrame] = []
    for place in estimates(stations).itertuples():
        chosen = sources(place.latitude, place.longitude, monitors(stations))
        weights = dict(zip(chosen["station_id"].astype(int), chosen["weight"]))
        pieces.append(blend(readings, weights).assign(station_id=int(place.station_id)))
    if not pieces:
        return pd.DataFrame(columns=["station_id", "time", "value"])
    return pd.concat(pieces, ignore_index=True)[["station_id", "time", "value"]]


def leave_one_out(readings: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """
    Test the estimating method on places where the truth is known.

    Each monitor in turn is hidden and estimated from the other monitors exactly as a place would be; the
    estimate is then compared with what that monitor really measured, hour by hour and as daily averages.
    Returns one row per monitor: ``name, source_distance_km`` (average distance to the monitors it was estimated
    from), ``hours``, ``mae_hourly``, ``correlation_hourly``, ``mae_daily`` and ``correlation_daily``.
    """
    real = monitors(stations)
    rows: list[dict[str, float | str | int]] = []
    for target in real.itertuples():
        others = real[real["station_id"] != target.station_id]
        chosen = sources(target.latitude, target.longitude, others)
        weights = dict(zip(chosen["station_id"].astype(int), chosen["weight"]))
        truth = readings.loc[readings["station_id"] == target.station_id, ["time", "value"]]
        pair = truth.merge(blend(readings, weights), on="time", suffixes=("_true", "_estimate")).dropna()
        if len(pair) < 2:
            continue
        daily = pair.set_index("time").resample("D").mean().dropna()
        rows.append(
            {
                "name": target.name,
                "source_distance_km": float(chosen["distance_km"].mean()),
                "hours": len(pair),
                "mae_hourly": float((pair["value_true"] - pair["value_estimate"]).abs().mean()),
                "correlation_hourly": float(pair["value_true"].corr(pair["value_estimate"])),
                "mae_daily": float((daily["value_true"] - daily["value_estimate"]).abs().mean()),
                "correlation_daily": float(daily["value_true"].corr(daily["value_estimate"])),
            }
        )
    return pd.DataFrame(rows)

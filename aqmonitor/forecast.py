"""
Forecast hourly PM2.5 up to three days ahead with Prophet plus a correction for "where the air is right now".

The model has two parts.

**Part 1 - the usual pattern (Prophet).** Prophet (Taylor & Letham, 2018) describes a time series as a sum of
simple pieces fitted together:

    log(1 + PM2.5) = trend(t) + daily(t) + weekly(t) + yearly(t) + sum_k beta_k * weather_k(t) + noise

The logarithm keeps forecasts positive and stops rare smoke spikes from dominating the fit; an additive effect on
the log scale is a percentage effect on the real scale. The weather terms are temperature, humidity, pressure,
rain, wind speed, and wind split into east-west (``wind_u``) and north-south (``wind_v``) parts so that 359° and
1° count as neighbours. Rain and wind speed enter as log(1 + x) because their first few units matter most.

**Part 2 - carrying today's surprise forward.** Prophet on its own does not know that the air is smoky right now:
it only knows the calendar and the weather. If the latest reading sits above Prophet's curve, the next hours will
almost certainly sit above it too. Let r(t) be the gap, on the log scale, between the reading and Prophet's
curve. The forecast made at time t for h hours ahead is

    forecast(t + h) = prophet(t + h) + c_h * r(t)

where c_h, the "carry" for horizon h, is the least-squares slope of r(t + h) on r(t) over the training period:

    c_h = sum_t r(t) * r(t + h) / sum_t r(t)^2,    clipped to [0, 1].

c_h starts near 1 (the next hour looks like this hour) and fades towards 0 (three days out, today's surprise says
little), so the forecast starts at the latest reading and relaxes into the usual pattern.

**Uncertainty.** The band around the forecast comes from the errors actually made. For each horizon h the errors
e = r(t + h) - c_h * r(t) are collected; their 10th and 90th percentiles become the lower and upper offsets,
giving a band meant to contain the truth 8 times in 10.

Deviations from Taylor & Letham: Prophet's own uncertainty simulation is switched off (``uncertainty_samples=0``)
because the empirical band above already accounts for the carry step, and no holiday effects are used.

Backtest caveat: the test replays past forecasts using archived weather-model output, which is closer to what
happened than a genuine three-day-ahead weather forecast would have been. Day-3 errors in real use will be
somewhat larger than the backtest reports.

References
----------
Taylor, S. J. & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.
Hyndman, R. J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice, 3rd ed., ch. 5 and 10. OTexts.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json

for _name in ("cmdstanpy", "prophet"):
    logging.getLogger(_name).addHandler(logging.NullHandler())
    logging.getLogger(_name).setLevel(logging.WARNING)

REGRESSORS: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "rain_log",
    "wind_log",
    "wind_u",
    "wind_v",
)

_LOWER_QUANTILE: float = 0.10
_UPPER_QUANTILE: float = 0.90
_MIN_TRAINING_HOURS: int = 24 * 60
_DAY: int = 24


@dataclass
class StationModel:
    """A fitted Prophet model with the per-horizon carry and error-band offsets (all on the log scale)."""

    prophet: Prophet
    carry: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    trained_through: str

    @property
    def horizon(self) -> int:
        """The furthest number of hours ahead the carry and band were estimated for."""
        return len(self.carry)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the model's weather features to a frame holding the raw Open-Meteo columns, filling short gaps."""
    out = frame.copy()
    angle = np.deg2rad(out["wind_direction_10m"])
    out["rain_log"] = np.log1p(out["precipitation"].clip(lower=0.0))
    out["wind_log"] = np.log1p(out["wind_speed_10m"].clip(lower=0.0))
    out["wind_u"] = -out["wind_speed_10m"] * np.sin(angle)
    out["wind_v"] = -out["wind_speed_10m"] * np.cos(angle)
    columns = list(REGRESSORS)
    out[columns] = out[columns].interpolate(limit_direction="both")
    return out


def _prophet_input(frame: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    data = frame[["time", *REGRESSORS]].rename(columns={"time": "ds"})
    if target is not None:
        data["y"] = np.log1p(frame[target].to_numpy())
    return data


def usual_pattern(model: StationModel, frame: pd.DataFrame) -> np.ndarray:
    """Prophet's curve, on the log scale, for every row of a frame that has ``time`` and the feature columns."""
    return model.prophet.predict(_prophet_input(frame))["yhat"].to_numpy()


def carry_coefficients(residual: np.ndarray, horizon: int) -> np.ndarray:
    """Least-squares slope of r(t + h) on r(t) for h = 1..horizon, ignoring missing hours, clipped to [0, 1]."""
    carry = np.zeros(horizon)
    for step in range(1, horizon + 1):
        now, later = residual[:-step], residual[step:]
        both = ~np.isnan(now) & ~np.isnan(later)
        denominator = float(np.sum(now[both] ** 2))
        if denominator > 0.0:
            carry[step - 1] = float(np.sum(now[both] * later[both])) / denominator
    return np.clip(carry, 0.0, 1.0)


def error_band(residual: np.ndarray, carry: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 10th and 90th percentiles of the carried-forecast error at each horizon, on the log scale."""
    lower, upper = np.zeros(len(carry)), np.zeros(len(carry))
    for step in range(1, len(carry) + 1):
        error = residual[step:] - carry[step - 1] * residual[:-step]
        error = error[~np.isnan(error)]
        if error.size:
            lower[step - 1], upper[step - 1] = np.quantile(error, [_LOWER_QUANTILE, _UPPER_QUANTILE])
    return lower, upper


def _hourly_grid(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop_duplicates("time").set_index("time").sort_index().asfreq("h").reset_index()


def fit(frame: pd.DataFrame, target: str, horizon: int) -> StationModel:
    """
    Fit Prophet to one station's hourly table, then estimate the carry and error band from its residuals.

    ``frame`` needs ``time``, the target column and the raw weather columns. Hours with no reading are skipped
    by the Prophet fit and by the carry estimate.
    """
    data = add_features(_hourly_grid(frame))
    known = data.dropna(subset=[target])
    if len(known) < _MIN_TRAINING_HOURS:
        raise ValueError(f"Only {len(known)} hours with a reading; need at least {_MIN_TRAINING_HOURS} to train.")
    prophet = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        uncertainty_samples=0,
    )
    for name in REGRESSORS:
        prophet.add_regressor(name)
    prophet.fit(_prophet_input(known, target))
    model = StationModel(prophet, np.zeros(horizon), np.zeros(horizon), np.zeros(horizon), str(data["time"].max()))
    residual = np.log1p(data[target].to_numpy()) - usual_pattern(model, data)
    model.carry = carry_coefficients(residual, horizon)
    model.lower, model.upper = error_band(residual, model.carry)
    return model


def forecast(model: StationModel, history: pd.DataFrame, future: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Forecast the target for every hour in ``future``, anchored on the newest reading in ``history``.

    Both frames need ``time`` and the raw weather columns; ``history`` also needs the target. Hours further
    ahead of the newest reading than the model's horizon get no carry (pure Prophet) and the widest band.
    Returns ``time, <target>, lower, upper`` in real units.
    """
    future = add_features(future)
    pattern = usual_pattern(model, future)
    known = add_features(history).dropna(subset=[target])
    steps = np.full(len(future), model.horizon + 1)
    surprise = 0.0
    if len(known):
        anchor = known.iloc[[-1]]
        surprise = float(np.log1p(anchor[target].iloc[0]) - usual_pattern(model, anchor)[0])
        steps = ((future["time"] - anchor["time"].iloc[0]) / pd.Timedelta(hours=1)).round().astype(int).to_numpy()
    position = np.clip(steps, 1, model.horizon) - 1
    carried = pattern + np.where(steps <= model.horizon, model.carry[position], 0.0) * surprise
    return pd.DataFrame(
        {
            "time": future["time"].to_numpy(),
            target: _to_real(carried),
            "lower": _to_real(carried + model.lower[position]),
            "upper": _to_real(carried + model.upper[position]),
        }
    )


def _to_real(log_value: np.ndarray) -> np.ndarray:
    return np.clip(np.expm1(log_value), 0.0, None)


def backtest(frame: pd.DataFrame, target: str, test_days: int, horizon: int) -> pd.DataFrame:
    """
    Train on everything except the last ``test_days`` days, then score forecasts made from every test hour.

    Three forecasters are compared on mean absolute error (real units), day by day of lead time:

    * ``model``        Prophet plus the carry step (what the dashboard shows);
    * ``prophet``      Prophet alone, with no knowledge of the latest reading;
    * ``persistence``  "it will stay exactly as it is now", the standard yardstick.

    ``coverage`` is how often the truth fell inside the model's 80% band, and ``skill`` is the share of the
    persistence error the model removes (positive is better than persistence).
    """
    data = add_features(_hourly_grid(frame))
    cutoff = data["time"].max() - pd.Timedelta(days=test_days)
    model = fit(data[data["time"] <= cutoff], target, horizon)
    test = data[data["time"] > cutoff].reset_index(drop=True)
    truth = test[target].to_numpy()
    pattern = usual_pattern(model, test)
    residual = np.log1p(truth) - pattern
    rows: list[dict[str, float | str]] = []
    for first in range(1, horizon + 1, _DAY):
        errors: dict[str, list[np.ndarray]] = {"model": [], "prophet": [], "persistence": [], "inside": []}
        for step in range(first, min(first + _DAY, horizon + 1)):
            actual, later_pattern = truth[step:], pattern[step:]
            carried = later_pattern + model.carry[step - 1] * residual[:-step]
            scored = ~np.isnan(actual) & ~np.isnan(carried)
            low = _to_real(carried + model.lower[step - 1])
            high = _to_real(carried + model.upper[step - 1])
            errors["model"].append(np.abs(_to_real(carried) - actual)[scored])
            errors["prophet"].append(np.abs(_to_real(later_pattern) - actual)[scored])
            errors["persistence"].append(np.abs(truth[:-step] - actual)[scored])
            errors["inside"].append(((actual >= low) & (actual <= high))[scored].astype(float))
        merged = {name: np.concatenate(parts) for name, parts in errors.items()}
        if not merged["model"].size:
            continue
        mae_model, mae_persistence = float(merged["model"].mean()), float(merged["persistence"].mean())
        rows.append(
            {
                "lead": f"{first}-{min(first + _DAY - 1, horizon)} h",
                "forecasts": int(merged["model"].size),
                "mae_model": mae_model,
                "mae_prophet": float(merged["prophet"].mean()),
                "mae_persistence": mae_persistence,
                "skill": 1.0 - mae_model / mae_persistence if mae_persistence > 0 else float("nan"),
                "coverage": float(merged["inside"].mean()),
            }
        )
    return pd.DataFrame(rows)


def save(model: StationModel, path: Path) -> None:
    """Write a fitted model to a JSON file."""
    payload = {
        "prophet": model_to_json(model.prophet),
        "carry": model.carry.tolist(),
        "lower": model.lower.tolist(),
        "upper": model.upper.tolist(),
        "trained_through": model.trained_through,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def load(path: Path) -> StationModel:
    """Read a fitted model back from a JSON file written by ``save``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return StationModel(
        model_from_json(payload["prophet"]),
        np.array(payload["carry"]),
        np.array(payload["lower"]),
        np.array(payload["upper"]),
        payload["trained_through"],
    )

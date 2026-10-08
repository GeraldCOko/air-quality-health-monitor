from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aqmonitor import forecast

_HORIZON = 6
_SEED = 7


def _weather(hours: int, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(_SEED)
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=hours, freq="h"),
            "temperature_2m": 10.0 + rng.normal(0.0, 3.0, hours),
            "relative_humidity_2m": 70.0 + rng.normal(0.0, 5.0, hours),
            "surface_pressure": 1010.0 + rng.normal(0.0, 4.0, hours),
            "precipitation": rng.choice([0.0, 0.0, 0.0, 1.5], hours),
            "wind_speed_10m": rng.gamma(2.0, 6.0, hours),
            "wind_direction_10m": rng.uniform(0.0, 360.0, hours),
        }
    )


def _setup(hours: int = 24 * 70) -> pd.DataFrame:
    frame = _weather(hours)
    rng = np.random.default_rng(_SEED + 1)
    noise = np.zeros(hours)
    for index in range(1, hours):
        noise[index] = 0.9 * noise[index - 1] + rng.normal(0.0, 0.15)
    daily = 0.3 * np.sin(2.0 * np.pi * frame["time"].dt.hour / 24.0)
    frame["pm25"] = np.expm1(1.6 + daily - 0.02 * frame["wind_speed_10m"] + noise).clip(lower=0.0)
    return frame


@pytest.fixture(scope="module")
def fitted() -> tuple[forecast.StationModel, pd.DataFrame]:
    frame = _setup()
    return forecast.fit(frame, "pm25", _HORIZON), frame


def test_add_features_north_wind_blows_towards_south() -> None:
    frame = _weather(3).assign(wind_direction_10m=0.0, wind_speed_10m=10.0)
    assert np.allclose(forecast.add_features(frame)["wind_v"], -10.0)


def test_add_features_west_wind_blows_towards_east() -> None:
    frame = _weather(3).assign(wind_direction_10m=270.0, wind_speed_10m=10.0)
    assert np.allclose(forecast.add_features(frame)["wind_u"], 10.0)


def test_add_features_fills_missing_weather() -> None:
    frame = _weather(5)
    frame.loc[2, "temperature_2m"] = np.nan
    assert forecast.add_features(frame)[list(forecast.REGRESSORS)].notna().all().all()


def test_carry_coefficients_recover_autoregressive_decay() -> None:
    rng = np.random.default_rng(_SEED)
    residual = np.zeros(20000)
    for index in range(1, len(residual)):
        residual[index] = 0.8 * residual[index - 1] + rng.normal()
    assert np.allclose(forecast.carry_coefficients(residual, 3), [0.8, 0.64, 0.512], atol=0.03)


def test_carry_coefficients_ignore_missing_hours() -> None:
    residual = np.array([1.0, np.nan, 1.0, 1.0, np.nan, 1.0, 1.0])
    assert forecast.carry_coefficients(residual, 1)[0] == 1.0


def test_carry_coefficients_negative_slope_is_clipped_to_zero() -> None:
    residual = np.array([1.0, -1.0] * 50)
    assert forecast.carry_coefficients(residual, 1)[0] == 0.0


def test_error_band_lower_offset_is_below_upper_offset() -> None:
    residual = np.random.default_rng(_SEED).normal(size=5000)
    lower, upper = forecast.error_band(residual, np.zeros(4))
    assert np.all(lower < upper)


def test_error_band_white_noise_matches_normal_quantiles() -> None:
    residual = np.random.default_rng(_SEED).normal(size=50000)
    lower, upper = forecast.error_band(residual, np.zeros(1))
    assert np.allclose([lower[0], upper[0]], [-1.2816, 1.2816], atol=0.05)


def test_fit_too_little_data_raises() -> None:
    with pytest.raises(ValueError, match="need at least"):
        forecast.fit(_setup(24 * 10), "pm25", _HORIZON)


def test_fit_carry_fades_with_lead_time(fitted: tuple[forecast.StationModel, pd.DataFrame]) -> None:
    model, _ = fitted
    assert model.carry[0] > model.carry[-1] > 0.0


def test_fit_learns_that_wind_lowers_pm25(fitted: tuple[forecast.StationModel, pd.DataFrame]) -> None:
    model, frame = fitted
    calm = frame.tail(24).assign(wind_speed_10m=1.0)
    windy = frame.tail(24).assign(wind_speed_10m=40.0)
    assert forecast.usual_pattern(model, forecast.add_features(calm)).mean() > (
        forecast.usual_pattern(model, forecast.add_features(windy)).mean()
    )


def test_forecast_first_hour_stays_close_to_latest_reading(fitted: tuple[forecast.StationModel, pd.DataFrame]) -> None:
    model, frame = fitted
    history = frame.iloc[:-_HORIZON].copy()
    history.loc[history.index[-1], "pm25"] = 60.0
    result = forecast.forecast(model, history, frame.iloc[-_HORIZON:], "pm25")
    assert result["pm25"].iloc[0] > 30.0


def test_forecast_band_contains_central_forecast(fitted: tuple[forecast.StationModel, pd.DataFrame]) -> None:
    model, frame = fitted
    result = forecast.forecast(model, frame.iloc[:-_HORIZON], frame.iloc[-_HORIZON:], "pm25")
    assert ((result["lower"] <= result["pm25"]) & (result["pm25"] <= result["upper"])).all()


def test_forecast_is_never_negative(fitted: tuple[forecast.StationModel, pd.DataFrame]) -> None:
    model, frame = fitted
    result = forecast.forecast(model, frame.iloc[:-_HORIZON], frame.iloc[-_HORIZON:], "pm25")
    assert (result[["pm25", "lower", "upper"]] >= 0.0).all().all()


def test_forecast_without_any_reading_falls_back_to_usual_pattern(
    fitted: tuple[forecast.StationModel, pd.DataFrame],
) -> None:
    model, frame = fitted
    future = frame.iloc[-_HORIZON:]
    blank = frame.iloc[:-_HORIZON].assign(pm25=np.nan)
    expected = np.expm1(forecast.usual_pattern(model, forecast.add_features(future)))
    assert np.allclose(forecast.forecast(model, blank, future, "pm25")["pm25"], expected.clip(min=0.0))


def test_forecast_old_reading_beyond_horizon_has_no_sway(fitted: tuple[forecast.StationModel, pd.DataFrame]) -> None:
    model, frame = fitted
    future = frame.iloc[-_HORIZON:]
    stale = frame.iloc[: -_HORIZON - 48].copy()
    stale.loc[stale.index[-1], "pm25"] = 500.0
    expected = np.expm1(forecast.usual_pattern(model, forecast.add_features(future)))
    assert np.allclose(forecast.forecast(model, stale, future, "pm25")["pm25"], expected.clip(min=0.0))


def test_backtest_returns_one_row_per_day_of_lead_time() -> None:
    scores = forecast.backtest(_setup(24 * 80), "pm25", test_days=10, horizon=48)
    assert scores["lead"].tolist() == ["1-24 h", "25-48 h"]


def test_backtest_model_beats_prophet_alone_at_short_lead() -> None:
    scores = forecast.backtest(_setup(24 * 80), "pm25", test_days=10, horizon=24)
    assert scores.loc[0, "mae_model"] < scores.loc[0, "mae_prophet"]


def test_save_then_load_gives_same_forecast(fitted: tuple[forecast.StationModel, pd.DataFrame], tmp_path: Path) -> None:
    model, frame = fitted
    path = tmp_path / "model.json"
    forecast.save(model, path)
    history, future = frame.iloc[:-_HORIZON], frame.iloc[-_HORIZON:]
    original = forecast.forecast(model, history, future, "pm25")
    assert np.allclose(forecast.forecast(forecast.load(path), history, future, "pm25")["pm25"], original["pm25"])

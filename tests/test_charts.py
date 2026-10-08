import pandas as pd

from aqmonitor import charts


def _setup(peak: float = 6.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.DataFrame({"time": pd.date_range("2026-10-06", periods=24, freq="h"), "pm25": [4.0] * 24})
    predicted = pd.DataFrame(
        {
            "time": pd.date_range("2026-10-07", periods=72, freq="h"),
            "pm25": [peak] * 72,
            "lower": [peak * 0.5] * 72,
            "upper": [peak * 1.5] * 72,
        }
    )
    return observed, predicted


def _threshold_labels(figure: charts.go.Figure) -> list[str]:
    return [annotation.text for annotation in figure.layout.annotations]


def test_forecast_figure_names_measured_and_forecast_series() -> None:
    names = [trace.name for trace in charts.forecast_figure(*_setup(), "pm25").data if trace.showlegend is not False]
    assert names == ["Measured", "Likely range (8 in 10)", "Forecast"]


def test_forecast_figure_has_single_y_axis() -> None:
    figure = charts.forecast_figure(*_setup(), "pm25")
    assert {trace.yaxis for trace in figure.data} == {None}


def test_forecast_figure_clean_air_shows_only_first_threshold() -> None:
    assert _threshold_labels(charts.forecast_figure(*_setup(), "pm25")) == ["Moderate from 9.1"]


def test_forecast_figure_smoky_air_shows_higher_thresholds() -> None:
    labels = _threshold_labels(charts.forecast_figure(*_setup(peak=60.0), "pm25"))
    assert "Unhealthy from 55.5" in labels


def test_forecast_figure_without_forecast_still_draws_measurements() -> None:
    observed, predicted = _setup()
    assert len(charts.forecast_figure(observed, predicted.iloc[:0], "pm25").data) == 1


def test_bar_figure_plots_one_bar_per_row() -> None:
    summary = pd.DataFrame({"weather": ["Dry hours", "Wet hours"], "average": [6.0, 3.0], "hours": [900, 100]})
    assert list(charts.bar_figure(summary, "weather", "").data[0].y) == [6.0, 3.0]


def test_accuracy_figure_compares_three_forecasters() -> None:
    scores = pd.DataFrame(
        {"lead": ["1-24 h"], "mae_model": [1.0], "mae_persistence": [1.5], "mae_prophet": [2.0]},
    )
    assert len(charts.accuracy_figure(scores).data) == 3

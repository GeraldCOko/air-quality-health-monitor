"""
Build the dashboard's Plotly charts.

House rules, so every chart reads the same way: one y-axis per chart; thin lines; a legend whenever there is more
than one series; colour marks identity (blue = measured, orange = forecast) and is never the only cue because
every series is also named; AQI categories appear as labelled threshold lines rather than colour alone.
"""

import pandas as pd
import plotly.graph_objects as go

from aqmonitor import aqi

MEASURED: str = "#2a78d6"
FORECAST: str = "#eb6834"
THIRD: str = "#1baf7a"
_BAND: str = "rgba(235, 104, 52, 0.18)"
_MUTED: str = "#898781"
_UNIT: str = "µg/m³"


def _layout(figure: go.Figure, y_title: str, height: int) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "traceorder": "normal"},
        yaxis={"title": y_title, "rangemode": "tozero"},
    )
    return figure


def forecast_figure(
    observed: pd.DataFrame, predicted: pd.DataFrame, parameter: str, observed_name: str = "Measured"
) -> go.Figure:
    """
    Measured readings followed by the forecast and its 80% band.

    Both frames need a ``time`` column (already in the timezone to display) and the parameter column;
    ``predicted`` also needs ``lower`` and ``upper``.
    """
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=observed["time"],
            y=observed[parameter],
            name=observed_name,
            mode="lines",
            line={"color": MEASURED, "width": 2},
            hovertemplate="%{y:.1f} " + _UNIT,
        )
    )
    if len(predicted):
        figure.add_trace(
            go.Scatter(
                x=predicted["time"],
                y=predicted["upper"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=predicted["time"],
                y=predicted["lower"],
                name="Likely range (8 in 10)",
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=_BAND,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=predicted["time"],
                y=predicted[parameter],
                name="Forecast",
                mode="lines",
                line={"color": FORECAST, "width": 2},
                customdata=predicted[["lower", "upper"]],
                hovertemplate="%{y:.1f} " + _UNIT + " (likely %{customdata[0]:.1f} to %{customdata[1]:.1f})",
            )
        )
    top = max(observed[parameter].max(skipna=True), predicted["upper"].max(skipna=True) if len(predicted) else 0.0)
    for band in aqi.CATEGORIES[1:]:
        if band.conc_low <= max(top * 1.15, aqi.CATEGORIES[1].conc_low * 1.3):
            figure.add_hline(
                y=band.conc_low,
                line={"color": _MUTED, "width": 1, "dash": "dot"},
                annotation_text=f"{band.name} from {band.conc_low:g}",
                annotation_position="top left",
                annotation_font={"size": 11, "color": _MUTED},
            )
    figure.update_layout(hovermode="x unified")
    return _layout(figure, f"PM2.5 ({_UNIT})", 420)


def bar_figure(summary: pd.DataFrame, category: str, x_title: str) -> go.Figure:
    """One bar per row of a summary with an ``average`` and an ``hours`` column."""
    figure = go.Figure(
        go.Bar(
            x=summary[category].astype(str),
            y=summary["average"],
            marker={"color": MEASURED, "cornerradius": 4},
            width=0.6,
            customdata=summary[["hours"]],
            hovertemplate="%{x}: %{y:.1f} " + _UNIT + " on average<br>%{customdata[0]:,} hours<extra></extra>",
        )
    )
    figure.update_xaxes(title=x_title, type="category")
    return _layout(figure, f"Average PM2.5 ({_UNIT})", 300)


def accuracy_figure(scores: pd.DataFrame) -> go.Figure:
    """Grouped bars of forecast error by lead time for the model, Prophet alone, and persistence."""
    series = [
        ("mae_model", "This model", MEASURED),
        ("mae_persistence", "Guess: stays as it is now", FORECAST),
        ("mae_prophet", "Prophet without latest reading", THIRD),
    ]
    figure = go.Figure()
    for column, name, colour in series:
        figure.add_trace(
            go.Bar(
                x=scores["lead"],
                y=scores[column],
                name=name,
                marker={"color": colour, "cornerradius": 4},
                hovertemplate=name + ": off by %{y:.2f} " + _UNIT + " on average<extra></extra>",
            )
        )
    figure.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    figure.update_xaxes(title="How far ahead", type="category")
    return _layout(figure, f"Average error ({_UNIT})", 320)

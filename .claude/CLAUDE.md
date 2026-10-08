# CLAUDE.md

## Project

Air quality and health monitor for Nova Scotia. Downloads hourly PM2.5 from OpenAQ and weather from Open-Meteo, trains
one Prophet model per station, forecasts 72 hours ahead and shows a map, forecasts and health advice in a Streamlit
dashboard. The owner is learning; explanations should be plain and step by step.

## Architecture

| File                    | Role                                                                           |
| ----------------------- | ------------------------------------------------------------------------------ |
| `aqmonitor/config.py`   | Paths, parameter, history start, forecast and backtest lengths, API key lookup |
| `aqmonitor/region.py`   | Region outline and point-in-polygon test                                       |
| `aqmonitor/places.py`   | Places with no monitor (Antigonish): inverse-distance blend, leave-one-out     |
| `aqmonitor/openaq.py`   | OpenAQ v3 API client and S3 archive listing/download                           |
| `aqmonitor/weather.py`  | Open-Meteo historical-forecast and forecast fetchers                           |
| `aqmonitor/dataset.py`  | Read archive files, clean to an hourly grid, join weather                      |
| `aqmonitor/forecast.py` | Prophet fit, residual carry, error band, backtest, save/load                   |
| `aqmonitor/live.py`     | Live refresh: readings + weather forecast -> 72 h forecast files               |
| `aqmonitor/aqi.py`      | PM2.5 -> US EPA AQI number, category, colour, advice                           |
| `aqmonitor/insights.py` | Daily health outlook and weather-vs-PM2.5 summaries                            |
| `aqmonitor/mapping.py`  | Folium station map                                                             |
| `aqmonitor/charts.py`   | Plotly figures                                                                 |
| `app.py`                | Streamlit layout only; reads files written by the scripts                      |
| `scripts/*.py`          | CLIs run as `python -m scripts.<name>`; shared helpers in `scripts/cli.py`     |
| `tests/test_<module>`   | One test file per library module                                               |

## Key design decisions

- All timestamps are timezone-naive UTC and mark the END of the hour; conversion to local time happens only in `app.py`.
- Training history comes from the public OpenAQ archive (no key); only station discovery and the last few days use the
  API. The project must keep working, clearly labelled, when no key is set.
- The model is Prophet on log1p(PM2.5) with weather regressors, plus a per-horizon carry of the latest residual. Prophet
  alone ignores the current reading and loses to persistence at short leads; the carry fixes that.
- Uncertainty bands are empirical residual quantiles per horizon, not Prophet's simulated intervals.
- The backtest hides the last 90 days and always reports persistence alongside the model.
- Short gaps (3 hours or less) are interpolated; longer gaps stay missing.
- Antigonish has no monitor. It is a "place" (id 900001, `kind == "estimate"`) whose PM2.5 is an inverse-distance blend
  of the two nearest reference monitors (Pictou and Port Hawkesbury) with its own local weather and its own model. It is
  added to the station list in memory by `places.with_places`, never written to `data/stations.csv`. Every screen and
  message must label it as an estimate.
- Live readings use `/sensors/{id}/measurements`, not `/hours`: the hourly endpoint returns nothing for Pictou although
  it is reporting. OpenAQ's location search also omits Pictou, so `find_stations` keeps known stations whose sensor
  still reports.
- The API key lives only in `.env`, which is git-ignored. Never print it or write it anywhere else.

## Style

- Python 3.12+, type hints on every signature, line length 120. Tools: black, isort (profile black), flake8 with
  max-complexity 10, codespell, mdformat with the GFM plugin. Run `format` then `verify` before finishing.
- Long module docstrings explain the method, the maths, deviations from references, and list References.
- No inline comments unless the why is non-obvious. Private names start with `_`.
- Tests: small pytest functions named `test_<function>_<condition>_<expected>`, usually one assertion, seeded RNG, a
  `_setup` helper. No network in tests; HTTP is faked with a stub session.

"""
Step 4: test the forecaster on the most recent 90 days, then train the final model for every station.

For each station the script first hides the last ``BACKTEST_DAYS`` days, trains on the rest and scores forecasts
against what really happened (the backtest). It then retrains on all the data and saves that model to
``models/``. Scores for every station go to ``models/metrics.csv``.

Usage:  python -m scripts.train [--skip-backtest]
"""

import argparse
import time

import pandas as pd

from aqmonitor import config, forecast, live
from scripts import cli


def _train_station(name: str, station_id: int, frame: pd.DataFrame, run_backtest: bool) -> pd.DataFrame | None:
    started = time.time()
    scores = None
    if run_backtest:
        scores = forecast.backtest(frame, config.PARAMETER, config.BACKTEST_DAYS, config.FORECAST_HOURS)
        scores.insert(0, "station_id", station_id)
        scores.insert(1, "name", name)
    model = forecast.fit(frame, config.PARAMETER, config.FORECAST_HOURS)
    forecast.save(model, live.model_path(station_id))
    cli.say(f"  {name}: trained through {model.trained_through} in {time.time() - started:.0f} s")
    return scores


def _report(metrics: pd.DataFrame) -> None:
    cli.say("\nBacktest: average error in µg/m³ over the last 90 days (smaller is better)\n")
    cli.say(f"  {'lead time':<10} {'model':>7} {'Prophet alone':>14} {'persistence':>12} {'band hit rate':>14}")
    for lead, group in metrics.groupby("lead", sort=False):
        weights = group["forecasts"]
        average = {
            column: (group[column] * weights).sum() / weights.sum()
            for column in ("mae_model", "mae_prophet", "mae_persistence", "coverage")
        }
        cli.say(
            f"  {lead:<10} {average['mae_model']:>7.2f} {average['mae_prophet']:>14.2f} "
            f"{average['mae_persistence']:>12.2f} {average['coverage']:>13.0%}"
        )


def main() -> None:
    """Backtest and train a model for every station that has enough data."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-backtest", action="store_true", help="only train the final models (faster)")
    arguments = parser.parse_args()
    stations, table = cli.load_stations(), cli.load_table()
    cli.say(f"Training {len(stations)} station models (about a minute or two each) ...")
    all_scores: list[pd.DataFrame] = []
    for station in stations.itertuples():
        frame = table[table["station_id"] == station.station_id]
        try:
            scores = _train_station(station.name, int(station.station_id), frame, not arguments.skip_backtest)
        except ValueError as error:
            cli.say(f"  {station.name}: skipped ({error})")
            continue
        if scores is not None:
            all_scores.append(scores)
    if all_scores:
        metrics = pd.concat(all_scores, ignore_index=True)
        metrics.to_csv(config.METRICS_FILE, index=False)
        _report(metrics)
    cli.say("\nDone. Next:  python -m scripts.update_live")


if __name__ == "__main__":
    main()

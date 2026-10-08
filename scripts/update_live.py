"""
Step 5: fetch the newest readings and weather forecast, and make a fresh 72-hour forecast for every station.

Run this whenever you want up-to-date numbers; the dashboard's "Refresh now" button runs the same thing.

Usage:  python -m scripts.update_live
"""

from aqmonitor import live
from scripts import cli


def main() -> None:
    """Refresh ``data/live/`` and print a short status."""
    stations, table = cli.load_stations(), cli.load_table()
    client = cli.client_or_none()
    if client is None:
        cli.say("No OpenAQ API key in .env, so readings come from the archive and are several days old.")
    cli.say("Fetching readings and weather, then forecasting ...")
    status = live.refresh(stations, table, client)
    cli.say(f"Updated {status['updated_utc']} UTC from: {status['source']}")
    cli.say(f"Forecasts made for {status['stations_with_forecast']} of {len(stations)} stations.")
    for problem in status["problems"]:
        cli.say(f"  note: {problem}")
    cli.say("\nDone. Open the dashboard with:  python -m streamlit run app.py")


if __name__ == "__main__":
    main()

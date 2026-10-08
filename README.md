# Air Quality & Health Monitor

Pulls live air-quality sensor data from OpenAQ for Nova Scotia, maps pollution levels, and forecasts fine-particle
pollution (PM2.5) for the next 72 hours with a Prophet time-series model that also uses the weather forecast. It also
estimates and forecasts the air in Antigonish, which has no monitor of its own.

## What it does, in plain words

- **Collects.** Government air monitors in Nova Scotia measure tiny smoke and dust particles (PM2.5) every hour. OpenAQ
  gathers those readings; this project downloads them.

- **Maps.** Each station is a dot on a map, coloured from green (clean) to maroon (hazardous).

- **Forecasts.** A model learns each station's usual pattern (time of day, day of week, season, wind, rain) and predicts
  the next three days, with a shaded "likely range" so you can see how sure it is.

- **Advises.** Each forecast is turned into an Air Quality Index category with plain health advice.

- **Links to weather.** The optional extra: weather from Open-Meteo feeds the forecast, and the dashboard shows how wind
  and rain relate to pollution at each station.

- **Estimates Antigonish.** Nobody measures the air in Antigonish, so the project blends the two nearest monitors to
  estimate it, then forecasts that estimate with Antigonish's own weather. See "Antigonish" below.

## Setup

Everything is already installed inside this folder. You only need these steps on a new computer.

1. Install Python 3.12 or newer from python.org.

1. Open a terminal **in this folder**: in File Explorer, open the folder, click the address bar, type `powershell` and
   press Enter.

1. Create the project's private Python environment and install the libraries:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
   ```

1. Get a free OpenAQ API key at <https://explore.openaq.org/register>. After you log in, the key is on your account
   page. Open the file `.env` in this folder with Notepad and paste the key after the `=` sign:

   ```text
   OPENAQ_API_KEY=paste-your-key-here
   ```

   Save the file. The key is like a password: keep it on your computer and do not share it.

Without a key the project still works, but the readings come from OpenAQ's archive, which is several days behind.

## How to run it

### The easy way

Double-click **`start_dashboard.bat`**. A black window opens and, a few seconds later, your browser shows the dashboard
at <http://localhost:8501>. Press **Refresh now** on the dashboard to fetch the newest readings and make a new forecast.
Close the black window to stop.

### The step-by-step way

Open a terminal in this folder (see Setup, step 2) and switch on the project's Python environment. You do this once each
time you open a new terminal; your prompt then starts with `(.venv)`:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses with a message about "running scripts is disabled", skip activation and type
`.\.venv\Scripts\python.exe` wherever the commands below say `python`.

Then run the steps in order. Each one prints what to run next.

| Step | Command                           | What it does                                                           | How long       |
| ---- | --------------------------------- | ---------------------------------------------------------------------- | -------------- |
| 1    | `python -m scripts.find_stations` | Asks OpenAQ which stations are in Nova Scotia. Needs the API key.      | seconds        |
| 2    | `python -m scripts.download_data` | Downloads hourly PM2.5 since 2022 and matching weather. No key needed. | 5-10 minutes   |
| 3    | `python -m scripts.build_dataset` | Cleans the downloads and joins air quality with weather.               | about a minute |
| 4    | `python -m scripts.train`         | Tests the forecaster on the last 90 days, then trains the real models. | 10-20 minutes  |
| 5    | `python -m scripts.update_live`   | Fetches the newest readings and weather; makes the 72-hour forecast.   | under a minute |
| 6    | `python -m streamlit run app.py`  | Opens the dashboard. Then visit <http://localhost:8501>.               | stays running  |

Steps 1 to 5 have already been run for you, so the data and trained models are in place. Day to day you only need step 6
(or the `.bat` file) and the **Refresh now** button.

### Keeping it fresh

- **Every time you look:** press **Refresh now** in the dashboard (same as step 5).
- **Every week or two:** run `python -m scripts.run_all`. It downloads the newest history, rebuilds the table, retrains
  the models and refreshes the forecast. Files already downloaded are skipped.

## Reading the dashboard

- **Map.** Colour shows the category of each station's newest reading. Grey means the station has not reported in the
  last 6 hours. Hover for the number; click for details.
- **Forecast and health.** The blue line is what was measured over the last 7 days. The orange line is the forecast, and
  the shaded band is where the model expects 8 readings out of 10 to land. Dotted lines mark where a worse category
  begins. The health outlook gives each of the next three days a category and advice.
- **Weather link.** Average pollution by wind speed, by wind direction, and in dry versus wet hours.
- **Forecast accuracy.** How far off the model was when tested on 90 days it had never seen, next to the simple guess
  "it will stay as it is now". If the model's bar is shorter, the model is adding something.

## How the forecast works

Think of the model as two helpers working together.

1. **The pattern-spotter (Prophet).** It studies years of readings and learns the usual shape: mornings versus nights,
   weekdays versus weekends, summer versus winter, and what wind, rain, temperature, humidity and pressure usually do to
   the air. Given a date, a time and a weather forecast, it says what is *typical*.
1. **The reality check.** The pattern-spotter cannot see that wildfire smoke rolled in an hour ago. So the model looks
   at the newest reading, measures how far it is from "typical", and carries that surprise forward. The surprise fades
   hour by hour, because today's smoke says a lot about the next hour and little about three days from now. How fast it
   fades is learned from the data.

The shaded band comes from the mistakes the model actually made in the past at each lead time. The full explanation,
with the formulas, is at the top of `aqmonitor/forecast.py`.

## How well it does

Tested on the 90 most recent days, which the model never saw during training. The numbers are the average miss in µg/m³
across the six stations and Antigonish (typical readings are 3 to 5 µg/m³), so smaller is better.

| How far ahead | This model | Guess "stays as it is now" | Prophet without the reality check |
| ------------- | ---------- | -------------------------- | --------------------------------- |
| 1-24 hours    | 1.57       | 1.67                       | 2.01                              |
| 25-48 hours   | 1.96       | 2.43                       | 2.01                              |
| 49-72 hours   | 1.99       | 2.58                       | 2.00                              |

The model beats the simple guess a little on day one and by about a fifth on days two and three. The truth landed inside
the shaded band 76-80% of the time, close to the 80% it aims for. The test replayed archived weather-model data, which
is more accurate than a real three-day weather forecast, so expect real day-2 and day-3 errors to be somewhat larger.
Re-running `python -m scripts.train` updates these numbers in `models/metrics.csv`.

## Antigonish: a place with no monitor

Antigonish has no air monitor. It sits between two that do: Port Hawkesbury, 49 km to the east, and Pictou, 55 km to the
west. Most of the fine particles in Nova Scotia's air arrive on the wind from far away (ocean air, air from the
continent, wildfire smoke), so places 50 km apart usually see similar air.

**How the estimate is made.** For every hour, the project takes the two readings and mixes them, giving the nearer
monitor a bigger say: 56% Port Hawkesbury and 44% Pictou. If one monitor is silent for an hour, the other is used alone.
This is called inverse-distance weighting.

**How the forecast is made.** Exactly like a real station: the estimated history is paired with the weather at
Antigonish itself, a Prophet model learns the pattern, and the newest estimate keeps the forecast in touch with what is
happening now. In the dashboard, Antigonish is the first choice in the list and its dot on the map has a dashed ring to
show it is an estimate.

**How far to trust it.** Two honest checks, both in `python -m scripts.check_estimate`:

- Pictou and Port Hawkesbury agree fairly well day by day (correlation 0.78) and less well hour by hour (0.60). On
  average they differ by 1.6 µg/m³.
- Each real monitor was hidden in turn and estimated from its two nearest neighbours in the same way. The estimate
  missed the true daily average by about 0.9 to 1.0 µg/m³ and the true hourly value by 1.2 to 1.5 µg/m³, where typical
  readings are 3 to 5. Those neighbours were 58 to 156 km away, further than Antigonish's, so Antigonish should do no
  worse.

So the estimate is a good guide to whether a day is clean or smoky, and a rougher guide to any single hour. It cannot
see anything local to Antigonish, such as wood-stove smoke on a still winter night or dust from a nearby road. The
forecast scores for Antigonish (day 1: 1.58 µg/m³ average miss, against 1.66 for the "stays as it is now" guess) are
measured against the estimate, not against air actually measured in Antigonish.

To add another place without a monitor, add a line to `PLACES` in `aqmonitor/places.py` and run
`python -m scripts.run_all`.

## What it cannot do

- **Wildfire smoke arriving from far away** is the main cause of very bad air days in Nova Scotia, and nothing in the
  model's inputs can see smoke before it arrives. The forecast reacts once the first smoky reading comes in.
- **Hospital admissions are not linked.** No public source gives daily hospital admissions for Nova Scotia, so that
  optional extra was left out and weather was linked instead.
- **The history has holes.** OpenAQ's archive is missing about half of 2022 and a third of 2023 for every Nova Scotia
  station, including one shared gap of 107 days. From 2024 on it is about 98% complete. Gaps longer than 3 hours are
  left empty and never filled with made-up numbers.
- **The categories are a guide.** The official index is defined on 24-hour averages; applying it to single hours, as the
  map does, is a simplification. This is a learning project, not medical advice.

## Project commands

| Command                            | What it does                                                        |
| ---------------------------------- | ------------------------------------------------------------------- |
| `python -m scripts.find_stations`  | Find active PM2.5 stations in the region; write `data/stations.csv` |
| `python -m scripts.download_data`  | Download OpenAQ archive files and Open-Meteo weather                |
| `python -m scripts.build_dataset`  | Build `data/processed/hourly.csv`                                   |
| `python -m scripts.train`          | Backtest and train; write `models/`                                 |
| `python -m scripts.update_live`    | Refresh `data/live/` with readings and forecasts                    |
| `python -m scripts.run_all`        | Steps 2 to 5 in one go                                              |
| `python -m scripts.check_estimate` | Test how good the Antigonish estimating method is                   |
| `python -m streamlit run app.py`   | Start the dashboard                                                 |
| `pytest`                           | Run the tests                                                       |
| `format`                           | Reformat code and Markdown (isort, black, mdformat)                 |
| `verify`                           | Check formatting, lint and spelling without changing files          |

## Where things are

| Path                  | What it holds                                                  |
| --------------------- | -------------------------------------------------------------- |
| `app.py`              | The dashboard                                                  |
| `aqmonitor/`          | The library: one file per job (see the top of each file)       |
| `scripts/`            | The commands in the tables above                               |
| `tests/`              | Tests, one file per library file                               |
| `data/stations.csv`   | The station list. Delete a row to drop a station.              |
| `data/raw/`           | Downloads exactly as received                                  |
| `data/processed/`     | The cleaned hourly table used for training                     |
| `data/live/`          | Newest readings and the current forecast                       |
| `models/`             | Trained models and their test scores (`metrics.csv`)           |
| `aqmonitor/config.py` | Settings: forecast length, history start date, test length     |
| `aqmonitor/region.py` | The outline of the monitored region                            |
| `aqmonitor/places.py` | Places with no monitor (Antigonish) and how they are estimated |
| `.env`                | Your private OpenAQ API key                                    |

## Data sources

- Air quality: [OpenAQ](https://openaq.org) (Nova Scotia Environment monitors, relayed through US EPA AirNow).
- Weather: [Open-Meteo](https://open-meteo.com).
- Health categories: US EPA Air Quality Index for PM2.5 (2024 breakpoints).

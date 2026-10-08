"""
Run the whole pipeline in one go: download new data, rebuild the table, retrain, and refresh the live forecast.

Use this every week or two to fold the newest history into the models. It expects ``data/stations.csv`` to
exist already (step 1, ``python -m scripts.find_stations``).

Usage:  python -m scripts.run_all
"""

import sys

from scripts import build_dataset, cli, download_data, train, update_live


def main() -> None:
    """Run steps 2 to 5 in order."""
    sys.argv = sys.argv[:1]
    for number, step in enumerate((download_data, build_dataset, train, update_live), start=2):
        cli.say(f"\n===== Step {number}: {step.__name__.split('.')[-1]} =====")
        step.main()


if __name__ == "__main__":
    main()

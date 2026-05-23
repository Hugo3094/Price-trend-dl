"""
main.py
-------
Entry point for the price-trend prediction pipeline.

Usage
-----
    uv run python main.py                         # full run
    uv run python main.py --config path/to.json   # custom config
"""

import argparse
import logging
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

from reimagining_trends.pipeline import Pipeline
from reimagining_trends.utils.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def main(config_path: str | None = None) -> None:
    Pipeline(Config(config_path)).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Price-trend pipeline")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to run_pipeline_config.json")
    args = parser.parse_args()
    main(config_path=args.config)
